"""VLM client with vision support.

Supports the OpenAI-compatible API served by vLLM, llama.cpp, or OpenRouter.
Handles multimodal messages with base64-encoded images.
"""

import base64
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Callable, Optional

import requests

logger = logging.getLogger("shrew.vlm")

# VLM concurrency gate. NOTE: this is PER-PROCESS, not cross-process — the
# original comment assumed uvicorn forks after import, but uvicorn SPAWNS
# workers, and each fresh interpreter builds its own independent semaphore.
# With SHREW_WORKERS=N the effective in-flight limit is N x VLM_CONCURRENCY;
# the server logs a loud warning at startup (audit issue #14).
_VLM_CONCURRENCY = int(os.environ.get("VLM_CONCURRENCY", "4"))
_vlm_gate = (
    multiprocessing.Semaphore(_VLM_CONCURRENCY) if _VLM_CONCURRENCY > 0 else None
)

# ── Cached readiness (fixes false-503-under-load, audit issue #17) ───────────
# A per-request inference probe queues behind real OCR under load and times
# out, wrongly declaring the model "unavailable" exactly when the server is
# busiest. Instead, a background task keeps a readiness stamp fresh and the
# per-request gate consults the CACHE. A merely slow/busy backend never
# rejects work (fail-open); only a DEFINITE fault (model absent from
# /v1/models, or an auth/config rejection) does.
READINESS_TTL_S = float(os.environ.get("VLM_READINESS_TTL_S", "60"))
# Whether the (background) deep probe includes a tiny inference. The
# per-request path NEVER runs inference regardless of this.
READINESS_PROBE_INFERENCE = os.environ.get(
    "VLM_READINESS_PROBE_INFERENCE", "1").lower() in ("1", "true", "yes")
# {key: {"ok": bool, "ts": float, "ever_ok": bool}}
_readiness: dict[str, dict] = {}


def _readiness_key(base_url: str, model: str) -> str:
    return f"{base_url}::{model}"


def _encode_image(image_path: Path | str, format: str = "png") -> str:
    """Encode an image file as a base64 data URI."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    mime = f"image/{format}"
    return f"data:{mime};base64,{data}"


def make_image_content(image_path: Path | str, detail: str = "high") -> dict:
    """Create an image_url content part for a multimodal message."""
    data_uri = _encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": data_uri, "detail": detail},
    }


class PinpointMismatchError(RuntimeError):
    """The served model config does not carry the tile-bucket pinpoint list.

    The processor cuts the tiles, but the MODEL unpads and packs features using
    `config.image_grid_pinpoints`. Patch the PROCESSOR only and every request
    fails (measured 40/40) with a feature/token count mismatch — an error whose
    text says nothing about the actual cause, which is why it is translated here.
    """


# vLLM surfaces the mismatch as a 400 whose body contains this phrase.
_PINPOINT_SIGNATURE = "image features and image tokens do not match"

_PINPOINT_HELP = (
    "The served model is missing the tile-bucket pinpoints. Serve a model "
    "directory whose config.json AND preprocessor_config.json both carry, in "
    '[height, width] order: "image_grid_pinpoints": '
    "[[1536,1152],[2304,1536],[3072,2304],[1152,1152]]. Patching only the "
    "processor fails EVERY request (SHREW_OCR_PREVIEW.md §2.1)."
)


def _translate_http_error(e: "requests.HTTPError") -> Exception:
    """Turn the opaque feature/token mismatch into the actionable message."""
    try:
        body = e.response.text or ""
    except Exception:
        body = ""
    if _PINPOINT_SIGNATURE in body.lower():
        return PinpointMismatchError(f"{_PINPOINT_HELP} (server said: {body[:300]})")
    return e


def make_text_content(text: str) -> dict:
    """Create a text content part."""
    return {"type": "text", "text": text}


class VLMClient:
    """Client for vLLM, llama.cpp, or OpenRouter with vision support."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        default_timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.default_timeout = default_timeout

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 8192,
        temperature: float = 0.2,
        timeout: Optional[int] = None,
        extra_params: Optional[dict] = None,
    ) -> dict:
        """Single chat completion call.

        Args:
            messages: OpenAI-format message list.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
            extra_params: Additional API params (top_p, top_k,
                presence_penalty, reasoning, etc.).

        Returns:
            Full response dict from the API.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_params:
            payload.update(extra_params)

        t = timeout or self.default_timeout
        start = time.time()

        gate = _vlm_gate
        if gate is not None:
            if not gate.acquire(block=False):
                logger.debug("VLM gate full, waiting for slot...")
                gate.acquire()
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=t,
            )
            resp.raise_for_status()
            result = resp.json()
            elapsed = time.time() - start

            usage = result.get("usage", {})
            logger.debug(
                f"VLM call: {elapsed:.1f}s, "
                f"prompt={usage.get('prompt_tokens', '?')}, "
                f"completion={usage.get('completion_tokens', '?')}"
            )
            return result

        except requests.Timeout:
            logger.error(f"VLM call timed out after {t}s")
            raise
        except requests.HTTPError as e:
            logger.error(f"VLM HTTP error: {e.response.status_code} - {e.response.text[:500]}")
            raise _translate_http_error(e) from e
        except requests.RequestException as e:
            logger.error(f"VLM request failed: {e}")
            raise
        finally:
            if gate is not None:
                gate.release()

    def chat_completion_stream(
        self,
        messages: list[dict],
        max_tokens: int = 8192,
        temperature: float = 0.2,
        timeout: Optional[int] = None,
        extra_params: Optional[dict] = None,
        on_delta: Optional[Callable[[str, str], Optional[str]]] = None,
        wall_clock_s: Optional[float] = None,
    ) -> dict:
        """Streaming chat completion, assembled into the non-streaming response shape.

        ``wall_clock_s`` bounds the stream duration FROM THE FIRST BYTE —
        scheduler queue wait must not convert healthy pages into aborts when
        the backend runs a deep queue; time-to-first-byte is still bounded by
        ``timeout``. ``timeout`` otherwise only
        bounds the gap between bytes, so a trickling generation never trips it.
        On expiry the stream aborts with ``finish_reason: "wall_clock_abort"``.

        Returns the same ``{"choices": [{"message": {...}, "finish_reason": ...}]}``
        dict as ``chat_completion``, so callers can treat the two identically.

        ``on_delta(chunk, accumulated)`` is invoked for every content delta. It
        normally returns None; returning a string aborts the request and that
        string becomes ``finish_reason``. This is how the §5.2 repetition guard
        stops a loop mid-generation instead of paying for the full output cap —
        closing the response releases the vLLM slot. The call holds the same
        cross-process concurrency gate as ``chat_completion`` for its whole
        duration: a streaming slot is still a slot.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if extra_params:
            payload.update(extra_params)

        t = timeout or self.default_timeout
        start = time.time()
        first_byte_at: Optional[float] = None
        parts: list[str] = []
        finish_reason = None
        aborted = None

        gate = _vlm_gate
        if gate is not None:
            if not gate.acquire(block=False):
                logger.debug("VLM gate full, waiting for slot...")
                gate.acquire()
        try:
            with requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=t,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if first_byte_at is None:
                        first_byte_at = time.time()
                    if (wall_clock_s is not None
                            and time.time() - first_byte_at > wall_clock_s):
                        aborted = "wall_clock_abort"
                        finish_reason = aborted
                        logger.warning(
                            f"wall_clock_abort after {time.time() - first_byte_at:.0f}s "
                            f"generating (+{first_byte_at - start:.0f}s queued, "
                            f"{len(''.join(parts))} chars emitted)")
                        break
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for choice in evt.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            parts.append(piece)
                            if on_delta is not None:
                                aborted = on_delta(piece, "".join(parts))
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                    if aborted:
                        # Abandon the response body; the context manager closes
                        # the connection, which cancels the generation server-side.
                        finish_reason = aborted
                        break

        except requests.Timeout:
            logger.error(f"VLM stream timed out after {t}s")
            raise
        except requests.HTTPError as e:
            logger.error(f"VLM HTTP error: {e.response.status_code} - {e.response.text[:500]}")
            raise _translate_http_error(e) from e
        except requests.RequestException as e:
            logger.error(f"VLM stream failed: {e}")
            raise
        finally:
            if gate is not None:
                gate.release()

        text = "".join(parts)
        logger.debug(f"VLM stream: {time.time() - start:.1f}s, "
                     f"{len(text)} chars, finish_reason={finish_reason}")
        return {
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }],
        }

    def simple_completion(
        self,
        system_prompt: str,
        user_content: list[dict] | str,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        timeout: Optional[int] = None,
        extra_params: Optional[dict] = None,
    ) -> str:
        """Simple single-turn completion without tools.

        Args:
            system_prompt: System message text.
            user_content: Either a string or a list of content parts (text + images).
            max_tokens: Max tokens.
            temperature: Temperature.
            extra_params: Additional API params.

        Returns:
            The assistant's text response.
        """
        messages = [{"role": "system", "content": system_prompt}]

        messages.append({"role": "user", "content": user_content})

        result = self.chat_completion(messages, max_tokens=max_tokens,
                                       temperature=temperature, timeout=timeout,
                                       extra_params=extra_params)
        choices = result.get("choices") or []
        if not choices:
            raise ValueError(f"VLM returned empty choices: {json.dumps(result)[:200]}")
        msg = choices[0]["message"]
        content = msg.get("content") or ""

        # Qwen 3.5 returns reasoning in a separate field. Only fall back to
        # reasoning if content is truly empty (not just thinking output).
        if not content.strip():
            reasoning = msg.get("reasoning", "")
            if reasoning:
                content = reasoning

        # Strip thinking blocks that some providers inline in content.
        # A trailing </think> indicates thinking leaked into the content
        # field; drop everything up to and including it regardless of how
        # the response started. Fall back to startswith-based markers for
        # providers that emit "Thinking Process" / "Answer:" style headers.
        end_think = content.rfind("</think>")
        if end_think >= 0:
            content = content[end_think + len("</think>"):]
        elif content.startswith("Thinking Process") or content.startswith("<think>"):
            for marker in ["\n\n---\n\n", "\n\nAnswer:\n", "\n\nSummary:\n",
                          "\n\n**Summary"]:
                idx = content.find(marker)
                if idx > 0:
                    content = content[idx + len(marker):]
                    break

        return content.strip()

    def probe(self, timeout: int = 10, do_inference: bool | None = None) -> tuple[str, str | None]:
        """Actively probe the backend and update the readiness cache.

        Returns (verdict, reason):
          "ok"   — reachable, model listed, (optional) inference succeeded.
          "hard" — a DEFINITE fault: model absent from /v1/models, or an
                   auth/config rejection (401/403/404). Admission should
                   reject on this.
          "soft" — reachable check failed transiently (timeout, connection
                   error, 5xx, inference slow). The backend may just be busy;
                   admission FAILS OPEN on this.

        Meant for the background refresher and monitoring — NOT the per-request
        path (see is_ready). do_inference defaults to READINESS_PROBE_INFERENCE.
        """
        if do_inference is None:
            do_inference = READINESS_PROBE_INFERENCE
        key = _readiness_key(self.base_url, self.model)
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models", timeout=timeout,
                headers=self._headers(),
            )
            if resp.status_code in (401, 403):
                logger.warning(f"Readiness probe: auth rejected ({resp.status_code})")
                self._mark_readiness(key, False)
                return "hard", f"auth {resp.status_code}"
            resp.raise_for_status()
            model_ids = {
                item.get("id")
                for item in resp.json().get("data", [])
                if isinstance(item, dict)
            }
            if self.model and self.model not in model_ids:
                logger.warning(f"Readiness probe: model '{self.model}' not in {model_ids}")
                self._mark_readiness(key, False)
                return "hard", "model not served"

            if do_inference:
                r2 = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json={"model": self.model,
                          "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 1, "temperature": 0},
                    timeout=timeout,
                )
                if r2.status_code in (401, 403, 404):
                    self._mark_readiness(key, False)
                    return "hard", f"inference {r2.status_code}"
                r2.raise_for_status()

            self._mark_readiness(key, True)
            return "ok", None
        except Exception as e:
            # Timeout / connection reset / 5xx — busy or transiently down, not
            # a definite fault. Leave a prior healthy stamp in place.
            logger.warning(f"Readiness probe soft-failed: {e}")
            return "soft", str(e)

    @staticmethod
    def _mark_readiness(key: str, ok: bool) -> None:
        prev = _readiness.get(key, {})
        _readiness[key] = {
            "ok": ok,
            "ts": time.time(),
            "ever_ok": prev.get("ever_ok", False) or ok,
        }

    def is_ready(self) -> bool:
        """Per-request admission gate. NEVER runs inference.

        Fast path: a fresh healthy cache stamp (kept warm by the background
        refresher) → admit. Cold/stale path: one lightweight /v1/models check
        (no inference). A soft failure fails OPEN when the backend was ever
        healthy — a busy backend must not reject queued work (issue #17).
        """
        key = _readiness_key(self.base_url, self.model)
        entry = _readiness.get(key)
        if entry and entry["ok"] and (time.time() - entry["ts"]) < READINESS_TTL_S:
            return True
        # Cache cold or stale — cheap reachability only, never inference.
        verdict, _reason = self.probe(timeout=5, do_inference=False)
        if verdict == "ok":
            return True
        if verdict == "hard":
            return False
        # soft: busy/transient. Admit if we have ever seen this backend healthy;
        # the pipeline's own per-page error handling covers a genuine outage.
        ever_ok = (entry or {}).get("ever_ok", False)
        if ever_ok:
            logger.warning(
                "Readiness probe slow/unreachable but backend was healthy "
                "before — admitting (fail-open); pipeline handles per-page errors")
        return ever_ok

    # Back-compat: monitoring callers that want a boolean deep check.
    def health_check(self, timeout: int = 10) -> bool:
        return self.probe(timeout=timeout)[0] == "ok"

    def readiness_snapshot(self) -> dict:
        """Cached readiness for the /health endpoint (no live inference)."""
        key = _readiness_key(self.base_url, self.model)
        entry = _readiness.get(key)
        if not entry:
            return {"ready": None, "age_s": None, "ever_ok": False}
        return {"ready": entry["ok"],
                "age_s": round(time.time() - entry["ts"], 1),
                "ever_ok": entry["ever_ok"]}
