"""VLM client with vision support.

Supports the OpenAI-compatible API served by vLLM, llama.cpp, or OpenRouter.
Handles multimodal messages with base64-encoded images.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("shrew.vlm")


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
            raise
        except requests.RequestException as e:
            logger.error(f"VLM request failed: {e}")
            raise

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

        # Strip thinking blocks that some providers inline in content
        if content.startswith("Thinking Process") or content.startswith("<think>"):
            for marker in ["\n\n---\n\n", "\n\nAnswer:\n", "\n\nSummary:\n",
                          "</think>\n", "\n\n**Summary"]:
                idx = content.find(marker)
                if idx > 0:
                    content = content[idx + len(marker):]
                    break

        return content.strip()

    def health_check(self, timeout: int = 5) -> bool:
        """Check if the VLM server is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models", timeout=timeout,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return True
        except Exception:
            return False
