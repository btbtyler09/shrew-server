"""Per-page structured extraction (shrew-ocr-preview, 5-key sentinel target).

Implements the serving contract in shrew_ocr/deploy/SHREW_OCR_PREVIEW.md.
Two input modalities, one output schema:

  * image — a page routed to a glyph-matched tile bucket by
    ``preprocess.prepare_image_bucketed`` (§2.1) and sent as a data-URI image
    part.
  * text  — messy HTML/markdown/code/plain text sent as a raw string. Same
    5-key output; ``figures[].bbox`` / ``tables[].bbox`` come back null.

Sampling is EXACTLY ``temperature 0, max_tokens 20000`` (§3.1). Gate F measured
penalty-free unconstrained greedy, so any decode knob here voids the published
eval numbers. Streaming is not a decode knob — it changes nothing about what the
model emits, only when we can see it, which is what lets the §5.2 guard abort a
loop instead of paying for the full cap. Because greedy is deterministic, an
identical retry reproduces an identical failure — the only retry allowed is ONE
attempt with ``structured_outputs`` enforcement, and any page that survives it
is flagged ``schema_coerced`` so downstream can filter or de-rank it.

# parse_json_lenient vendored from shrew_ocr/structured_eval/run_structured.py; validate_schema/SECTION_ENUM from shrew_ocr/structured_eval/metrics_v2.py — keep in sync.
"""

import json
import logging
import os
import re
import threading
import zlib
from pathlib import Path

from .generation import get_generation_params
from .vlm_client import make_image_content

logger = logging.getLogger("shrew.structured_page")

SENTINEL = "structured_extraction"

# §3.1 (e4, supersedes the 12000/16384 pair). `max_model_len` bounds prompt and
# output TOGETHER, and on the largest tile bucket the image dominates: at
# --max-model-len 16384 a B3 page could emit only 7,872 tokens, while the median
# correct newspaper-class label is 6,891 with a tail past 12k. That cap truncated
# real content, and the truncation is indistinguishable from a loop downstream —
# it was misdiagnosed as model truncation twice. Measured cost of the raise: none
# (KV cache 1,157,984 tokens, 36x concurrency headroom at TP=4).
MAX_TOKENS = int(os.environ.get("SHREW_MAX_TOKENS", "20000"))
# §4 serving recipe: vllm serve ... --max-model-len 32768
MAX_MODEL_LEN = int(os.environ.get("SHREW_MAX_MODEL_LEN", "32768"))

# §2.1 image-token cost per tile bucket, and the measured prompt-text overhead
# (sentinel + chat template + image placeholder scaffolding). Used to verify the
# output budget is actually REACHABLE on the largest bucket rather than merely
# configured — the silent-ceiling failure this constant pair exists to prevent.
BUCKET_IMAGE_TOKENS = {"B0": 1200, "B1": 1920, "B2": 3672, "B3": 7152}
PROMPT_TEXT_TOKENS = 1360
# Conservative chars-per-token. Real ratios on this corpus run 3.5-4.5;
# a low divisor over-estimates tokens, which fails safe.
CHARS_PER_TOKEN = 3.5

# Largest text page we will send. This is a DISTRIBUTION bound, not a capacity
# one. Measured over the 1000-row text-arm test split
# (shrew_structured_v2/data/test.text_reassembled.manifest1000.jsonl):
#
#     input chars   median 2286   p90 4197   p99 6996   max 11324
#     est. tokens   median  653   p90 1199   p99 1999   max  3235
#
# Every text request the model was trained and Gate-F'd on is a *page* of text
# at that scale. extract_text() on a 10k-row spreadsheet returns one string
# orders of magnitude larger; sending it whole is far out of distribution even
# though it would nominally fit the context window. Pagination keeps each
# request inside the measured envelope — "one request = one page" (§2).
TEXT_PAGE_MAX_CHARS = 11000

# §5.1 degeneration gate (permanent, leg-4 verdict): clean pages compress to
# ≈2.4 median / 5.3 p99; loop pages hit 9+.
ZLIB_GATE_RATIO = 7.0


def context_limit_chars(max_model_len: int = MAX_MODEL_LEN,
                         max_tokens: int = MAX_TOKENS) -> int:
    """Hard capacity backstop, in characters.

    vLLM's OpenAI server rejects a request when input_tokens + max_tokens
    exceeds max_model_len. With the contract's fixed max_tokens this leaves
    ~4.1k input tokens — well above TEXT_PAGE_MAX_CHARS, so in practice the
    distribution bound binds first and this only catches a misconfigured
    max_model_len.
    """
    return int((max_model_len - max_tokens) * CHARS_PER_TOKEN)


def output_room(bucket: str, max_model_len: int = MAX_MODEL_LEN) -> int:
    """Tokens a page in `bucket` can actually emit, given the served context.

    This is the number that bit twice (§3.1): `max_tokens` is an upper bound the
    server will silently undercut when the prompt leaves less room, so a page can
    stop at 7,872 tokens with `max_tokens=12000` configured and look like model
    truncation.
    """
    return max_model_len - BUCKET_IMAGE_TOKENS.get(bucket, 0) - PROMPT_TEXT_TOKENS


def check_output_budget(max_model_len: int = MAX_MODEL_LEN,
                        max_tokens: int = MAX_TOKENS) -> dict:
    """Verify the configured output budget is REACHABLE on every tile bucket.

    Acceptance criterion for shrew-server-public#1/#2: an effective ceiling below
    the configured `max_tokens` is a silent truncation, so it is surfaced on
    /health and logged loudly at startup rather than discovered from bad output.
    """
    rooms = {b: output_room(b, max_model_len) for b in BUCKET_IMAGE_TOKENS}
    short = {b: r for b, r in rooms.items() if r < max_tokens}
    return {
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "output_room": rooms,
        "ok": not short,
        "constrained_buckets": short,
    }


def warn_if_budget_unreachable(max_model_len: int = MAX_MODEL_LEN,
                               max_tokens: int = MAX_TOKENS) -> dict:
    report = check_output_budget(max_model_len, max_tokens)
    if not report["ok"]:
        logger.error(
            "OUTPUT BUDGET UNREACHABLE: max_model_len=%d leaves %s — below the "
            "configured max_tokens=%d. Pages in those buckets will be truncated "
            "silently and the truncation is indistinguishable from a loop. "
            "Serve with --max-model-len %d (§3.1).",
            max_model_len,
            ", ".join(f"{b}:{r}" for b, r in sorted(report["constrained_buckets"].items())),
            max_tokens,
            max(BUCKET_IMAGE_TOKENS.values()) + PROMPT_TEXT_TOKENS + max_tokens,
        )
    return report


# --------------------------------------------------------------------------- vendored: parse_json_lenient (run_structured.py)

# Escape-repair regex. Alternation order matters: \b/\f followed by a letter is
# checked FIRST — in this corpus that is always un-escaped LaTeX (\beta, \frac),
# never an intentional backspace/formfeed (audit 2026-06-10: 43 corrupted \beta
# vs 0 legitimate uses). Group 1 = escapes kept verbatim: the \\ pair is consumed
# atomically so doubling can't corrupt it, and \uXXXX requires the 4 hex digits
# (bare \u is LaTeX like \underline). Anything else starting with \ gets its
# backslash doubled. \t, \r and \n before letters are left alone — tab-indented
# lists (sam_gov) and pretty-printed HTML tables (pubtabnet) use them legitimately
# far more often than un-escaped \theta/\rho appears.
_ESC_RX = re.compile(
    r'\\[bf](?=[A-Za-z])'
    r'|(\\\\|\\u[0-9a-fA-F]{4}|\\["/nrtbf])'
    r'|\\'
)


def _esc_fix(m: "re.Match") -> str:
    if m.group(1):
        return m.group(1)
    return "\\" + m.group(0)


_DECODER = json.JSONDecoder()


def parse_json_lenient(text: str):
    """Parse the model's JSON, tolerating ```json fences or leading prose."""
    if not text:
        return None, "empty output"
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    # carve out the outermost {...} if there's surrounding chatter
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        t = t[i:j + 1]
    try:
        parsed = json.loads(t)
    except Exception as e:
        # Retry after escaping stray backslashes (LaTeX like \eta written as a
        # single backslash is an invalid JSON escape — guided decoding still lets
        # it through).
        try:
            return json.loads(_ESC_RX.sub(_esc_fix, t)), None
        except Exception:
            pass
        # Last resort: decode the FIRST complete JSON object starting at the
        # first '{', ignoring any trailing junk (a duplicate object, a ```fence,
        # or commentary after the close — the "Extra data" failure mode).
        k = t.find("{")
        if k != -1:
            frag = t[k:]
            for cand in (frag, _ESC_RX.sub(_esc_fix, frag)):
                try:
                    return _DECODER.raw_decode(cand)[0], None
                except Exception:
                    pass
        return None, f"json parse: {e!r}"
    # Un-escaped \beta/\frac are VALID JSON escapes and silently decode to
    # control chars. On strictly-valid input the repair only rewrites those
    # \b/\f-before-letter cases, so a changed string means corruption was present.
    repaired = _ESC_RX.sub(_esc_fix, t)
    if repaired != t:
        try:
            return json.loads(repaired), None
        except Exception:
            pass
    return parsed, None


# --------------------------------------------------------------------------- vendored: schema (metrics_v2.py)

SECTION_ENUM = {
    "abstract", "introduction", "methodology", "results",
    "discussion", "conclusion", "technical_content", "appendix",
}
FIVE_KEYS = ("metadata", "summary", "semantic_chunks", "figures", "tables")
META_FIELDS = ("title", "authors", "organization", "year", "doc_type")


def validate_schema(sj) -> tuple[bool, list[str]]:
    """5-key conformance. Returns (ok, errors). RL-ready (always-on validity reward)."""
    if not isinstance(sj, dict):
        return False, ["not-a-dict"]
    errs: list[str] = []
    for k in FIVE_KEYS:
        if k not in sj:
            errs.append(f"missing:{k}")
    if isinstance(sj.get("metadata"), dict):
        for f in META_FIELDS:
            if f not in sj["metadata"]:
                errs.append(f"meta-missing:{f}")
    elif "metadata" in sj:
        errs.append("metadata:not-dict")
    if "summary" in sj and not (sj["summary"] is None or isinstance(sj["summary"], str)):
        errs.append("summary:bad-type")
    for lk in ("semantic_chunks", "figures", "tables"):
        if lk in sj and not isinstance(sj[lk], list):
            errs.append(f"{lk}:not-list")
    for c in (sj.get("semantic_chunks") or []):
        if not isinstance(c, dict):
            errs.append("chunk:not-dict"); continue
        if c.get("section_type") not in SECTION_ENUM:
            errs.append(f"chunk:bad-section_type:{c.get('section_type')}")
        for f in ("chunk_id", "title", "content"):
            if f not in c:
                errs.append(f"chunk-missing:{f}")
    for lk in ("figures", "tables"):
        for o in (sj.get(lk) or []):
            if not isinstance(o, dict):
                errs.append(f"{lk}-item:not-dict"); continue
            bb = o.get("bbox")
            if bb is not None and (not isinstance(bb, list) or len(bb) != 4
                                   or not all(isinstance(v, (int, float)) for v in bb)):
                errs.append(f"{lk}:bad-bbox")
            if lk == "tables" and not isinstance(o.get("html", ""), str):
                errs.append("table:bad-html-type")
    return (len(errs) == 0), errs


# --------------------------------------------------------------------------- enforcement schema (retry tier only)

# Mirrors validate_schema. Used ONLY on the flagged retry (§3): first-pass
# enforcement is off, because the 0.990/0.993 valid-JSON rates ARE the model's
# unenforced rates and constraining the first pass would void them.
#
# Fork gotcha ([[vllm_fork_structured_outputs]]): OpenAI-style `guided_json` is
# SILENTLY IGNORED by our vLLM fork. Only `structured_outputs: {"json": ...}`
# actually constrains. VLMClient posts raw REST, so this goes in as a top-level
# payload key — the raw-request equivalent of the SDK's extra_body.
ENFORCEMENT_SCHEMA = {
    "type": "object",
    "required": list(FIVE_KEYS),
    "properties": {
        "metadata": {
            "type": "object",
            "required": list(META_FIELDS),
            "properties": {
                "title": {"type": ["string", "null"]},
                "authors": {"type": "array", "items": {"type": "string"}},
                "organization": {"type": ["string", "null"]},
                "year": {"type": ["string", "integer", "null"]},
                "doc_type": {"type": ["string", "null"]},
            },
        },
        "summary": {"type": ["string", "null"]},
        "semantic_chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["chunk_id", "title", "content", "section_type"],
                "properties": {
                    "chunk_id": {"type": ["string", "integer"]},
                    "title": {"type": ["string", "null"]},
                    "content": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "section_type": {"enum": sorted(SECTION_ENUM)},
                },
            },
        },
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {"type": ["array", "null"],
                              "items": {"type": "number"},
                              "minItems": 4, "maxItems": 4},
                    "caption": {"type": ["string", "null"]},
                },
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {"type": ["array", "null"],
                              "items": {"type": "number"},
                              "minItems": 4, "maxItems": 4},
                    "caption": {"type": ["string", "null"]},
                    "html": {"type": "string"},
                },
            },
        },
    },
}


# Retry-tier presence penalty (0 disables). The retry must differ from the
# deterministic first pass; a stronger penalty is the measured lever that
# breaks repetition ruts (12/12 loop-failed pages rescued at 0.6, VHR
# 2026-08-16) and it composes with enforcement below.
RETRY_PP = float(os.environ.get("SHREW_RETRY_PP", "0.6"))


def enforcement_params() -> dict:
    """extra_params for the one flagged retry (§3 retry tier).

    Emits BOTH schema-enforcement dialects so the retry actually constrains
    on whichever backend is serving:
    - ``structured_outputs: {"json": ...}`` — our vLLM fork's field (OpenAI
      ``guided_json`` is silently ignored by the fork; see the ENFORCEMENT_SCHEMA
      note).
    - ``json_schema`` — llama.cpp's native field. Without this a GGUF deployment
      retried with the penalty ALONE and no grammar (measured 2026-08-25).

    Each backend honors its own key and tolerates the other's (both verified
    live: vLLM returns conforming output and ignores the stray json_schema;
    llama.cpp enforces json_schema and ignores structured_outputs). Env
    ``SHREW_ENFORCE_DIALECT`` can pin to "vllm" or "llamacpp" if a future
    backend rejects the unknown field.
    """
    dialect = os.environ.get("SHREW_ENFORCE_DIALECT", "both").lower()
    out: dict = {}
    if dialect in ("both", "vllm"):
        out["structured_outputs"] = {"json": ENFORCEMENT_SCHEMA}
    if dialect in ("both", "llamacpp"):
        out["json_schema"] = ENFORCEMENT_SCHEMA
    if RETRY_PP:
        out["presence_penalty"] = RETRY_PP
    return out


# --------------------------------------------------------------------------- §5.1 degeneration gate


def zlib_ratio(raw: str) -> float:
    """Compression ratio of the raw completion (§5.1).

    Clean pages sit around 2.4 median / 5.3 p99; repetition-collapse loops go
    to 9+. Empty output returns 0.0 so it can never trip the gate — an empty
    completion is the §5.3 empty-200 signal, a different failure.
    """
    if not raw:
        return 0.0
    payload = raw.encode("utf-8", errors="replace")
    compressed = zlib.compress(payload)
    if not compressed:
        return 0.0
    return len(payload) / len(compressed)


def is_degenerate(raw: str, gate: float = ZLIB_GATE_RATIO) -> bool:
    return zlib_ratio(raw) > gate


# --------------------------------------------------------------------------- §5.2 streaming repetition guard

# Detect loops DURING generation, not after. Without this a loop runs to the
# full output cap: 46% of newspaper pages consumed all 20,000 tokens producing
# garbage, and enabling the guard took screening throughput from 37 to 415
# pages/hr on the same hardware.
#
# Whole-stream compression cannot do this job — a page that emits 8k tokens of
# correct content and then degenerates still has a healthy overall ratio,
# because the good prefix dominates. The window must be trailing.
#
# CALIBRATED on 252 dense pages (§5.2), NOT inherited from the §5.1 document
# gate: a short window of structured JSON — repeated keys, field names,
# punctuation — compresses far harder than a whole document, so 7.0 would fire
# on legitimate content. At these settings the guard fires at median 4,404
# chars with ZERO false positives across 150 clean pages (clean window maxima:
# median 2.1, p95 3.9, max 10.1) while loops score 34-86. 15.0 sits in a wide
# empty gap; 10 would not.
STREAM_GUARD_WINDOW_CHARS = int(os.environ.get("SHREW_LOOP_WINDOW", "2000"))
STREAM_GUARD_CHECK_EVERY_CHARS = int(os.environ.get("SHREW_LOOP_CHECK_EVERY", "800"))
STREAM_GUARD_RATIO = float(os.environ.get("SHREW_LOOP_RATIO", "15.0"))
STREAM_GUARD_CONSECUTIVE = int(os.environ.get("SHREW_LOOP_CONSECUTIVE", "2"))
STREAM_GUARD_ENABLED = os.environ.get("SHREW_LOOP_GUARD", "1").strip().lower() not in {
    "0", "false", "no", "off",
}

REPETITION_ABORT = "repetition_abort"

# Wall-clock bound on one streamed generation. The §5.2 guard only catches
# COMPRESSIBLE loops; a manuscript page emitting varied gibberish rides to the
# 20k-token cap unaborted, and at congested per-request decode speeds that ran
# past two hours (measured live, VHR corpus run 2026-08-14: GNHK handwriting
# tail p100 5,408s). requests' `timeout` cannot bound this — it applies to gaps
# between bytes, and a grinding stream never stops trickling. 3600s allows the
# slowest legitimate page (20k tokens at ~10 tok/s is ~2000s) with margin.
WALL_CLOCK_ABORT = "wall_clock_abort"
STREAM_WALL_CLOCK_S = float(os.environ.get("SHREW_PAGE_WALL_CLOCK", "3600"))


class RepetitionGuard:
    """Trailing-window compression check over a token stream.

    Use as the ``on_delta`` callback of ``VLMClient.chat_completion_stream``:
    returns None to continue, or ``REPETITION_ABORT`` to stop the generation.

    Requires ``consecutive`` violating windows before aborting. A single window
    can spike on legitimately repetitive content (a dense numeric table, a long
    run of near-identical rows); requiring the condition to persist costs a few
    hundred tokens of latency and removes that false-positive class. Checks
    start only once a FULL window has accumulated — a partial window of JSON
    scaffolding compresses much harder than a real one.
    """

    def __init__(self, window: int = STREAM_GUARD_WINDOW_CHARS,
                 check_every: int = STREAM_GUARD_CHECK_EVERY_CHARS,
                 threshold: float = STREAM_GUARD_RATIO,
                 consecutive: int = STREAM_GUARD_CONSECUTIVE,
                 page_no: int | None = None,
                 attempt: str | None = None):
        self.window = window
        self.check_every = check_every
        self.threshold = threshold
        self.consecutive = consecutive
        # Observability only: pages stream concurrently, so an unprefixed
        # warning burst can't be correlated with page results. page_no is the
        # pipeline's one-based document page index (not a number printed on
        # the page); attempt distinguishes first pass from the enforced retry.
        self.page_no = page_no
        self.attempt = attempt
        self.violations = 0
        self.ratio = 0.0
        self.max_ratio = 0.0
        self.position = 0
        self.fired = False
        self._next_check = window

    def __call__(self, chunk: str, accumulated: str):
        n = len(accumulated)
        if n < self._next_check:
            return None
        self._next_check = n + self.check_every
        self.ratio = zlib_ratio(accumulated[-self.window:])
        self.max_ratio = max(self.max_ratio, self.ratio)
        if self.ratio <= self.threshold:
            self.violations = 0
            return None
        self.violations += 1
        if self.violations < self.consecutive:
            return None
        self.fired = True
        self.position = n
        context = ""
        if self.page_no is not None:
            context = f"Page {self.page_no}"
            if self.attempt:
                context += f" ({self.attempt})"
            context += ": "
        logger.warning(
            "%srepetition_abort at %d chars: trailing-window zlib ratio %.1f > %.1f "
            "for %d consecutive windows",
            context, n, self.ratio, self.threshold, self.violations,
        )
        return REPETITION_ABORT

    def stats(self) -> dict:
        return {
            "aborted": self.fired,
            "ratio": round(self.ratio, 2),
            "max_window_ratio": round(self.max_ratio, 2),
            "position_chars": self.position,
            "threshold": self.threshold,
            "window_chars": self.window,
        }


# --------------------------------------------------------------------------- message construction


def build_image_messages(image_path: str | Path) -> list[dict]:
    """§2 image modality: sentinel system prompt, user content = image ONLY."""
    return [
        {"role": "system", "content": SENTINEL},
        {"role": "user", "content": [make_image_content(image_path)]},
    ]


def build_text_messages(text: str) -> list[dict]:
    """§2 text modality: sentinel system prompt, user content = the raw string.

    Note the content is a plain string, not a content-part list — that is the
    shape the text arm was trained and Gate-F'd on.
    """
    return [
        {"role": "system", "content": SENTINEL},
        {"role": "user", "content": text},
    ]


# --------------------------------------------------------------------------- core call + gates


# §5.3 empty-200 watch. Empty completions returned with HTTP 200 are the
# signature of TP-rank desync from XGMI faults (2026-07-22): serving breaks
# silently and every page comes back blank. Baseline XGMI noise (~150/day) is
# normal and self-clears, so we alert on CONSECUTIVE empties, not a rate.
# State is process-wide because a storm spans requests.
EMPTY_200_ALERT_THRESHOLD = 5
_empty_streak = 0
_empty_lock = threading.Lock()


def _note_completion(was_empty: bool) -> int:
    """Track the consecutive-empty streak. Returns the streak length."""
    global _empty_streak
    with _empty_lock:
        _empty_streak = _empty_streak + 1 if was_empty else 0
        streak = _empty_streak
    if was_empty and streak >= EMPTY_200_ALERT_THRESHOLD:
        logger.error(
            "EMPTY-200 ALERT: %d consecutive empty completions with HTTP 200. "
            "This is the TP-rank-desync signature (XGMI fault) — the serving "
            "process needs a restart; output is silently blank until then.",
            streak,
        )
    return streak


def empty_200_streak() -> int:
    """Current consecutive-empty-completion streak (for /health)."""
    with _empty_lock:
        return _empty_streak


def reset_empty_200_streak() -> None:
    global _empty_streak
    with _empty_lock:
        _empty_streak = 0


def _result(ok, data, status, error, attempts, raw_len, *,
            schema_coerced=False, degenerate=False, repetition_abort=False,
            loop_guard=None) -> dict:
    return {
        "ok": ok,
        "data": data,
        "status": status,
        "error": error,
        "attempts": attempts,
        "raw_len": raw_len,
        "schema_coerced": schema_coerced,
        "degenerate": degenerate,
        "repetition_abort": repetition_abort,
        "loop_guard": loop_guard,
    }


def _gate(text: str, finish_reason: str | None, guard: "RepetitionGuard | None" = None):
    """Run the §5 gates over one completion.

    Returns (parsed, verdict, error) where verdict is one of
    "ok" | "parse" | "schema" | "degenerate" | "length" | "empty".
    """
    if finish_reason == REPETITION_ABORT:
        # §5.2: an aborted page is looped BY THE ABORT, never re-derived from
        # the truncated text. Stopping early leaves less output, and a loop
        # killed at ~1.5k chars can score a whole-string ratio around 5 — under
        # the §5.1 gate — so re-deriving would silently count it clean.
        obs = guard.ratio if guard else 0.0
        pos = guard.position if guard else len(text)
        return None, "degenerate", (f"repetition_abort: trailing-window zlib ratio "
                                    f"{obs:.1f} at {pos} chars")
    if finish_reason == "length":
        # Truncated mid-object. Never parse a partial per §3 — treat as a
        # parse-class failure so it takes the one enforcement retry.
        return None, "length", "hit max_tokens"
    if not text.strip():
        # §5.3: empty completion with HTTP 200. Surfaced distinctly so the
        # server-side watchdog can count consecutive occurrences.
        return None, "empty", "empty completion (HTTP 200)"
    # Degeneration is checked on the RAW string before parsing — a loop often
    # still parses (a chunk list repeating the same block) and would otherwise
    # sail through the schema gate.
    if is_degenerate(text):
        return None, "degenerate", f"zlib ratio {zlib_ratio(text):.2f} > {ZLIB_GATE_RATIO}"
    parsed, perr = parse_json_lenient(text)
    if parsed is None:
        return None, "parse", perr
    schema_ok, serrs = validate_schema(parsed)
    if not schema_ok:
        return parsed, "schema", "schema:" + ";".join(serrs)
    return parsed, "ok", None


def _call(messages, client, *, max_tokens, temperature, extra_params, timeout,
          page_no=None, attempt=None):
    """One model call, streamed when the §5.2 guard is on.

    Streaming changes nothing about sampling — greedy is greedy — so the eval
    numbers are untouched; it only makes the output observable while it is being
    produced, which is what lets a loop be aborted instead of paid for in full.
    Returns (text, finish_reason, guard). A server that cannot stream falls back
    to the blocking path with the guard disabled rather than failing the page.
    """
    if not (STREAM_GUARD_ENABLED and hasattr(client, "chat_completion_stream")):
        result = client.chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature,
            extra_params=extra_params, timeout=timeout,
        )
        choice = result["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason"), None

    guard = RepetitionGuard(page_no=page_no, attempt=attempt)
    result = client.chat_completion_stream(
        messages, max_tokens=max_tokens, temperature=temperature,
        extra_params=extra_params, timeout=timeout, on_delta=guard,
        wall_clock_s=STREAM_WALL_CLOCK_S,
    )
    choice = result["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason"), guard


def _content_chars(sj: dict) -> int:
    """Characters of actual extracted content in a 5-key result — chunk text,
    table markup, figure captions, summary. JSON scaffolding excluded."""
    n = len(sj.get("summary") or "")
    for c in sj.get("semantic_chunks") or []:
        if isinstance(c, dict):
            n += len(c.get("content") or "") + len(c.get("title") or "")
    for t in sj.get("tables") or []:
        if isinstance(t, dict):
            n += len(t.get("html") or "")
    for f in sj.get("figures") or []:
        if isinstance(f, dict):
            n += len(f.get("caption") or "")
    return n


# Coercion-density floor for image pages (chars of extracted content). The one
# measured slop case carried 206; a dense page's real text runs 5-15k. 300 is
# far under any legitimate coerced dense page while comfortably above the slop.
# The floor alone over-fires on honest sparse pages (handwritten notes,
# diagram-only pages: 204/206 gate kills on VHR 2026-08-15 had <300 chars in
# the reference arm too), so it only counts when the coerced output is ALSO
# tiny relative to the first-pass emission — a page the model ground 33k chars
# against does not honestly rescue to 206; a page whose first pass said little
# legitimately coerces to little.
COERCED_MIN_CHARS = int(os.environ.get("SHREW_COERCED_MIN_CHARS", "300"))
COERCED_MIN_RATIO = float(os.environ.get("SHREW_COERCED_MIN_RATIO", "0.02"))


def _extract(messages: list[dict], client, *, max_tokens: int, timeout=None,
             min_content_chars: int = 0, page_no: int | None = None) -> dict:
    """Run the §3 first pass + at most one flagged enforcement retry."""
    params = get_generation_params(client.model, "structured_page")

    # ── First pass: minimal params, no enforcement ──────────────────────────
    text, finish_reason, guard = _call(
        messages, client,
        max_tokens=max_tokens,
        temperature=params["temperature"],
        extra_params=params["extra_params"],
        timeout=timeout,
        page_no=page_no, attempt="attempt 1/2, first pass",
    )
    parsed, verdict, error = _gate(text, finish_reason, guard)
    _note_completion(verdict == "empty")
    aborted = finish_reason == REPETITION_ABORT

    if verdict == "ok":
        return _result(True, parsed, "ok", None, 1, len(text),
                       loop_guard=guard.stats() if guard else None)

    if finish_reason == WALL_CLOCK_ABORT:
        # No retry: greedy is deterministic, so an enforcement retry grinds for
        # the full cap again before failing the same way. Fail the page now and
        # keep the doc moving — the retry tier is for failures a second call
        # can actually change.
        return _result(False, None, "wall_clock_abort",
                       f"stream exceeded {STREAM_WALL_CLOCK_S:.0f}s wall clock "
                       f"({len(text)} chars emitted)",
                       1, len(text), loop_guard=guard.stats() if guard else None)

    # ── Retry tier: ONE attempt, enforcement on, flagged ────────────────────
    # Greedy is deterministic: an identical request reproduces the identical
    # failure, so the retry MUST differ, and enforcement is the only permitted
    # difference. Never retry at temperature > 0 (§3.3) — that output is a
    # lottery ticket nobody evaluated.
    first_verdict = verdict
    first_error = error

    rtext, rfinish, rguard = _call(
        messages, client,
        max_tokens=max_tokens,
        temperature=params["temperature"],
        extra_params={**(params["extra_params"] or {}), **enforcement_params()},
        timeout=timeout,
        page_no=page_no, attempt="attempt 2/2, enforced retry",
    )
    rparsed, rverdict, rerror = _gate(rtext, rfinish, rguard)
    _note_completion(rverdict == "empty")
    aborted = aborted or rfinish == REPETITION_ABORT
    stats = (rguard or guard).stats() if (rguard or guard) else None

    if rverdict == "ok":
        # Coercion-density gate. Enforcement can produce WELL-FORMED slop: a
        # dense broadsheet whose first pass looped came back from the retry as
        # valid JSON carrying 206 chars of hallucinated names against 13,948
        # chars of real page text — it passed every gate and silently poisoned
        # its 43 retrieval queries (VHR run 2026-08-14). A coerced rescue that
        # emits almost nothing from a page the FIRST PASS ground thousands of
        # chars against is not a rescue. Sparse pages are exempt two ways:
        # first-pass successes never reach this gate, and a coerced result is
        # only slop if it is also tiny relative to the first-pass emission —
        # honest sparse pages (handwriting photos, diagram-only pages) fail
        # first-pass on FORM, not volume, and rescue to proportionate content.
        _cc = _content_chars(rparsed)
        if (min_content_chars and _cc < min_content_chars
                and _cc < COERCED_MIN_RATIO * len(text)):
            return _result(False, None, "coerced_empty",
                           f"coerced output carries {_cc} content chars "
                           f"(< {min_content_chars} floor and < {COERCED_MIN_RATIO:.0%} "
                           f"of the {len(text)}-char first-pass emission) "
                           f"— hallucination-slop signature, counted failed",
                           2, len(rtext), schema_coerced=True,
                           degenerate=(first_verdict == "degenerate"),
                           repetition_abort=aborted, loop_guard=stats)
        # Well-formed but outside the validated distribution — flag it so
        # downstream can filter or de-rank.
        return _result(True, rparsed, "ok_coerced", None, 2, len(rtext),
                       schema_coerced=True, degenerate=(first_verdict == "degenerate"),
                       repetition_abort=aborted, loop_guard=stats)

    # Still failing → page-level failure record; the doc-level pipeline continues.
    status = {
        "length": "overlong_failed",
        "degenerate": "degenerate",
        "empty": "empty_completion",
    }.get(first_verdict, "failed")
    return _result(False, None, status,
                   f"first={first_verdict}:{first_error}; retry={rverdict}:{rerror}",
                   2, len(rtext), degenerate=(first_verdict == "degenerate"
                                              or rverdict == "degenerate"),
                   repetition_abort=aborted, loop_guard=stats)


# --------------------------------------------------------------------------- public API


def extract_page(image_path: str | Path, client, *, max_tokens: int = MAX_TOKENS,
                  timeout=None, page_no: int | None = None) -> dict:
    """Run structured extraction on one page image (§2 image modality).

    ``image_path`` must already be ``preprocess.prepare_image_bucketed`` output
    — fitted to its glyph-routed tile bucket, then enhanced. Feeding anything
    else (a raw render, the legacy fixed DPI downscale) changes the model's
    output distribution and voids the eval numbers.

    Returns {"ok", "data", "status", "error", "attempts", "raw_len",
             "schema_coerced", "degenerate", "repetition_abort", "loop_guard"}.
    """
    # The coercion-density floor applies to image pages only: a rendered page
    # that needed the enforcement retry AND produced almost no content is the
    # hallucination-slop signature. Text pages are exempt — a tiny text segment
    # legitimately coerces to a tiny result.
    return _extract(build_image_messages(image_path), client,
                    max_tokens=max_tokens, timeout=timeout,
                    min_content_chars=COERCED_MIN_CHARS, page_no=page_no)


def extract_text_page(text: str, client, *, max_tokens: int = MAX_TOKENS,
                       max_model_len: int = MAX_MODEL_LEN,
                       max_chars: int = TEXT_PAGE_MAX_CHARS, timeout=None,
                       page_no: int | None = None) -> dict:
    """Run structured extraction on one text page (§2 text modality).

    §5.4 — never truncate: a page over the size bound is filtered and reported,
    not shortened. Callers should run text through ``paginate_text`` first, so
    this only fires on a single unsplittable line (one enormous CSV record, a
    minified HTML blob).
    """
    limit = min(max_chars, context_limit_chars(max_model_len, max_tokens))
    if len(text) > limit:
        est = int(len(text) / CHARS_PER_TOKEN)
        return _result(False, None, "oversize",
                       f"{len(text)} chars (~{est} tok) exceeds the {limit}-char "
                       f"text-page bound; filtered rather than truncated",
                       0, len(text))
    return _extract(build_text_messages(text), client,
                    max_tokens=max_tokens, timeout=timeout, page_no=page_no)


# --------------------------------------------------------------------------- text pagination


# Split preference order: blank line (paragraph), then single newline. We never
# split inside a line — a line is the smallest unit the extractors emit, and a
# markdown table row or a CSV record cut in half is worse than a filtered page.
_PARA_SPLIT = re.compile(r"\n\s*\n")


def paginate_text(text: str, max_chars: int = TEXT_PAGE_MAX_CHARS) -> list[str]:
    """Split extracted text into page-sized blocks for the text modality.

    "One request = one page" (§2): the text arm only ever saw page-sized input
    (see TEXT_PAGE_MAX_CHARS for the measured envelope), so a spreadsheet dump
    or a long markdown file is paginated rather than sent whole. Paragraphs are
    packed greedily; a paragraph over the bound is broken on line boundaries; a
    single line over the bound is emitted as its own page, which
    ``extract_text_page`` then filters and reports (§5.4 — never truncate).
    """
    if not text.strip():
        return []

    def _pack(units: list[str], joiner: str) -> list[str]:
        pages: list[str] = []
        cur = ""
        for u in units:
            candidate = u if not cur else cur + joiner + u
            if len(candidate) <= max_chars:
                cur = candidate
                continue
            if cur:
                pages.append(cur)
            if len(u) <= max_chars:
                cur = u
            else:
                # Too big even alone — break it down a level.
                if joiner == "\n\n":
                    pages.extend(_pack(u.split("\n"), "\n"))
                else:
                    pages.append(u)  # single oversize line: filtered downstream
                cur = ""
        if cur:
            pages.append(cur)
        return pages

    return [p for p in _pack(_PARA_SPLIT.split(text.strip()), "\n\n") if p.strip()]
