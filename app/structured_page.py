"""Per-page structured extraction (shrew-ocr-preview, 5-key sentinel target).

Implements the serving contract in shrew_ocr/deploy/SHREW_OCR_PREVIEW.md.
Two input modalities, one output schema:

  * image — a page rendered at 200 DPI and put through ``preprocess.prepare_image``
    (luminance, downscaled to 100 DPI), sent as a data-URI image part.
  * text  — messy HTML/markdown/code/plain text sent as a raw string. Same
    5-key output; ``figures[].bbox`` / ``tables[].bbox`` come back null.

Sampling is EXACTLY ``temperature 0, max_tokens 12000`` (§3). Gate F measured
penalty-free unconstrained greedy, so any decode knob here voids the published
eval numbers. Because greedy is deterministic, an identical retry reproduces
an identical failure — the only retry allowed is ONE attempt with
``structured_outputs`` enforcement, and any page that survives it is flagged
``schema_coerced`` so downstream can filter or de-rank it.

# parse_json_lenient vendored from shrew_ocr/structured_eval/run_structured.py; validate_schema/SECTION_ENUM from shrew_ocr/structured_eval/metrics_v2.py — keep in sync.
"""

import json
import logging
import re
import threading
import zlib
from pathlib import Path

from .generation import get_generation_params
from .vlm_client import make_image_content

logger = logging.getLogger("shrew.structured_page")

SENTINEL = "structured_extraction"

# §3: the eval setting. Longest GT ≈ 6.1k tokens; 12000 leaves headroom.
MAX_TOKENS = 12000
# §4 serving recipe: vllm serve ... --max-model-len 16384
MAX_MODEL_LEN = 16384
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


def enforcement_params() -> dict:
    """extra_params for the one flagged retry (§3 retry tier)."""
    return {"structured_outputs": {"json": ENFORCEMENT_SCHEMA}}


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
            schema_coerced=False, degenerate=False) -> dict:
    return {
        "ok": ok,
        "data": data,
        "status": status,
        "error": error,
        "attempts": attempts,
        "raw_len": raw_len,
        "schema_coerced": schema_coerced,
        "degenerate": degenerate,
    }


def _gate(text: str, finish_reason: str | None):
    """Run the §5 gates over one completion.

    Returns (parsed, verdict, error) where verdict is one of
    "ok" | "parse" | "schema" | "degenerate" | "length" | "empty".
    """
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


def _extract(messages: list[dict], client, *, max_tokens: int, timeout=None) -> dict:
    """Run the §3 first pass + at most one flagged enforcement retry."""
    params = get_generation_params(client.model, "structured_page")

    # ── First pass: minimal params, no enforcement ──────────────────────────
    result = client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=params["temperature"],
        extra_params=params["extra_params"],
        timeout=timeout,
    )
    choice = result["choices"][0]
    text = choice["message"].get("content") or ""
    parsed, verdict, error = _gate(text, choice.get("finish_reason"))
    _note_completion(verdict == "empty")

    if verdict == "ok":
        return _result(True, parsed, "ok", None, 1, len(text))

    # ── Retry tier: ONE attempt, enforcement on, flagged ────────────────────
    # Greedy is deterministic: an identical request reproduces the identical
    # failure, so the retry MUST differ, and enforcement is the only permitted
    # difference. Never retry at temperature > 0 (§3.3) — that output is a
    # lottery ticket nobody evaluated.
    first_verdict = verdict
    first_error = error

    retry = client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=params["temperature"],
        extra_params={**(params["extra_params"] or {}), **enforcement_params()},
        timeout=timeout,
    )
    rchoice = retry["choices"][0]
    rtext = rchoice["message"].get("content") or ""
    rparsed, rverdict, rerror = _gate(rtext, rchoice.get("finish_reason"))
    _note_completion(rverdict == "empty")

    if rverdict == "ok":
        # Well-formed but outside the validated distribution — flag it so
        # downstream can filter or de-rank.
        return _result(True, rparsed, "ok_coerced", None, 2, len(rtext),
                       schema_coerced=True, degenerate=(first_verdict == "degenerate"))

    # Still failing → page-level failure record; the doc-level pipeline continues.
    status = {
        "length": "overlong_failed",
        "degenerate": "degenerate",
        "empty": "empty_completion",
    }.get(first_verdict, "failed")
    return _result(False, None, status,
                   f"first={first_verdict}:{first_error}; retry={rverdict}:{rerror}",
                   2, len(rtext), degenerate=(first_verdict == "degenerate"
                                              or rverdict == "degenerate"))


# --------------------------------------------------------------------------- public API


def extract_page(image_path: str | Path, client, *, max_tokens: int = MAX_TOKENS,
                  timeout=None) -> dict:
    """Run structured extraction on one page image (§2 image modality).

    ``image_path`` must already be ``preprocess.prepare_image`` output — a
    200-DPI render reduced to 100-DPI luminance. Feeding anything else (a raw
    render, a different DPI) changes the model's output distribution and voids
    the eval numbers.

    Returns {"ok", "data", "status", "error", "attempts", "raw_len",
             "schema_coerced", "degenerate"}.
    """
    return _extract(build_image_messages(image_path), client,
                    max_tokens=max_tokens, timeout=timeout)


def extract_text_page(text: str, client, *, max_tokens: int = MAX_TOKENS,
                       max_model_len: int = MAX_MODEL_LEN,
                       max_chars: int = TEXT_PAGE_MAX_CHARS, timeout=None) -> dict:
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
                    max_tokens=max_tokens, timeout=timeout)


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
