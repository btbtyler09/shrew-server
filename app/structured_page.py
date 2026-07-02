"""Per-page structured extraction (v2, 5-key sentinel target).

Calls the v2 model on a single page image with the ``structured_extraction``
sentinel system prompt, parses/validates the 5-key JSON response, and applies
a never-truncate retry policy: one retry with a higher token cap on
``finish_reason == "length"``, and one resample on parse/schema/degeneration
failure. See assembly-spec §8 for the retry-ladder rationale.

# parse_json_lenient vendored from shrew_ocr/structured_eval/run_structured.py; validate_schema/degeneration_score/SECTION_ENUM from shrew_ocr/structured_eval/metrics_v2.py — keep in sync.
"""

import json
import re
from collections import Counter
from pathlib import Path

from .generation import get_generation_params
from .vlm_client import make_image_content

SENTINEL = "structured_extraction"

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


# --------------------------------------------------------------------------- vendored: schema/degeneration (metrics_v2.py)

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


def degeneration_score(raw: str) -> tuple[bool, int, float]:
    """Repetition-collapse detector (mirrors eval_student_vllm.degeneration_score, kept here
    so this module is standalone for RL). Returns (degenerate, max_repeat, uniq_line_ratio).
    Catches both whole-line loops and sub-line 30-char shingle loops over the output tail."""
    lines = [l.strip() for l in raw.splitlines() if len(l.strip()) > 3]
    if not lines:
        return (False, 0, 1.0)
    c = Counter(lines)
    max_rep = c.most_common(1)[0][1]
    uniq_ratio = len(c) / len(lines)
    tail = raw[-8000:]
    shingles = Counter(tail[i:i + 30] for i in range(0, max(0, len(tail) - 30), 7))
    max_shingle = shingles.most_common(1)[0][1] if shingles else 0
    degenerate = max_rep >= 15 or uniq_ratio < 0.4 or max_shingle >= 25
    return (degenerate, max(max_rep, max_shingle), round(uniq_ratio, 3))


# --------------------------------------------------------------------------- extract_page


def extract_page(image_path: str | Path, client, *, max_tokens: int = 12000,
                  max_tokens_cap: int = 24000) -> dict:
    """Run v2 structured extraction on a single page image.

    Calls ``client.chat_completion`` with the ``structured_extraction`` sentinel
    system prompt and a single image user message. Applies a never-truncate
    retry policy: on ``finish_reason == "length"`` retry once with a higher
    ``max_tokens`` cap (never parses/truncates the partial output); on parse,
    schema, or degeneration failure, resample once with the same params.

    Returns a dict: {"ok", "data", "status", "error", "attempts", "raw_len"}.
    """
    messages = [
        {"role": "system", "content": SENTINEL},
        {"role": "user", "content": [make_image_content(image_path)]},
    ]
    params = get_generation_params(client.model, "structured_page")

    attempts = 0
    cur_max = max_tokens
    length_retried = False
    parse_resampled = False
    last_error = None
    raw_len = 0

    while True:
        result = client.chat_completion(
            messages,
            max_tokens=cur_max,
            temperature=params["temperature"],
            extra_params=params["extra_params"],
        )
        attempts += 1
        choice = result["choices"][0]
        finish = choice.get("finish_reason")
        text = choice["message"].get("content") or ""
        raw_len = len(text)

        if finish == "length":
            if not length_retried:
                length_retried = True
                cur_max = min(max_tokens_cap, int(cur_max * 1.5))
                continue
            return {"ok": False, "data": None, "status": "overlong_failed",
                    "error": "hit max_tokens after retry", "attempts": attempts,
                    "raw_len": raw_len}

        parsed, perr = parse_json_lenient(text)
        schema_ok, serrs = (validate_schema(parsed) if parsed is not None
                             else (False, ["unparseable"]))
        degen = degeneration_score(text)[0]

        if parsed is not None and schema_ok and not degen:
            return {"ok": True, "data": parsed, "status": "ok", "error": None,
                    "attempts": attempts, "raw_len": raw_len}

        last_error = perr or ("degenerate" if degen else ("schema:" + ";".join(serrs)))
        if not parse_resampled:
            parse_resampled = True
            continue
        return {"ok": False, "data": None, "status": "failed", "error": last_error,
                "attempts": attempts, "raw_len": raw_len}
