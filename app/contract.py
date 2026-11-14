"""Model contracts: what the server sends the OCR model and what it expects back.

Two contracts are served side by side so one server can A/B either model.
The names are the model's release line (semver major.minor):

``v0.4``            shrew-ocr-preview v0.4.x. Byte-identical to v0.4.1: the
                   ``structured_extraction`` sentinel, a bare-string text turn,
                   2-point ``[x0, y0, x1, y1]`` table/figure boxes, the sampled
                   v0.4 decode recipe.

``v0.5`` (default)  shrew-ocr-preview v0.5.x. Prompts are commands:
                   the system turn is the bare command and nothing
                   else; the user turn is a list of content parts (image only
                   for pages and figure crops, one text part for text input).
                   Table/figure ``bbox`` is FOUR CORNERS ``[[x, y] x 4]``
                   (top-left, top-right, bottom-right, bottom-left of the
                   element's own reading frame, 0-1000 page coordinates); the
                   API returns it as-is, in place of the 2-point box (there
                   is no second box field).
                   Figures may carry ``chart_data``. Decode is v0.4.0's
                   sampled recipe, identical to v0.4; greedy is a diagnostic
                   opt-in only, never a default.

Selected per server by ``SHREW_CONTRACT`` and per request by the ``contract``
form field.
"""
from __future__ import annotations

import math
import os
import re

V04 = "v0.4"
V05 = "v0.5"
CONTRACTS = (V04, V05)

# v0.5 commands (the whole system turn). The server sends only these three.
CMD_EXTRACT = "structured_extraction"
CMD_CLASSIFY_FIGURE = "classify_figure"
CMD_FIX_HEADINGS = "fix_headings"

# v0.5 image tokens: Granite4VisionProcessor._get_number_of_features (the
# checkpoint's processing.py, a LlavaNextProcessor with a 4/8 downsample), as
# pure arithmetic so the server needs no HF processor. SigLIP 384 tiles, patch
# 16 -> 24 patches, x 4/8 -> 12; strategy "full" adds no extra token. The count
# depends on the fitted size only and is NOT the
# bucket's nominal number (a B0 page can cost 1,476 against a nominal 1,188, a
# B3 page 7,152 against 6,300: V05_BUCKET_IMAGE_TOKENS below).
# Verified token-exact against the real processor: tests/test_v05_contract.py.
V05_PINPOINTS = [[1536, 1152], [2304, 1536], [3072, 2304], [1152, 1152]]   # [h, w]
V05_TILE = 384
V05_GRID = 12                       # (384 // 16) * 4/8
V05_TEXT_OVERHEAD = 18              # system + user + generation-prompt tokens (training)


def _select_best_resolution(h: int, w: int, pins) -> tuple[int, int]:
    best, max_eff, min_waste = None, 0, float("inf")
    for ph, pw in pins:
        scale = min(pw / w, ph / h)
        dw, dh = int(w * scale), int(h * scale)
        eff = min(dw * dh, w * h)
        waste = pw * ph - eff
        if eff > max_eff or (eff == max_eff and waste < min_waste):
            best, max_eff, min_waste = (ph, pw), eff, waste
    return best


def v05_image_tokens(width: int, height: int, pins=V05_PINPOINTS) -> int:
    """Exact <image> token count the v0.5 processor emits for an image of this size."""
    bh, bw = _select_best_resolution(height, width, pins)
    cur_h, cur_w = V05_GRID * (bh // V05_TILE), V05_GRID * (bw // V05_TILE)
    if width / height > cur_w / cur_h:
        new_h = int(round(height * (cur_w / width), 7))
        cur_h -= ((cur_h - new_h) // 2) * 2
    else:
        new_w = int(round(width * (cur_h / height), 7))
        cur_w -= ((cur_w - new_w) // 2) * 2
    return cur_h * cur_w + cur_h + V05_GRID * V05_GRID


# v0.5 token budget: the cut was built under a 30,000-token cap
# (train max_seq_length 30000) and served at --max-model-len 32768.
# max_tokens = budget - prompt tokens, where the prompt is counted exactly
# (v05_tokens: the served tokenizer; v05_image_tokens: the processor's count).
V05_TOKEN_BUDGET = int(os.environ.get("SHREW_V05_TOKEN_BUDGET", "30000"))
# The largest image-token count of any size that fits INSIDE each bucket's box,
# reached at an exact fill of the pinpoint (B3 3072x2304 -> 96*72 + 96 + 144).
# Swept over every size in each box and over 200-6000 px on both axes
# (2026-11-14); the four fills are checked against the real processor in
# tests/test_v05_budget.py. These replace the nominal table (1188 / 1846 /
# 3182 / 6300), which sat below what the processor emits.
V05_BUCKET_IMAGE_TOKENS = {"B0": 1476, "B1": 1920, "B2": 3672, "B3": 7152}
# No image of any size costs more than this.
V05_IMAGE_TOKENS_MAX = 7152
# What a call falls back to when the model-input size is not known. NOT the
# bucket's own maximum: fit_to_bucket leaves an image within 2% of the box
# unresized, and one that is up to 2% LARGER than its box is picked up by the
# next pinpoint (a "B1" page at 1175x1560 costs 3,280; 1,727 of 438,814 training
# image rows cost more than their bucket's maximum this way: B1 1,437, B2 276,
# B0 14, up to 7,152 for a "B2" page).
# So an unknown size is charged the global maximum, whatever the bucket.
V05_IMAGE_TOKENS_FALLBACK = V05_IMAGE_TOKENS_MAX


# Calls that were charged the fallback because no size was known (/health).
_size_fallbacks = 0

# Runaway cut in OUTPUT tokens, per bucket and for text pages, measured on the
# training data rather than picked. A completion longer than any label the
# model was trained on for that kind of input is a loop or a runaway, whatever
# its characters look like.
#
# DERIVED from the v0.5 training set (480,513 rows; bucket chosen by the same
# routing the server uses: B0 5,656 / B1 258,804 / B2 122,988 / B3 51,366),
# assistant labels
# tokenized with the served tokenizer. structured_extraction label tokens:
#            n        p50     p99    p99.9     max
#   B0      4,828     897   3,618    4,788    5,619
#   B1    257,635     777   2,960    5,233   22,887
#   B2    120,317   1,279  12,041   16,443   26,389
#   B3     51,034     887  10,080   18,423   24,041
#   text   18,986     826   3,440    4,220    6,363
# Cut = training max + 10%, rounded up to the next 500. The largest bucket does
# not have the longest labels: B3 pays up to 7,152 tokens for its image, so the
# densest pages the model saw whole are B2 (and B1) pages.
# Cross-check, so that no real page is cut: of 17,269 pages the server returned
# ok with a known bucket (two benchmark sets, three checkpoints) none is above its
# bucket's TRAINING max (B0 2,812 / B1 18,409 / B2 13,247 / B3 14,287).
# The cut never exceeds the call's max_tokens (30,000 - prompt), which stays the
# hard ceiling; for B2 and most B3 pages that ceiling is the lower of the two.
# classify_figure (label = 1 token) and fix_headings (max 497) keep their own
# small max_tokens (8 / 4,096) and need no cut.
V05_OUTPUT_TOKEN_CUT = {"B0": 6500, "B1": 25500, "B2": 29500, "B3": 26500, "text": 7000}
V05_OUTPUT_TOKEN_CUT_ENV_ERROR = None       # shown in /health; a bad value never stops the server
if os.environ.get("SHREW_V05_OUTPUT_TOKEN_CUT"):
    import json as _json
    try:
        _env_cut = _json.loads(os.environ["SHREW_V05_OUTPUT_TOKEN_CUT"])
        _unknown = sorted(set(_env_cut) - set(V05_OUTPUT_TOKEN_CUT))
        if _unknown:
            raise ValueError(f"unknown keys {_unknown} (use B0 B1 B2 B3 text)")
        V05_OUTPUT_TOKEN_CUT.update({k: int(v) for k, v in _env_cut.items()})
    except Exception as _e:                 # noqa: BLE001
        V05_OUTPUT_TOKEN_CUT_ENV_ERROR = f"SHREW_V05_OUTPUT_TOKEN_CUT ignored, defaults kept: {_e}"
        import logging as _logging
        _logging.getLogger(__name__).error(V05_OUTPUT_TOKEN_CUT_ENV_ERROR)


def v05_output_token_cut(kind: str, bucket: str | None = None,
                         image_tokens: int | None = None) -> int | None:
    """The runaway cut (output tokens) for one extraction call, or None (0 = off).

    Text pages have one value. An image page takes its bucket's value, or the
    value of the bucket its image-token count belongs to when that is larger:
    a page named B1 that sits up to 2% outside its box is tokenized as a B2
    page (see V05_IMAGE_TOKENS_FALLBACK), and a real page is never to be cut.
    Without a known bucket the largest value applies: the image cost alone does
    not tell the bucket (a wide B2 strip can cost fewer tokens than a B0 page).
    """
    if kind == "text":
        return V05_OUTPUT_TOKEN_CUT["text"] or None
    if bucket not in V05_BUCKET_IMAGE_TOKENS:
        return max(V05_OUTPUT_TOKEN_CUT[b] for b in V05_BUCKET_IMAGE_TOKENS) or None
    cuts = [V05_OUTPUT_TOKEN_CUT[bucket]]
    if image_tokens:
        by_cost = next((b for b in ("B0", "B1", "B2", "B3")
                        if image_tokens <= V05_BUCKET_IMAGE_TOKENS[b]), "B3")
        cuts.append(V05_OUTPUT_TOKEN_CUT[by_cost])
    return max(cuts) or None


# v0.5 text-page bound in TOKENS (2026-11-14), next to the 11,000-character one.
# Characters do not bound tokens (11,000 CJK characters are 11-16k tokens), and
# the model was trained to extract text faithfully only up to a few thousand:
# of the 18,986 structured_extraction text rows of the v0.5 training set, the 18,663
# whose label is at least 0.8x the prompt (a real extraction) have prompt
# tokens p50 579 / p99 2,772 / p99.9 3,455 / max 4,382. The 253 rows whose
# label is under half the prompt (prompts up to 9,601) are not that: 210 are
# markdown with an embedded base64 image, which the label leaves out, and the
# rest are long filings with abridged labels.
# Bound = that p99.9, rounded up: 3,500 (13 of 18,663 faithful rows are above
# it). At the largest expansion training shows for such prompts (label 1.81x
# the prompt) a page at the bound writes 6,335 tokens, under the 7,000 text
# cut; prompt + cut = 3,500 + 18 + 7,000, far inside the 30,000 budget.
V05_TEXT_PAGE_MAX_TOKENS = int(os.environ.get("SHREW_V05_TEXT_PAGE_TOKENS", "3500"))


class V05BudgetError(ValueError):
    """The prompt alone leaves no room for output under the v0.5 token budget."""


def _aliases() -> dict:
    """``SHREW_CONTRACT_ALIASES="name=v0.5,other=v0.4"``: extra names a deployment
    accepts for a contract (older tooling that sends its own labels). Unset = none.
    An entry that does not map to a known contract is ignored."""
    out = {}
    for item in os.environ.get("SHREW_CONTRACT_ALIASES", "").split(","):
        if not item.strip():
            continue
        name, _, target = item.partition("=")
        name, target = name.strip().lower(), target.strip().lower()
        if name and target in CONTRACTS:
            out[name] = target
        elif item not in _bad_aliases:
            _bad_aliases.add(item)
            import logging
            logging.getLogger(__name__).warning(
                "SHREW_CONTRACT_ALIASES: ignoring %r (use name=%s)", item, "|".join(CONTRACTS))
    return out


_bad_aliases: set = set()
_PATCH = re.compile(r"(v\d+\.\d+)\.\d+")


def validate_contract(value) -> str:
    """A contract name. A full model version is accepted for its release line
    (``v0.4.1`` -> ``v0.4``), then a deployment alias."""
    v = (str(value).strip().lower() if value is not None else "")
    m = _PATCH.fullmatch(v)
    if m and m.group(1) in CONTRACTS:
        v = m.group(1)
    if v not in CONTRACTS:
        v = _aliases().get(v, v)
    if v not in CONTRACTS:
        raise ValueError(f"contract must be one of {list(CONTRACTS)}, got {value!r}")
    return v


# The server default follows the current model release. A deployment that still
# serves a v0.4 model sets SHREW_CONTRACT=v0.4.
DEFAULT_CONTRACT = V05


def default_contract() -> str:
    # An empty value (an unset ${SHREW_CONTRACT} in a compose file) means unset.
    return validate_contract(os.environ.get("SHREW_CONTRACT", "").strip() or DEFAULT_CONTRACT)


def v05_plan(command: str, *, size: tuple[int, int] | None = None, text: str | None = None,
             cap: int | None = None) -> dict:
    """Token plan of one v0.5 call: ``{prompt_tokens, image_tokens, max_tokens,
    total, exact}`` with ``total = prompt_tokens + max_tokens <= V05_TOKEN_BUDGET``
    on every path (training: max_tokens = 30000 - prompt tokens).

    Text call: ``text`` is tokenized inside the rendered chat template. Image
    call: the exact image-token count of the model-input ``size`` (w, h); with
    no size, the global maximum (V05_IMAGE_TOKENS_FALLBACK, fail-safe).
    ``cap`` lowers max_tokens, never raises it. Raises V05BudgetError when the
    prompt leaves no room for a single output token.
    """
    from . import v05_tokens
    if text is not None:
        image = 0
        prompt = v05_tokens.text_prompt_tokens(command, text)
        exact = v05_tokens.status()["exact"]
    else:
        exact = size is not None
        if not exact:
            global _size_fallbacks
            _size_fallbacks += 1
        image = v05_image_tokens(*size) if exact else V05_IMAGE_TOKENS_FALLBACK
        prompt = v05_tokens.image_prompt_tokens(command, image)
        exact = exact and v05_tokens.status()["exact"]
    room = V05_TOKEN_BUDGET - prompt
    if room < 1:
        raise V05BudgetError(f"prompt is {prompt} tokens; the v0.5 budget is {V05_TOKEN_BUDGET}")
    max_tokens = max(1, min(room, cap)) if cap else room
    return {"prompt_tokens": prompt, "image_tokens": image, "max_tokens": max_tokens,
            "total": prompt + max_tokens, "exact": exact}


def v05_budget_status() -> dict:
    """What the v0.5 budget is counted with (/health)."""
    from . import v05_tokens
    return {"token_budget": V05_TOKEN_BUDGET, "tokenizer": v05_tokens.status(),
            "text_page_max_tokens": V05_TEXT_PAGE_MAX_TOKENS,
            "image_tokens_max_per_bucket": dict(V05_BUCKET_IMAGE_TOKENS),
            "image_tokens_when_size_unknown": V05_IMAGE_TOKENS_FALLBACK,
            "calls_with_unknown_image_size": _size_fallbacks,
            "output_token_cut_env_error": V05_OUTPUT_TOKEN_CUT_ENV_ERROR}


def v05_max_tokens(command: str = CMD_EXTRACT, *, size: tuple[int, int] | None = None,
                   text: str | None = None, cap: int | None = None) -> int:
    """max_tokens for one v0.5 call (see ``v05_plan``)."""
    return v05_plan(command, size=size, text=text, cap=cap)["max_tokens"]


# --------------------------------------------------------------------------- boxes

def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_quad(bb) -> bool:
    return (isinstance(bb, list) and len(bb) == 4
            and all(isinstance(p, list) and len(p) == 2 and all(_num(c) for c in p) for p in bb))


def is_flat(bb) -> bool:
    return isinstance(bb, list) and len(bb) == 4 and all(_num(c) for c in bb)


def flat_to_quad(bb) -> list:
    """[x0, y0, x1, y1] -> its rectangle's corners TL, TR, BR, BL."""
    x0, y0, x1, y1 = bb
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def envelope(bb):
    """Axis-aligned [x0, y0, x1, y1] around a quad or a flat box; None stays None.

    The internal geometry (cropping, table stitching, fragment padding) works on
    rectangles; the API ships the box exactly as the contract emits it."""
    if bb is None:
        return None
    if is_quad(bb):
        xs = [p[0] for p in bb]
        ys = [p[1] for p in bb]
        return [min(xs), min(ys), max(xs), max(ys)]
    if is_flat(bb):
        return list(bb)
    return None


# --------------------------------------------------------------------------- v0.5 output schema

FIVE_KEYS = ("metadata", "summary", "semantic_chunks", "figures", "tables")
META_FIELDS = ("title", "authors", "organization", "year", "doc_type")


def chart_data_ok(cd) -> bool:
    """The training label validator's chart_data rule, the same in effect."""
    if not (isinstance(cd, dict) and {"header", "rows"} <= set(cd)
            and set(cd) <= {"header", "rows", "values", "precision"}):
        return False
    h, rows = cd["header"], cd["rows"]
    if not (isinstance(h, list) and h and all(isinstance(x, str) for x in h)):
        return False
    if not (isinstance(rows, list) and rows and all(
            isinstance(r, list) and len(r) == len(h) and all(isinstance(c, str) for c in r)
            for r in rows)):
        return False
    v, pr = cd.get("values", "printed"), cd.get("precision")
    if v not in ("printed", "estimated_from_graphic"):
        return False
    if v == "estimated_from_graphic":
        return (isinstance(pr, str) and pr.strip() != "") or (_num(pr) and pr > 0)
    return pr is None


def normalize_v05(sj) -> list[str]:
    """Tolerant, in-place repairs of v0.5 model output. Returns notes, one per repair.

    Model output is graded tolerantly (the training label validator is the CLOSED
    schema for labels; output gets allow_empty and no unknown-field failure):
      * a flat 4-number box is converted to its rectangle's corners — v0.5 ships
        one box shape, never two;
      * a malformed ``chart_data`` is dropped (the figure itself stands).
    """
    notes: list[str] = []
    if not isinstance(sj, dict):
        return notes
    for lk in ("figures", "tables"):
        for o in (sj.get(lk) or []):
            if not isinstance(o, dict):
                continue
            if is_flat(o.get("bbox")):
                o["bbox"] = flat_to_quad(o["bbox"])
                notes.append(f"{lk}:flat-bbox->quad")
            if lk == "figures" and "chart_data" in o and not chart_data_ok(o["chart_data"]):
                del o["chart_data"]
                notes.append("figures:chart_data-malformed-dropped")
    return notes


def validate_schema_v05(sj) -> tuple[bool, list[str]]:
    """v0.5 5-key conformance: v0.4's presence/type checks with quad boxes."""
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
            errs.append("chunk:not-dict")
            continue
        for f in ("chunk_id", "title", "content"):
            if f not in c:
                errs.append(f"chunk-missing:{f}")
    for lk in ("figures", "tables"):
        for o in (sj.get(lk) or []):
            if not isinstance(o, dict):
                errs.append(f"{lk}-item:not-dict")
                continue
            bb = o.get("bbox")
            if bb is not None and not is_quad(bb):
                errs.append(f"{lk}:bad-bbox")
            if lk == "tables" and not isinstance(o.get("html", ""), str):
                errs.append("table:bad-html-type")
    return (len(errs) == 0), errs


def enforcement_schema_v05(section_enum) -> dict:
    """JSON schema for the v0.5 enforced retry: the v0.4 schema with quad boxes
    and an optional chart_data on figures."""
    point = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    quad = {"type": ["array", "null"], "items": point, "minItems": 4, "maxItems": 4}
    return {
        "type": "object",
        "required": list(FIVE_KEYS),
        "properties": {
            "metadata": {
                "type": "object",
                "required": list(META_FIELDS),
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "authors": {"type": ["array", "null"], "items": {"type": "string"}},
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
                        "section_type": {"enum": sorted(section_enum)},
                    },
                },
            },
            "figures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": quad,
                        "caption": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                        "chart_data": {
                            "type": "object",
                            "required": ["header", "rows"],
                            "properties": {
                                "header": {"type": "array", "items": {"type": "string"}},
                                "rows": {"type": "array", "items": {
                                    "type": "array", "items": {"type": "string"}}},
                                "values": {"enum": ["printed", "estimated_from_graphic"]},
                                "precision": {"type": ["string", "number"]},
                            },
                        },
                    },
                },
            },
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": quad,
                        "caption": {"type": ["string", "null"]},
                        "html": {"type": "string"},
                        "description": {"type": ["string", "null"]},
                    },
                },
            },
        },
    }


# --------------------------------------------------------------------------- tool commands

FIGURE_VERDICTS = ("figure", "artifact")


def parse_figure_verdict(text: str) -> str | None:
    """classify_figure answers the bare word ``figure`` or ``artifact``."""
    w = (text or "").strip().strip(".").strip().lower()
    return w if w in FIGURE_VERDICTS else None


def heading_text(line: str) -> str:
    return line.lstrip("#").strip()


def parse_fixed_headings(sent: list[str], text: str) -> list[str] | None:
    """fix_headings returns the SAME heading lines, in order, with levels fixed.

    Accepted only if the line count and every heading's text are unchanged and
    each line is still a markdown heading; anything else is rejected whole
    (None) — the model may re-level a heading, never rewrite or drop one."""
    out = [ln for ln in (text or "").strip("\n").split("\n")]
    if len(out) != len(sent):
        return None
    for a, b in zip(sent, out):
        lv = len(b) - len(b.lstrip("#"))
        if not (1 <= lv <= 6) or not b[lv:lv + 1] == " " or heading_text(a) != heading_text(b):
            return None
    return out
