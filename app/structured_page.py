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
import hashlib
import inspect
from dataclasses import dataclass, replace
import logging
import os
import re
import threading
import time
import zlib
from pathlib import Path

from . import contract as _ct
from . import loop_content
from .contract import V05, V04
from .generation import get_generation_params
from .section_types import SECTION_SET, canonical
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
# v0.5: max_tokens is the context budget minus the prompt (contract.v05_max_tokens);
# this optional cap (0 = none) only ever lowers it.
V05_MAX_TOKENS_CAP = int(os.environ.get("SHREW_V05_MAX_TOKENS", "0")) or None

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

# The model's TRAINED taxonomy (v0.3.14, GitLab #27). section_types.py is
# vendored raw-identical from shrew_ocr/section_types.py — THE contract the
# labels, the eval and this server all derive from (36 values, 2026-09-14);
# tests/test_section_taxonomy.py fails if the copies diverge. validate_schema
# gates on SECTION_ENUM AND the enforced-retry grammar (ENFORCEMENT_SCHEMA)
# derives from it, so the two can never disagree again.
#
# History: the server carried only the 8-value v2 set while 100% of broadsheet
# rows and ~99% of dense rows were trained on the wider taxonomy. A CORRECT
# first pass therefore failed schema (496/541 errors = `body`, 38 =
# `news_article`), the retry grammar forced a wrong v2 value, and the page
# landed "coerced" with mislabeled chunks after a second full generation —
# 1,138 pages, 13% of the corpus, most of the newspaper first-pass loss.
# Never map labels DOWN: the model is trained on these values.
SECTION_ENUM = SECTION_SET

# A section_type outside SECTION_ENUM is normalized in place by canonical()
# (legacy/near-duplicate labels fold onto their canonical value, e.g.
# article→news_article, contents→index; anything unmatched and a missing value
# land on SECTION_FALLBACK) and logged in the gate log — the page is NEVER
# failed for it. A stray label is a labeling-drift signal, not a reason to
# burn a second full generation.
SECTION_FALLBACK = canonical(None)
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


# Retry-tier presence penalty (0 disables). v0.4.0 default 0.3 — the retry runs
# the first-pass contract at the same penalty, and the sampled retry's fresh
# seed is what makes it differ from try 1. (The v0.3.x greedy ladder used 0.6
# with enforcement: 12/12 loop-failed pages rescued, VHR 2026-08-16; set
# SHREW_RETRY_PP=0.6 to restore it.)
RETRY_PP = float(os.environ.get("SHREW_RETRY_PP", "0.3"))


# v0.5 decodes with the SAME recipe and knobs as v0.4: greedy decoding is never
# a production default. Only the enforcement schema differs, for
# when an operator opts into the enforced retry: v0.5 boxes are quads.
ENFORCEMENT_SCHEMA_V05 = _ct.enforcement_schema_v05(SECTION_ENUM)


def enforcement_params(contract: str = V04) -> dict:
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
    schema = ENFORCEMENT_SCHEMA_V05 if contract == V05 else ENFORCEMENT_SCHEMA
    pp = RETRY_PP
    out: dict = {}
    if dialect in ("both", "vllm"):
        out["structured_outputs"] = {"json": schema}
    if dialect in ("both", "llamacpp"):
        out["json_schema"] = schema
    if pp:
        out["presence_penalty"] = pp
    return out


# --------------------------------------------------------------------------- decode policy (temperature / seed)

# Operator knobs for the first-pass and retry sampling temperature and the
# sampling seed (2026-09-26). v0.4.0 DEFAULTS are the recipe shipped with
# shrew-ocr-preview v0.4 (two sampled attempts, then the fallback client when
# one is configured): try 1 at temperature 0.4 / presence penalty 0.3, no
# same-config re-roll, try 2 un-enforced at 0.4 / 0.3, fixed seeds. The v0.3.x
# greedy ladder is one env block away (README "Decode recipe"); its payloads are
# pinned byte for byte by tests/fixtures/decode_payloads_v0315.json.
#
#   SHREW_TRY1_TEMP    first pass AND its same-config re-rolls   (default 0.4)
#   SHREW_TRY2_TEMP    the retry                                 (default 0.4)
#   SHREW_SEED_POLICY  fixed (default) | random | <integer>
#   SHREW_TRY2_ENFORCE retry keeps schema enforcement (default OFF). Off: the
#                      retry uses the first-pass contract (no grammar) at
#                      SHREW_TRY2_TEMP with the retry presence penalty
#                      (SHREW_RETRY_PP). On: the v0.3.x enforced retry.
#
# fixed   -> seed derived from the page input (the model-input image, or the
#            text page) plus the attempt index: a rerun of the same page
#            reproduces, and each later attempt is a genuinely different sample.
# random  -> no seed sent; the backend samples freshly every time.
# <int>   -> that seed on every attempt (a re-roll then repeats its sample).
#
# A seed only exists for a SAMPLED attempt (temperature > 0); a greedy attempt
# never carries one, so the policy is inert at the defaults. Per-request
# overrides (/v1/convert `temperature`, `retry_temperature`, `seed_policy`) are
# validated against the same rules; an invalid value is a 400, never a fallback.
SEED_POLICIES = ("fixed", "random")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
MAX_TEMPERATURE = 2.0
_SEED_MOD = 2 ** 31


def parse_temperature(value, name: str = "temperature") -> float:
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in [0, {MAX_TEMPERATURE}], got {value!r}") from None
    if not (0.0 <= t <= MAX_TEMPERATURE):
        raise ValueError(f"{name} must be in [0, {MAX_TEMPERATURE}], got {t}")
    return t


def parse_bool(value, name: str) -> bool:
    v = str(value).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, on/off, got {value!r}")


def parse_seed_policy(value, name: str = "seed_policy"):
    """'fixed' | 'random' | a non-negative integer below 2**31."""
    v = str(value).strip().lower()
    if v in SEED_POLICIES:
        return v
    try:
        n = int(v)
    except ValueError:
        raise ValueError(f"{name} must be 'fixed', 'random' or an integer, got {value!r}") from None
    if not (0 <= n < _SEED_MOD):
        raise ValueError(f"{name} integer must be in [0, {_SEED_MOD}), got {n}")
    return n


@dataclass(frozen=True)
class DecodePolicy:
    try1_temp: float = 0.4
    try2_temp: float = 0.4
    seed_policy: object = "fixed"          # "fixed" | "random" | int
    try2_enforce: bool = False
    contract: str = V04                    # "v0.4" | "v0.5" (contract.py)
    figure_check: bool = False             # v0.5: classify_figure on every figure crop
    heading_fix: bool = False              # v0.5: fix_headings over the document outline

    @classmethod
    def from_env(cls, contract: str | None = None) -> "DecodePolicy":
        """Env defaults for ``contract`` (None = SHREW_CONTRACT).

        BOTH contracts decode with v0.4.0's shipped recipe from the same knobs
        (SHREW_TRY1_TEMP / SHREW_TRY2_TEMP / SHREW_SEED_POLICY /
        SHREW_TRY2_ENFORCE, stage "structured_page"). Greedy is never a
        default: it is an explicit opt-in only (temperature=0 per request, or
        the env knobs). v0.5 adds figure_check and heading_fix, on by default."""
        contract = _ct.validate_contract(contract) if contract is not None else _ct.default_contract()
        if contract == V05:
            return replace(cls.from_env(V04), contract=V05,
                           figure_check=parse_bool(os.environ.get("SHREW_FIGURE_CHECK", "1"), "SHREW_FIGURE_CHECK"),
                           heading_fix=parse_bool(os.environ.get("SHREW_HEADING_FIX", "1"), "SHREW_HEADING_FIX"))
        return cls(
            try1_temp=parse_temperature(os.environ.get("SHREW_TRY1_TEMP", "0.4"), "SHREW_TRY1_TEMP"),
            try2_temp=parse_temperature(os.environ.get("SHREW_TRY2_TEMP", "0.4"), "SHREW_TRY2_TEMP"),
            seed_policy=parse_seed_policy(os.environ.get("SHREW_SEED_POLICY", "fixed"), "SHREW_SEED_POLICY"),
            try2_enforce=parse_bool(os.environ.get("SHREW_TRY2_ENFORCE", "0"), "SHREW_TRY2_ENFORCE"),
        )

    @property
    def stage(self) -> str:
        return "structured_page"            # one sampling recipe for both contracts

    def with_overrides(self, *, temperature=None, retry_temperature=None,
                       seed_policy=None, retry_enforce=None, contract=None,
                       figure_check=None, heading_fix=None) -> "DecodePolicy":
        """Per-request overrides; None keeps the value already in force.

        A ``contract`` different from the one in force first resets every knob
        to that contract's env defaults; the other overrides then apply."""
        base = self
        if contract is not None:
            c = _ct.validate_contract(contract)
            if c != self.contract:
                base = DecodePolicy.from_env(c)
        self = base
        fc = self.figure_check if figure_check is None else parse_bool(figure_check, "figure_check")
        hf = self.heading_fix if heading_fix is None else parse_bool(heading_fix, "heading_fix")
        if self.contract != V05 and (fc or hf):
            raise ValueError("figure_check / heading_fix require contract=v0.5")
        return replace(
            self,
            figure_check=fc,
            heading_fix=hf,
            try1_temp=self.try1_temp if temperature is None else parse_temperature(temperature),
            try2_temp=(self.try2_temp if retry_temperature is None
                       else parse_temperature(retry_temperature, "retry_temperature")),
            seed_policy=self.seed_policy if seed_policy is None else parse_seed_policy(seed_policy),
            try2_enforce=(self.try2_enforce if retry_enforce is None
                          else parse_bool(retry_enforce, "retry_enforce")),
        )

    def seed_for(self, input_key: str, attempt_index: int, temperature: float):
        """The seed to send for one attempt, or None to send no seed field."""
        if not temperature or self.seed_policy == "random":
            return None
        if isinstance(self.seed_policy, int):
            return self.seed_policy
        digest = hashlib.sha256(f"{input_key}:{attempt_index}".encode()).digest()
        return int.from_bytes(digest[:4], "big") % _SEED_MOD

    def retry_extra(self, first_pass_extra) -> dict:
        """extra_params for the retry rung, before any seed. Enforced (default):
        the first-pass extras overlaid with enforcement_params() — the shipped
        merge. Un-enforced: the same overlay minus the schema-enforcement keys,
        i.e. the first-pass contract at the retry presence penalty."""
        merged = {**(first_pass_extra or {}), **enforcement_params(self.contract)}
        if not self.try2_enforce:
            merged.pop("structured_outputs", None)
            merged.pop("json_schema", None)
        return merged

    def describe(self) -> dict:
        """The whole per-document recipe, so a log check can verify it."""
        from .generation import get_generation_params
        fp = get_generation_params("", self.stage)["extra_params"] or {}
        out = {"try1_temperature": self.try1_temp, "try2_temperature": self.try2_temp,
               "seed_policy": self.seed_policy,
               "try1_pp": fp.get("presence_penalty", 0.0),
               "retry_pp": RETRY_PP,
               "try2_enforce": self.try2_enforce}
        if self.contract == V05:
            # v0.4 keeps the v0.4.1 keys exactly; v0.5 names itself.
            out.update(contract=V05, max_tokens="budget", token_budget=_ct.V05_TOKEN_BUDGET,
                       figure_check=self.figure_check, heading_fix=self.heading_fix)
        return out


def page_input_key(messages: list[dict]) -> str:
    """Stable hash of a page request's input — for image pages this covers the
    base64 model-input PNG, i.e. it is the image hash."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def _sampling(base_temp, base_extra, decode_temp, decode: "DecodePolicy",
              input_key: str, attempt_index: int):
    """(temperature, extra_params) for one attempt. At temperature 0 this
    returns the stage's own values UNCHANGED (same objects, same JSON: the
    table's int 0 stays `0`, not `0.0`), so the default payload is byte-identical
    to the pre-knob ladder."""
    if not decode_temp:
        return base_temp, base_extra
    extra = dict(base_extra or {})
    seed = decode.seed_for(input_key, attempt_index, decode_temp)
    if seed is not None:
        extra["seed"] = seed
    return decode_temp, (extra or None)


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


def content_zlib_ratio(raw: str) -> float:
    """``zlib_ratio`` of the completion with its markup removed."""
    return zlib_ratio(loop_content.content(raw))


def is_degenerate_v05(raw: str, gate: float = ZLIB_GATE_RATIO) -> bool:
    """The §5.1 gate for v0.5 pages: degenerate only when the completion
    compresses past the gate both as emitted and with its markup removed.

    v0.5 tables are HTML, and a correct dense table is mostly ``<tr><td>``: of
    the 12 normally-finished completions over 7.0 in the 2026-11-14 replay
    corpus (40,741 texts), 6 were correct tables (OmniDocBench page-213e6a57
    at 7.8-8.0, a 67-row tRNA table at 7.0-7.5) whose content alone compresses
    3.7-3.8; the 6 others stay above 7 either way. The second condition can
    only pass a page the raw gate fails, never fail one it passes.
    """
    return zlib_ratio(raw) > gate and content_zlib_ratio(raw) > gate


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
                 attempt: str | None = None,
                 doc_id: str | None = None,
                 long_window: int | None = None,
                 long_threshold: float | None = None,
                 length_cap_chars: int | None = None,
                 periodic_chars: int | None = None,
                 repeat_chars: int | None = None,
                 numeric_repeats: int | None = None,
                 counter_chars: int | None = None,
                 token_cap: int | None = None,
                 tail_chars: int = 0):
        # v0.5 extensions (None = off, the v0.4 guard exactly): a long trailing
        # window for long-period loops, and an output-length cap for slow loops
        # no compression window can see. Derived offline, see V05_LOOP_* below.
        self.long_window = long_window
        self.long_threshold = long_threshold
        self.length_cap_chars = length_cap_chars
        self.long_violations = 0
        self.long_max_ratio = 0.0
        self.abort_guard = None
        # v0.5 content guards (None = off), see loop_content.py and V05_LOOP_*.
        self.periodic_chars = periodic_chars
        self.repeat_chars = repeat_chars
        self.numeric_repeats = numeric_repeats
        # A periodic run of numbered rows (loop_content.periodic) is held to its
        # own, higher threshold; None = the periodic threshold.
        self.counter_chars = counter_chars
        # v0.5 runaway cut in output TOKENS, from the usage the server reports on
        # the stream (``usage()``). While usage arrives it replaces the
        # character cap; a stream that never reports usage keeps the cap.
        self.token_cap = token_cap
        self.completion_tokens = 0
        self.prompt_tokens = None
        self.usage_seen = False
        self.tail_chars = tail_chars
        self.periodic_max = 0
        self.counter_max = 0
        self.repeat_max = 0
        self.numeric_max = 0.0
        self.evidence = None
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
        # Joins guard lines to their document when many documents stream at once.
        self.doc_id = doc_id
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
        if (self.length_cap_chars and n >= self.length_cap_chars
                and not (self.token_cap and self.usage_seen)):
            return self._fire(n, "length_cap", accumulated)
        self.ratio = zlib_ratio(accumulated[-self.window:])
        self.max_ratio = max(self.max_ratio, self.ratio)
        if self.ratio > self.threshold:
            self.violations += 1
            if self.violations >= self.consecutive:
                return self._fire(n, "window", accumulated)
        else:
            self.violations = 0
        if self.long_window and n >= self.long_window:
            lr = zlib_ratio(accumulated[-self.long_window:])
            self.long_max_ratio = max(self.long_max_ratio, lr)
            if lr > self.long_threshold:
                self.long_violations += 1
                if self.long_violations >= self.consecutive:
                    self.ratio = lr
                    return self._fire(n, "long_window", accumulated)
            else:
                self.long_violations = 0
        if self.periodic_chars or self.numeric_repeats:
            per = loop_content.periodic(accumulated)
            limit = self.periodic_chars
            if per["numbered_rows"]:
                self.counter_max = max(self.counter_max, per["score"])
                limit = self.counter_chars or limit
            else:
                self.periodic_max = max(self.periodic_max, per["score"])
            self.numeric_max = max(self.numeric_max, per["numeric_repeats"])
            if self.periodic_chars and per["score"] >= limit:
                return self._fire(n, "periodic", accumulated, score=per["score"],
                                  threshold=limit, detail=per)
            if self.numeric_repeats and per["numeric_repeats"] >= self.numeric_repeats:
                return self._fire(n, "numeric", accumulated, score=per["numeric_repeats"],
                                  threshold=self.numeric_repeats, detail=per)
        if self.repeat_chars:
            dup = loop_content.repeated(accumulated)
            self.repeat_max = max(self.repeat_max, dup["score"])
            if dup["score"] >= self.repeat_chars:
                return self._fire(n, "repeated", accumulated, score=dup["score"],
                                  threshold=self.repeat_chars, detail=dup)
        return None

    def usage(self, usage: dict, accumulated: str = ""):
        """``on_usage`` callback: the server's running token usage for this stream."""
        done = usage.get("completion_tokens")
        if not isinstance(done, int):
            return None
        self.usage_seen = True
        self.completion_tokens = max(self.completion_tokens, done)
        if isinstance(usage.get("prompt_tokens"), int):
            self.prompt_tokens = usage["prompt_tokens"]
        # ">" and not ">=": a page that ends on exactly the cut is complete, and
        # when the cut is the call's max_tokens the server ends it ("length").
        if self.token_cap and done > self.token_cap and not self.fired:
            return self._fire(len(accumulated), "token_cap", accumulated, score=done,
                              threshold=self.token_cap)
        return None

    def _fire(self, n: int, which: str, accumulated: str = "", score=None, threshold=None,
              detail: dict | None = None):
        self.abort_guard = which
        self.fired = True
        self.position = n
        context = self.log_context()
        if self.tail_chars:
            # Evidence: every abort carries what fired, by how
            # much, and the text it fired on, so a false abort can be proven.
            if which == "window":
                score, threshold, window = round(self.ratio, 2), self.threshold, self.window
            elif which == "long_window":
                score, threshold, window = round(self.ratio, 2), self.long_threshold, self.long_window
            elif which == "length_cap":
                score, threshold, window = n, self.length_cap_chars, None
            elif which == "token_cap":
                window = None
            else:
                window = (detail or {}).get("window_chars")
            self.evidence = {"guard": which, "score": score, "threshold": threshold,
                             "window_chars": window, "at_chars": n,
                             "tail": accumulated[-self.tail_chars:]}
            if which in ("periodic", "numeric"):
                self.evidence.update(period_chars=detail["period_chars"], repeats=detail["repeats"],
                                     numbered_rows=detail["numbered_rows"], unit=detail["unit"][:300])
            elif which == "repeated":
                self.evidence.update(novel_share=detail["novel"], top_units=detail["top"])
        if which == "token_cap":
            logger.warning("%srepetition_abort at %d chars: %d output tokens >= the %d-token "
                           "runaway cut (v0.5 token cap)", context, n, score, threshold)
            return REPETITION_ABORT
        if which == "periodic":
            logger.warning("%srepetition_abort at %d chars: periodic tail re-emits %d content chars >= %d "
                           "(period %d chars x %.1f%s) (v0.5 periodic guard)", context, n, score,
                           threshold, detail["period_chars"], detail["repeats"],
                           ", numbered rows" if detail["numbered_rows"] else "")
            return REPETITION_ABORT
        if which == "numeric":
            logger.warning("%srepetition_abort at %d chars: a run of numbers with no letters "
                           "repeats %.1f times >= %d (period %d chars) (v0.5 numeric guard)", context,
                           n, score, threshold, detail["period_chars"])
            return REPETITION_ABORT
        if which == "repeated":
            logger.warning("%srepetition_abort at %d chars: %d of the last %d content chars are "
                           "units already emitted %d+ times >= %d (v0.5 repeated guard)", context, n,
                           score, detail["window_chars"], loop_content.REPEAT_MIN_PRIOR, threshold)
            return REPETITION_ABORT
        if which == "window":            # the v0.4 guard's line, verbatim
            logger.warning(
                "%srepetition_abort at %d chars: trailing-window zlib ratio %.1f > %.1f "
                "for %d consecutive windows",
                context, n, self.ratio, self.threshold, self.violations,
            )
        elif which == "long_window":
            logger.warning(
                "%srepetition_abort at %d chars: %d-char window zlib ratio %.1f > %.1f "
                "for %d consecutive windows (v0.5 long-window guard)",
                context, n, self.long_window, self.ratio, self.long_threshold,
                self.long_violations,
            )
        else:
            logger.warning("%srepetition_abort at %d chars: output passed the %d-char "
                           "length cap (v0.5 runaway guard)", context, n, self.length_cap_chars)
        return REPETITION_ABORT

    def log_context(self) -> str:
        """'Doc <id> Page N (attempt): ' — each part only when known."""
        context = ""
        if self.page_no is not None:
            context = f"Page {self.page_no}"
            if self.attempt:
                context += f" ({self.attempt})"
        if self.doc_id:
            context = f"Doc {self.doc_id} {context}".rstrip()
        return context + ": " if context else ""

    def stats(self) -> dict:
        return {
            "aborted": self.fired,
            "ratio": round(self.ratio, 2),
            "max_window_ratio": round(self.max_ratio, 2),
            "position_chars": self.position,
            "threshold": self.threshold,
            "window_chars": self.window,
            **({"abort_guard": self.abort_guard,
                "long_max_window_ratio": round(self.long_max_ratio, 2)}
               if (self.long_window or self.length_cap_chars or self.periodic_chars
                   or self.repeat_chars or self.numeric_repeats) else {}),
            **({"max_periodic_chars": self.periodic_max, "max_counter_chars": self.counter_max,
                "max_repeat_chars": self.repeat_max, "max_numeric_repeats": self.numeric_max,
                "completion_tokens": self.completion_tokens if self.usage_seen else None,
                "prompt_tokens": self.prompt_tokens, "token_cap": self.token_cap,
                "usage_seen": self.usage_seen,
                "abort_evidence": self.evidence}
               if (self.periodic_chars or self.repeat_chars or self.numeric_repeats) else {}),
        }


# Stop-at-JSON-close (2026-09-20). Belt and braces against "close the
# top-level object, then keep generating until the cap" — a case that is
# THEORETICAL (unobserved in the paired eval, no reward gradient toward it), so
# this is cheap insurance, not a fix. It cannot touch the loop failure, which
# happens INSIDE the JSON before it ever closes: on those pages there is no
# closing brace to stop on and the repetition guard above stays the instrument.
# Serving path only — never the RL rollouts (cutting completions there hid loops
# from the penalty and selected for early-stoppers, the leg-1 rollback) and never
# screen_dense.py (the eval contract). Deliberately NOT grammar-constrained
# decoding, which changes the sampled distribution and costs latency on every
# token for a guard that should almost never fire. Kill switch:
# SHREW_JSON_CLOSE_GUARD=0.
JSON_CLOSE = "json_close"
JSON_CLOSE_GUARD_ENABLED = os.environ.get("SHREW_JSON_CLOSE_GUARD", "1").strip().lower() not in {
    "0", "false", "no", "off",
}


class JsonCloseGuard:
    """Streaming brace-depth counter; fires when the model keeps generating
    AFTER the top-level object has closed.

    Tracks string state and escapes so a ``}`` inside a string value does not
    count. Depth counts both ``{}`` and ``[]``; counting arms at the first
    opener and the guard is inert if that opener was ``[`` (a top-level array
    is not the contract — the parse gate owns it). When depth returns to 0 the
    close is RECORDED (``closed``, ``close_index`` = length of the accumulated
    text through the closing brace) but the stream is not aborted: a model
    that stops on its own produces its natural ``stop`` one delta later, so a
    firing guard means exactly one thing — non-whitespace content arrived
    after the close. That is what keeps the ``json_close`` count informative:
    it is the count of the case the guard exists for, not of every page.
    """

    def __init__(self):
        self.depth = 0
        self.armed = False          # first opener seen
        self.inert = False          # first opener was `[` -> never fire
        self.in_string = False
        self.escape = False
        self.closed = False         # top-level object has closed
        self.close_index = 0        # accumulated length through the closing brace
        self.fired = False          # content arrived after the close
        self.trailing = 0           # non-dropped chars seen after the close at fire time
        self._seen = 0              # chars consumed so far

    def __call__(self, chunk: str, accumulated: str):
        if self.fired or self.inert:
            return None
        base = self._seen
        self._seen += len(chunk)
        for i, ch in enumerate(chunk):
            if self.closed:
                if not ch.isspace():
                    self.fired = True
                    self.trailing = len(accumulated) - self.close_index
                    return JSON_CLOSE
                continue
            if self.in_string:
                if self.escape:
                    self.escape = False
                elif ch == "\\":
                    self.escape = True
                elif ch == '"':
                    self.in_string = False
                continue
            if ch == '"':
                self.in_string = True
            elif ch in "{[":
                if not self.armed:
                    self.armed = True
                    if ch == "[":
                        self.inert = True
                        return None
                self.depth += 1
            elif ch in "}]":
                if not self.armed:
                    continue        # stray closer before any opener: not ours
                self.depth -= 1
                if self.depth == 0:
                    self.closed = True
                    self.close_index = base + i + 1
        return None

    def stats(self) -> dict:
        return {
            "fired": self.fired,
            "closed": self.closed,
            "close_chars": self.close_index,
            "trailing_chars_dropped": self.trailing,
        }


class StreamGuards:
    """Composite ``on_delta``: the JSON-close guard first (cheap, exact), then
    the repetition guard. On a never-closing loop the first never fires and the
    second behaves exactly as before; ``stats()`` keeps the repetition guard's
    shape and adds a ``json_close`` block so a fired — or never-fired — close
    guard is countable per page.
    """

    def __init__(self, repetition: "RepetitionGuard", json_close: "JsonCloseGuard | None"):
        self.repetition = repetition
        self.json_close = json_close

    def __call__(self, chunk: str, accumulated: str):
        if self.json_close is not None:
            reason = self.json_close(chunk, accumulated)
            if reason:
                context = self.repetition.log_context()
                logger.warning("%sjson_close at %d chars (%d trailing chars dropped)",
                               context, self.json_close.close_index, self.json_close.trailing)
                return reason
        return self.repetition(chunk, accumulated)

    def usage(self, usage: dict, accumulated: str = ""):
        return self.repetition.usage(usage, accumulated)

    # RepetitionGuard attributes the gate and the ladder read directly.
    @property
    def ratio(self):
        return self.repetition.ratio

    @property
    def position(self):
        return self.repetition.position

    @property
    def fired(self):
        return self.repetition.fired

    def stats(self) -> dict:
        out = self.repetition.stats()
        out["json_close"] = (self.json_close.stats() if self.json_close is not None
                             else {"fired": False, "closed": False, "close_chars": 0,
                                   "trailing_chars_dropped": 0})
        return out


# --------------------------------------------------------------------------- message construction


def v05_image_part(image_path: str | Path) -> dict:
    """v0.5 image part: the shape the model was trained and benchmarked with (no ``detail``)."""
    part = make_image_content(image_path)
    return {"type": "image_url", "image_url": {"url": part["image_url"]["url"]}}


def build_command_messages(command: str, *, image_path=None, text=None) -> list[dict]:
    """v0.5: system = the bare command; user = image-only or ONE text part."""
    if (image_path is None) == (text is None):
        raise ValueError("exactly one of image_path / text")
    part = (v05_image_part(image_path) if image_path is not None
            else {"type": "text", "text": text})
    return [{"role": "system", "content": command},
            {"role": "user", "content": [part]}]


def build_image_messages(image_path: str | Path, contract: str = V04) -> list[dict]:
    """§2 image modality: sentinel system prompt, user content = image ONLY."""
    if contract == V05:
        return build_command_messages(_ct.CMD_EXTRACT, image_path=image_path)
    return [
        {"role": "system", "content": SENTINEL},
        {"role": "user", "content": [make_image_content(image_path)]},
    ]


def build_text_messages(text: str, contract: str = V04) -> list[dict]:
    """§2 text modality: sentinel system prompt, user content = the raw string.

    v0.4: the content is a plain string, not a content-part list — that is the
    shape the v0.4 text arm was trained and Gate-F'd on. v0.5 trains on a list
    holding ONE text part.
    """
    if contract == V05:
        return build_command_messages(_ct.CMD_EXTRACT, text=text)
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


# Which rung produced a page's result. schema_coerced predates the un-enforced
# retry and means "rescued by the retry tier" (kept for the density floor and the
# gate counts); rescued_by is the unambiguous name for the rung.
RESCUED_BY = ("none", "reroll", "retry_enforced", "retry_unenforced", "fallback")


def _result(ok, data, status, error, attempts, raw_len, *,
            schema_coerced=False, degenerate=False, repetition_abort=False,
            loop_guard=None, rescued_by="none", model_calls=None) -> dict:
    return {
        "rescued_by": rescued_by,
        # `attempts` keeps its historical value (2 for every retry-tier result,
        # even after a re-roll); model_calls is the exact number of VLM calls.
        "model_calls": attempts if model_calls is None else model_calls,
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


def validate_types(sj: dict) -> list[str]:
    """Server-side TYPE conformance for the fields validate_schema only checks
    for PRESENCE (GitLab #23). Kept separate so the vendored validate_schema
    stays byte-identical to the eval copy it is synced from.

    The first pass runs unenforced, so the model can emit ``"caption": 42``
    or ``"content": null`` and still satisfy a presence check — then
    assemble_document() dies in table_flat_text()/_is_prose() with a
    TypeError. fallback._coerce_five_key already re-types exactly these
    fields; this guards the primary door the same way, by turning a bad type
    into a "schema" verdict (retry/fallback ladder) instead of a crash.
    Only str-or-None is accepted where the schema means text; null caption
    and null title are legal; a missing keywords key is legal.
    """
    errs: list[str] = []
    for c in (sj.get("semantic_chunks") or []):
        if not isinstance(c, dict):
            continue
        if not isinstance(c.get("content"), str):
            errs.append("chunk:bad-content-type")
        if not isinstance(c.get("title"), (str, type(None))):
            errs.append("chunk:bad-title-type")
        if "keywords" in c and not isinstance(c["keywords"], list):
            errs.append("chunk:bad-keywords-type")
    for lk in ("figures", "tables"):
        for o in (sj.get(lk) or []):
            if not isinstance(o, dict):
                continue
            if not isinstance(o.get("caption"), (str, type(None))):
                errs.append(f"{lk}:bad-caption-type")
            # v0.3.13: description is carried through to the output and the
            # fidelity regex — a non-string there would crash downstream.
            if not isinstance(o.get("description"), (str, type(None))):
                errs.append(f"{lk}:bad-description-type")
    return errs


def normalize_section_types(parsed) -> list[str]:
    """Rewrite any chunk ``section_type`` outside SECTION_ENUM (or missing) to
    its canonical() value, in place. Returns the original stray values, one per
    chunk touched, and logs each so the drift stays visible in the gate log
    without burning a second full generation on the page (GitLab #27).

    Only dict chunks with a str-or-missing section_type are touched; anything
    else is left for validate_schema/validate_types to report as before.
    """
    stray: list[str] = []
    chunks = parsed.get("semantic_chunks") if isinstance(parsed, dict) else None
    if not isinstance(chunks, list):
        return stray
    for i, c in enumerate(chunks):
        if not isinstance(c, dict):
            continue
        st = c.get("section_type")
        if st in SECTION_ENUM:
            continue
        if st is not None and not isinstance(st, str):
            continue
        c["section_type"] = canonical(st)
        stray.append("<missing>" if st is None else st)
        logger.info("chunk %d: section_type %r not in taxonomy -> %r",
                    i, st, c["section_type"])
    return stray


def _gate(text: str, finish_reason: str | None, guard: "RepetitionGuard | None" = None,
          contract: str = V04):
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
        which = getattr(guard, "abort_guard", None)
        if which in ("periodic", "repeated", "numeric", "length_cap", "token_cap"):
            ev = guard.evidence
            how = f" {ev['score']} >= {ev['threshold']}" if ev else ""
            return None, "degenerate", f"repetition_abort: {which} guard{how} at {pos} chars"
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
    if is_degenerate_v05(text) if contract == V05 else is_degenerate(text):
        return None, "degenerate", f"zlib ratio {zlib_ratio(text):.2f} > {ZLIB_GATE_RATIO}"
    parsed, perr = parse_json_lenient(text)
    if parsed is None:
        return None, "parse", perr
    # GitLab #27: a section_type outside the taxonomy is a labeling-drift
    # signal, not a failed page. Normalize BEFORE validate_schema so the
    # vendored gate stays byte-identical to the eval copy and never sees it.
    normalize_section_types(parsed)
    if contract == V05:
        for note in _ct.normalize_v05(parsed):
            logger.info("v0.5 output repaired: %s", note)
        schema_ok, serrs = _ct.validate_schema_v05(parsed)
    else:
        schema_ok, serrs = validate_schema(parsed)
    if schema_ok:
        # Presence passed; now the field TYPES the downstream assembly relies
        # on (GitLab #23). Same verdict, same ladder.
        serrs = validate_types(parsed)
        schema_ok = not serrs
    if not schema_ok:
        return parsed, "schema", "schema:" + ";".join(serrs)
    return parsed, "ok", None


# v0.5-only loop guard extensions (2026-11-09). DERIVED, not picked:
# replayed offline over 6,606 clean pages (OmniDocBench 1,639 + a second parse
# benchmark 4,635 + 332 v0.5 pages) and 18 v0.5 cap-hit loops. Clean page
# max 8,000-char-window zlib ratio = 9.5 -> threshold 12 (26% margin); clean
# max output = 41,003 chars -> cap 60,000 (46% margin). Result: 0/6,606 clean
# false aborts; 18/18 loops caught (11 by the 2k window, 6 by the 8k window at
# 9.2-18k chars, 1 by the cap). v0.4 never uses them.
#
# 2026-11-14: the guard was rebuilt. The 8k zlib window is OFF by
# default: it aborted a correct 68-row table (OmniDocBench page-213e6a57, ratio
# 12.2 — <td> markup and short cell values compress like a loop). It is
# replaced by three content guards (loop_content.py) that score what is being
# re-emitted instead of how well the markup compresses. The knobs stay so the
# old guard can be replayed (SHREW_V05_LOOP_LONG_WINDOW=8000).
#
# DERIVED by replay (scripts/loopguard_replay.py) over 41,441 saved outputs
# (v0.5 raw dumps and guard-free runs, the 6,606-page pool above, rebuilt
# benchmark responses), 324 of them labelled by reading the
# text. Rule: the threshold sits at >= 2x the largest score of any text read
# as legitimate; loops have no upper bound (they grow until cut), so the loop
# side fixes only how many characters a loop costs before the abort.
#   periodic  content chars re-emitted by an exactly-repeating tail
#             legitimate max 1,431 (a contract paragraph x3; 1,266 for numbered
#             line items, 715 for 40 "[illegible signature]" lines) -> 3,000
#   counter   the same score when the tail is numbered rows (a number counts
#             and each repeat re-emits < 200 content chars): a payment schedule,
#             an answer sheet and a looping list are one shape. Corpus
#             legitimate max 552 (a 10-K exhibit list), so 2x would be ~1,100;
#             but constructed pages score higher: a 120-row payment
#             schedule 3,753, a 300-question answer sheet 3,572, 60 schedule
#             rows with a 43-letter description 3,016 -> 6,000 (1.6x the
#             built max, 11x the corpus max; not a full 2x over the built
#             pages: ~190 such rows on one page abort). This conservative
#             value was kept over 4,500 because it saves good pages; it costs
#             real numbered-row loops 0.1-3.6k chars more
#             than the old guard before the abort.
#   repeated  chars of the last 8,000 in units already emitted 3+ times, while
#             <= 10% of the window is new
#             legitimate max 1,191 (0 on a page of 13 foreclosure notices that
#             share their boilerplate) -> 4,000
#   numeric   cycles of a run of numbers with no letters, or of numbered rows
#             legitimate max in the corpus 39 (a numbered form); built: a
#             "300 chart" number grid 283, 300 answer lines 276 -> 600
# Result on the labelled set (325 texts): clean 0/152 aborted (old guard 2/152,
# both the page above); loops 135/140 (old 126/140); the 5 not reached are dump
# texts the old guard had cut at 9-12k chars while the score was still rising.
# The numbered-row false aborts above were found by constructing such pages;
# they are tests now (test_numbered_rows_of_a_real_page_are_not_a_loop).
#
# The §5.1 whole-document gate has the same blind spot and failed the same
# page after it finished normally: see is_degenerate_v05.
V05_LOOP_LONG_WINDOW = int(os.environ.get("SHREW_V05_LOOP_LONG_WINDOW", "0"))
V05_LOOP_LONG_RATIO = float(os.environ.get("SHREW_V05_LOOP_LONG_RATIO", "12.0"))
V05_LOOP_MAX_CHARS = int(os.environ.get("SHREW_V05_LOOP_MAX_CHARS", "60000"))
V05_LOOP_PERIODIC_CHARS = int(os.environ.get("SHREW_V05_LOOP_PERIODIC_CHARS", "3000"))
V05_LOOP_REPEAT_CHARS = int(os.environ.get("SHREW_V05_LOOP_REPEAT_CHARS", "4000"))
V05_LOOP_COUNTER_CHARS = int(os.environ.get("SHREW_V05_LOOP_COUNTER_CHARS", "6000"))
V05_LOOP_NUMERIC_REPEATS = int(os.environ.get("SHREW_V05_LOOP_NUMERIC_REPEATS", "600"))
V05_LOOP_TAIL_CHARS = int(os.environ.get("SHREW_V05_LOOP_TAIL_CHARS", "2000"))


# Runaway cut in output tokens (2026-11-14). The 60,000-character cap above
# says nothing about tokens (CJK text is ~1 token per character, English ~4), so
# the cut is taken on the completion tokens vLLM reports on the stream. The
# values are per bucket / per text command, from the training labels: see
# V05_OUTPUT_TOKEN_CUT in contract.py. A server that reports no usage keeps the
# character cap, and that is visible: /health `loop_guard_v05.runaway_cut` and
# the startup log say which cut is active and count the streams without usage.
V05_STREAM_USAGE = os.environ.get("SHREW_V05_STREAM_USAGE", "1") not in ("0", "false", "no", "off")
_V05_CUT_LOCK = threading.Lock()
_V05_CUT = {"streams": 0, "streams_with_usage": 0, "streams_without_usage": 0,
            "token_cut_aborts": 0, "char_cap_aborts_without_usage": 0}


def _note_v05_stream(guard: "RepetitionGuard", by_tokens: bool) -> None:
    with _V05_CUT_LOCK:
        _V05_CUT["streams"] += 1
        _V05_CUT["streams_with_usage" if guard.usage_seen else "streams_without_usage"] += 1
        if guard.abort_guard == "token_cap":
            _V05_CUT["token_cut_aborts"] += 1
        elif guard.abort_guard == "length_cap":
            _V05_CUT["char_cap_aborts_without_usage"] += 1
        if not guard.usage_seen and _V05_CUT["streams_without_usage"] in (1, 10, 100, 1000):
            logger.warning("v0.5 runaway cut: the server sent no token usage on a stream (%d so far); "
                           "the %d-character cap is the active cut for those calls",
                           _V05_CUT["streams_without_usage"], V05_LOOP_MAX_CHARS)


def v05_runaway_cut_status() -> dict:
    """Which runaway cut is configured and which one the streams actually got."""
    with _V05_CUT_LOCK:
        c = dict(_V05_CUT)
    configured = "tokens" if V05_STREAM_USAGE else "chars"
    if configured == "chars" or (c["streams_without_usage"] and not c["streams_with_usage"]):
        observed = "chars"
    elif c["streams_with_usage"] and not c["streams_without_usage"]:
        observed = "tokens"
    elif c["streams_with_usage"]:
        observed = "mixed"
    else:
        observed = "unknown (no v0.5 stream yet)"
    return {"configured": configured, "observed": observed,
            "token_cut": dict(_ct.V05_OUTPUT_TOKEN_CUT), "char_cap": V05_LOOP_MAX_CHARS or None, **c}


def v05_guard_params() -> dict:
    """RepetitionGuard keyword arguments of the v0.5 contract (0 = that guard off)."""
    return {"long_window": V05_LOOP_LONG_WINDOW or None,
            "long_threshold": V05_LOOP_LONG_RATIO,
            "length_cap_chars": V05_LOOP_MAX_CHARS or None,
            "periodic_chars": V05_LOOP_PERIODIC_CHARS or None,
            "repeat_chars": V05_LOOP_REPEAT_CHARS or None,
            "numeric_repeats": V05_LOOP_NUMERIC_REPEATS or None,
            "counter_chars": V05_LOOP_COUNTER_CHARS or None,
            "tail_chars": V05_LOOP_TAIL_CHARS}


def _call(messages, client, *, max_tokens, temperature, extra_params, timeout,
          page_no=None, attempt=None, doc_id=None, contract: str = V04,
          token_cap: int | None = None):
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

    v05_ext = v05_guard_params() if contract == V05 else {}
    # Only a client that can pass usage on (VLMClient); any other keeps the character cap.
    by_tokens = (contract == V05 and V05_STREAM_USAGE and bool(token_cap)
                 and "on_usage" in inspect.signature(client.chat_completion_stream).parameters)
    if by_tokens:
        v05_ext["token_cap"] = min(token_cap, max_tokens)
    rep = RepetitionGuard(page_no=page_no, attempt=attempt, doc_id=doc_id, **v05_ext)
    guard = StreamGuards(rep, JsonCloseGuard()) if JSON_CLOSE_GUARD_ENABLED else rep
    more = {"on_usage": guard.usage} if by_tokens else {}
    result = client.chat_completion_stream(
        messages, max_tokens=max_tokens, temperature=temperature,
        extra_params=extra_params, timeout=timeout, on_delta=guard,
        wall_clock_s=STREAM_WALL_CLOCK_S, **more,
    )
    if contract == V05 and token_cap:
        _note_v05_stream(rep, by_tokens)
    choice = result["choices"][0]
    text = choice["message"].get("content") or ""
    finish_reason = choice.get("finish_reason")
    if finish_reason == JSON_CLOSE and isinstance(guard, StreamGuards):
        # The abort lands mid-chunk: keep the document, drop what followed the
        # closing brace in that same chunk. The object itself is untouched.
        text = text[:guard.json_close.close_index]
    return text, finish_reason, guard


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

# Near-blank exemption from the coercion-density gate (v0.5 contract only).
# On a blank page the first pass can hallucinate until the loop guard aborts
# it; the retry then correctly returns almost nothing, and the ratio above
# measures that against the hallucination — the gate failed real blank pages.
# A short retry is believable when the page carries too little ink to hold
# COERCED_MIN_CHARS of text. Measured with page_ink() on the native renders of
# OHR-Bench (8,561 pages) and OmniDocBench (1,651 pages) against ground-truth
# text: no page with >= 300 chars of real text is below 0.0048 (lowest 0.00484,
# OHR; OmniDocBench lowest 0.0070), while visually blank pages — tinted, aged,
# creased or hole-punched paper — reach 0.0044. The bound is that text floor.
BLANK_INK_MAX = float(os.environ.get("SHREW_BLANK_INK_MAX", "0.0048"))
if not 0.0 <= BLANK_INK_MAX < 1.0:   # also rejects nan, which would exempt every page
    raise ValueError(f"SHREW_BLANK_INK_MAX must be a fraction in [0, 1), got "
                     f"{os.environ.get('SHREW_BLANK_INK_MAX')!r}")


def page_ink(img) -> float:
    """Ink on a page image: the fraction of pixels darker than their local
    background, with glyph_height's detector (adaptive Gaussian threshold,
    31 px window, C=10, measured at <= 2600 px on the long side). Paper tint,
    yellowing and gradual shading read as background; text, rules and drawings
    read as ink. ``img`` is a PIL image or a path; measure the NATIVE render,
    not the enhanced model image (contrast enhancement lifts paper grain)."""
    import cv2 as _cv
    import numpy as _np
    from PIL import Image as _Image
    if not isinstance(img, _Image.Image):
        with _Image.open(img) as _im:
            return page_ink(_im.convert("L"))
    im = img.convert("L")
    w, h = im.size
    s = min(1.0, 2600 / max(w, h))
    if s < 1.0:
        im = im.resize((int(w * s), int(h * s)), _Image.BOX)
    g = _cv.adaptiveThreshold(_np.asarray(im), 255, _cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              _cv.THRESH_BINARY_INV, 31, 10)
    return float(_np.count_nonzero(g)) / g.size


def _blank_page_ink(ink_source) -> float | None:
    """page_ink of ``ink_source``, or None when there is none or it can't be read
    (the gate then applies unchanged)."""
    if ink_source is None:
        return None
    try:
        return page_ink(ink_source)
    except Exception as e:  # noqa: BLE001 — never fail a page on the measurement
        logger.warning("near-blank check: cannot measure ink on %s (%s)", ink_source,
                       type(e).__name__)
        return None


# Opt-in raw-completion dump: v0.5 only, off unless the caller
# sets a directory. One JSON line per model call, appended to <dir>/<doc_id>.jsonl,
# so benchmark sweeps capture loop texts for offline guard replay. Never in v0.4.
RAW_DUMP_DIR = os.environ.get("SHREW_RAW_DUMP_DIR", "").strip() or None
_dump_lock = threading.Lock()


def _dump_raw(contract, doc_id, page_no, rung, text, finish_reason, verdict) -> None:
    if not RAW_DUMP_DIR or contract != V05:
        return
    try:
        os.makedirs(RAW_DUMP_DIR, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(doc_id or "nodoc"))[:80] + ".jsonl"
        line = json.dumps({"doc_id": doc_id, "page": page_no, "rung": rung,
                           "finish_reason": finish_reason, "verdict": verdict,
                           "chars": len(text), "text": text}, ensure_ascii=False)
        with _dump_lock, open(os.path.join(RAW_DUMP_DIR, name), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:  # noqa: BLE001 — a diagnostic dump never fails a page
        logger.warning("raw dump failed: %s", e)


def _rung(trail, rung, t0, text, finish_reason, verdict, guard, max_tokens, temperature,
          dump=None):
    """One attempt's record for the per-page trail (processing_log.pages, v0.5)."""
    if dump:
        _dump_raw(dump[0], dump[1], dump[2], rung, text, finish_reason, verdict)
    st = guard.stats() if guard is not None and hasattr(guard, "stats") else {}
    trail.append({
        "rung": rung, "finish_reason": finish_reason, "verdict": verdict,
        "chars": len(text), "max_tokens": max_tokens, "temperature": temperature,
        "seconds": round(time.monotonic() - t0, 2),
        "abort_at_chars": (st.get("position_chars") if finish_reason == REPETITION_ABORT else None),
        "abort_guard": (st.get("abort_guard") or "window") if finish_reason == REPETITION_ABORT else None,
        "max_window_ratio": st.get("max_window_ratio"),
        "long_max_window_ratio": st.get("long_max_window_ratio"),
        "max_periodic_chars": st.get("max_periodic_chars"),
        "max_counter_chars": st.get("max_counter_chars"),
        "max_repeat_chars": st.get("max_repeat_chars"),
        "max_numeric_repeats": st.get("max_numeric_repeats"),
        # tokens the server reported on the stream (None: it reported none)
        "prompt_tokens": st.get("prompt_tokens"), "completion_tokens": st.get("completion_tokens"),
        "token_cap": st.get("token_cap"),
        # guard, score vs threshold, window and the trailing text of the abort
        "abort_evidence": st.get("abort_evidence") if finish_reason == REPETITION_ABORT else None,
    })


def _extract(messages: list[dict], client, *, max_tokens: int, timeout=None,
             min_content_chars: int = 0, page_no: int | None = None,
             decode: "DecodePolicy | None" = None, doc_id: str | None = None,
             token_cap: int | None = None, ink_source=None) -> dict:
    """The ladder below, plus ``trail``: one record per model call (rung,
    finish_reason, verdict, chars, abort position, seconds)."""
    trail: list = []
    res = _extract_ladder(messages, client, max_tokens=max_tokens, timeout=timeout,
                          min_content_chars=min_content_chars, page_no=page_no,
                          decode=decode, doc_id=doc_id, trail=trail, token_cap=token_cap,
                          ink_source=ink_source)
    res["trail"] = trail
    return res


def _extract_ladder(messages: list[dict], client, *, max_tokens: int, timeout=None,
                    min_content_chars: int = 0, page_no: int | None = None,
                    decode: "DecodePolicy | None" = None, doc_id: str | None = None,
                    trail: list, token_cap: int | None = None, ink_source=None) -> dict:
    """Run the §3 first pass + at most one flagged enforcement retry.

    ``decode`` carries the temperature/seed knobs (env defaults when None);
    at the defaults every request is identical to the pre-knob ladder."""
    decode = decode or DecodePolicy.from_env()
    params = get_generation_params(client.model, decode.stage)
    input_key = page_input_key(messages)
    t1, x1 = _sampling(params["temperature"], params["extra_params"],
                       decode.try1_temp, decode, input_key, 0)

    # ── First pass: minimal params, no enforcement ──────────────────────────
    _t0 = time.monotonic()
    text, finish_reason, guard = _call(
        messages, client,
        max_tokens=max_tokens,
        temperature=t1,
        extra_params=x1,
        timeout=timeout,
        doc_id=doc_id, contract=decode.contract, token_cap=token_cap,
        page_no=page_no, attempt="attempt 1/2, first pass",
    )
    parsed, verdict, error = _gate(text, finish_reason, guard, decode.contract)
    _dump = (decode.contract, doc_id, page_no)
    _rung(trail, "first", _t0, text, finish_reason, verdict, guard, max_tokens, t1, _dump)
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

    # ── Re-roll tier: repetition_abort is a loop the streaming guard caught.
    # The old assumption "greedy is deterministic, so an identical request
    # reproduces the identical failure" does NOT hold on a batched backend:
    # measured on the gfx908 TP=4 stack, the same request decodes clean on a
    # different batch composition (the loop RATE holds; WHICH borderline pages
    # loop shuffles). The enforced pp-0.6 retry never recovers these dense
    # loops (2/74 → 0/72), so on a repetition_abort we re-run the SAME
    # first-pass config first — the only thing that recovers this class. N is
    # tunable (SHREW_ABORT_REROLL); the enforced retry stays as the last rung.
    # v0.4.0 default 0: under the sampled recipe try 2 is already a fresh
    # sample of the same contract, so the re-roll rung is folded into it
    # (two attempts, then the fallback). SHREW_ABORT_REROLL=1 restores v0.3.x.
    rerolls = int(os.environ.get("SHREW_ABORT_REROLL", "0")) if aborted else 0
    attempt = 1
    while aborted and rerolls > 0:
        rerolls -= 1
        attempt += 1
        # Same config as the first pass; under a sampled first pass with the
        # fixed seed policy the attempt index makes it a different sample.
        rt, rx = _sampling(params["temperature"], params["extra_params"],
                           decode.try1_temp, decode, input_key, attempt - 1)
        _t0 = time.monotonic()
        text, finish_reason, guard = _call(
            messages, client,
            max_tokens=max_tokens,
            temperature=rt,
            extra_params=rx,
            timeout=timeout,
            doc_id=doc_id, contract=decode.contract, token_cap=token_cap,
            page_no=page_no, attempt=f"attempt {attempt}, re-roll (same config)",
        )
        parsed, verdict, error = _gate(text, finish_reason, guard, decode.contract)
        _rung(trail, "reroll", _t0, text, finish_reason, verdict, guard, max_tokens, rt, _dump)
        _note_completion(verdict == "empty")
        aborted = finish_reason == REPETITION_ABORT
        if verdict == "ok":
            return _result(True, parsed, "ok", None, attempt, len(text),
                           loop_guard=guard.stats() if guard else None,
                           rescued_by="reroll")

    # ── Retry tier: ONE attempt, enforcement on, flagged ────────────────────
    # Enforcement is for FORM failures (parse/schema) a changed request can fix,
    # and is the last rung after any re-rolls above.
    first_verdict = verdict
    first_error = error

    t2, x2 = _sampling(params["temperature"], decode.retry_extra(params["extra_params"]),
                       decode.try2_temp, decode, input_key, attempt)
    _t0 = time.monotonic()
    rtext, rfinish, rguard = _call(
        messages, client,
        max_tokens=max_tokens,
        temperature=t2,
        extra_params=x2,
        timeout=timeout,
        doc_id=doc_id, contract=decode.contract, token_cap=token_cap,
        page_no=page_no, attempt=("attempt 2/2, enforced retry" if decode.try2_enforce
                                  else "attempt 2/2, un-enforced retry"),
    )
    rparsed, rverdict, rerror = _gate(rtext, rfinish, rguard, decode.contract)
    _rung(trail, "retry_enforced" if decode.try2_enforce else "retry_unenforced",
          _t0, rtext, rfinish, rverdict, rguard, max_tokens, t2, _dump)
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
        near_blank = None
        if (min_content_chars and _cc < min_content_chars
                and _cc < COERCED_MIN_RATIO * len(text)):
            # v0.5: a page too bare to hold min_content_chars of text honestly
            # rescues to almost nothing (BLANK_INK_MAX). Measured only here.
            ink = _blank_page_ink(ink_source) if decode.contract == V05 else None
            if ink is None or ink >= BLANK_INK_MAX:
                ink_note = ("" if ink is None else
                            f"; page ink {ink:.4f} >= {BLANK_INK_MAX:g} near-blank bound")
                return _result(False, None, "coerced_empty",
                               f"coerced output carries {_cc} content chars "
                               f"(< {min_content_chars} floor and < {COERCED_MIN_RATIO:.0%} "
                               f"of the {len(text)}-char first-pass emission{ink_note}) "
                               f"— hallucination-slop signature, counted failed",
                               2, len(rtext), schema_coerced=True,
                               degenerate=(first_verdict == "degenerate"),
                               repetition_abort=aborted, loop_guard=stats,
                               model_calls=attempt + 1)
            near_blank = {"ink": round(ink, 6), "max": BLANK_INK_MAX,
                          "content_chars": _cc}
        # Well-formed but outside the validated distribution — flag it so
        # downstream can filter or de-rank.
        res = _result(True, rparsed, "ok_coerced", None, 2, len(rtext),
                      schema_coerced=True, degenerate=(first_verdict == "degenerate"),
                      repetition_abort=aborted, loop_guard=stats,
                      rescued_by=("retry_enforced" if decode.try2_enforce
                                  else "retry_unenforced"),
                      model_calls=attempt + 1)
        if near_blank:
            res["near_blank"] = near_blank
        return res

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
                   repetition_abort=aborted, loop_guard=stats, model_calls=attempt + 1)


# --------------------------------------------------------------------------- public API


def extract_page(image_path: str | Path, client, *, max_tokens: int | None = None,
                  timeout=None, page_no: int | None = None,
                  decode: "DecodePolicy | None" = None, doc_id: str | None = None,
                  bucket: str | None = None, ink_source=None) -> dict:
    """Run structured extraction on one page image (§2 image modality).

    ``image_path`` must already be ``preprocess.prepare_image_bucketed`` output
    — fitted to its glyph-routed tile bucket, then enhanced. Feeding anything
    else (a raw render, the legacy fixed DPI downscale) changes the model's
    output distribution and voids the eval numbers.

    ``ink_source`` (v0.5): the NATIVE page render (path or PIL image) the
    near-blank exemption measures when the coercion-density gate would fail a
    short retry (see BLANK_INK_MAX). None keeps the gate unconditional.

    Returns {"ok", "data", "status", "error", "attempts", "raw_len",
             "schema_coerced", "degenerate", "repetition_abort", "loop_guard"}
    (+ "near_blank" when the exemption accepted the page).
    """
    # The coercion-density floor applies to image pages only: a rendered page
    # that needed the enforcement retry AND produced almost no content is the
    # hallucination-slop signature. Text pages are exempt — a tiny text segment
    # legitimately coerces to a tiny result.
    decode = decode or DecodePolicy.from_env()
    token_cap = None
    if decode.contract == V05:
        # The size is read from the very file that is sent, so the image-token
        # count is exact on every image path (pages, cross-page composites).
        # It is unknown only when that file cannot be opened, and then the call
        # is charged the largest image there is (contract.v05_plan).
        size = None
        try:
            from PIL import Image as _Image
            with _Image.open(image_path) as _im:          # header only, no decode
                size = _im.size
        except Exception as e:
            logger.warning("v0.5 budget: cannot read the size of %s (%s); charging the "
                           "maximum image-token count", image_path, type(e).__name__)
        plan = _ct.v05_plan(_ct.CMD_EXTRACT, size=size,
                            cap=min(filter(None, (max_tokens, V05_MAX_TOKENS_CAP)), default=None))
        max_tokens = plan["max_tokens"]
        token_cap = _ct.v05_output_token_cut("image", bucket, plan["image_tokens"])
    elif max_tokens is None:
        max_tokens = MAX_TOKENS
    return _extract(build_image_messages(image_path, decode.contract), client,
                    max_tokens=max_tokens, timeout=timeout,
                    min_content_chars=COERCED_MIN_CHARS, page_no=page_no, decode=decode,
                    doc_id=doc_id, token_cap=token_cap, ink_source=ink_source)


def extract_text_page(text: str, client, *, max_tokens: int | None = None,
                       max_model_len: int = MAX_MODEL_LEN,
                       max_chars: int = TEXT_PAGE_MAX_CHARS, timeout=None,
                       page_no: int | None = None,
                       decode: "DecodePolicy | None" = None,
                       doc_id: str | None = None) -> dict:
    """Run structured extraction on one text page (§2 text modality).

    §5.4 — never truncate: a page over the size bound is filtered and reported,
    not shortened. Callers should run text through ``paginate_text`` first, so
    this only fires on a single unsplittable line (one enormous CSV record, a
    minified HTML blob).
    """
    decode = decode or DecodePolicy.from_env()
    if decode.contract == V05:
        # The v0.5 budget is derived per call from the text itself, so the only
        # size bound is the distribution one (TEXT_PAGE_MAX_CHARS).
        limit = max_chars
    else:
        if max_tokens is None:
            max_tokens = MAX_TOKENS
        limit = min(max_chars, context_limit_chars(max_model_len, max_tokens))
    if len(text) > limit:
        est = int(len(text) / CHARS_PER_TOKEN)
        return _result(False, None, "oversize",
                       f"{len(text)} chars (~{est} tok) exceeds the {limit}-char "
                       f"text-page bound; filtered rather than truncated",
                       0, len(text))
    token_cap = None
    if decode.contract == V05:
        from . import v05_tokens
        n_tok = v05_tokens.text_tokens(text)
        if n_tok > _ct.V05_TEXT_PAGE_MAX_TOKENS:
            return _result(False, None, "oversize",
                           f"{n_tok} tokens exceeds the {_ct.V05_TEXT_PAGE_MAX_TOKENS}-token "
                           f"text-page bound; filtered rather than truncated", 0, len(text))
        try:
            plan = _ct.v05_plan(_ct.CMD_EXTRACT, text=text,
                                cap=min(filter(None, (max_tokens, V05_MAX_TOKENS_CAP)), default=None))
        except _ct.V05BudgetError as e:
            return _result(False, None, "oversize",
                           f"{e}; filtered rather than truncated", 0, len(text))
        max_tokens = plan["max_tokens"]
        token_cap = _ct.v05_output_token_cut("text")
        tag = v05_tokens.template_tag(text)
        if tag:
            logger.warning("v0.5 text page %s (doc %s) contains the chat-template task tag <%s>: the served "
                           "template replaces the whole page text by that task's prompt", page_no, doc_id, tag)
    return _extract(build_text_messages(text, decode.contract), client,
                    max_tokens=max_tokens, timeout=timeout, page_no=page_no, decode=decode,
                    doc_id=doc_id, token_cap=token_cap)


# --------------------------------------------------------------------------- text pagination


# Split preference order: blank line (paragraph), then single newline. We never
# split inside a line — a line is the smallest unit the extractors emit, and a
# markdown table row or a CSV record cut in half is worse than a filtered page.
_PARA_SPLIT = re.compile(r"\n\s*\n")


# An embedded data: URI (docling markdown: ``![Image](data:image/png;base64,...)``)
# is a binary payload, not text. In the v0.5 training set 210 text prompts carry
# one and their labels transcribe the text around it and leave the payload out.
# Split into token-sized pages it would be a dozen pages of base64 sent to the
# model, so under v0.5 the payload is replaced by a marker before pagination and
# counted (``text_pagination.data_uri_payloads_removed``). SHREW_V05_STRIP_DATA_URIS=0
# sends it as it is.
V05_STRIP_DATA_URIS = os.environ.get("SHREW_V05_STRIP_DATA_URIS", "1") not in ("0", "false", "no", "off")
# The payload is one unbroken base64 run: no whitespace in the class, or the
# match would run on through the prose that follows.
_DATA_URI = re.compile(r"(data:[\w.+-]+/[\w.+-]+(?:;[\w=.+-]+)*;base64,)[A-Za-z0-9+/]{200,}={0,2}")
DATA_URI_MARKER = "[payload removed]"

# Where an over-long line may be cut, best first: after a sentence end, then at
# whitespace. CJK sentence marks end a sentence anywhere; ". ! ? ;" only before
# whitespace, so "3.14" and "e.g.x" stay whole.
_SENTENCE_END = re.compile(r"(?<=[。！？；…])|(?<=[.!?;:])(?=\s)")
_WORD_END = re.compile(r"(?<=\s)(?=\S)")


def strip_data_uris(text: str) -> tuple[str, int]:
    """(text with base64 data-URI payloads replaced by a marker, how many)."""
    return _DATA_URI.subn(lambda m: m.group(1) + DATA_URI_MARKER, text)


def _split_long_line(line: str, max_chars: int, max_tokens: int) -> list[str]:
    """Split one line that is over the page bound into pieces that fit: at
    sentence ends, then at whitespace, then on token boundaries. The pieces
    join back to ``line`` exactly — nothing is dropped or rewritten."""
    from . import v05_tokens

    def fits(piece: str) -> bool:
        return len(piece) <= max_chars and v05_tokens.text_tokens(piece) <= max_tokens

    def pack(parts: list[str]) -> list[str]:
        out, cur, cur_tok = [], "", 0
        for part in parts:
            tok = v05_tokens.text_tokens(part) + 1            # +1: a join can add a token
            if cur and (len(cur) + len(part) > max_chars or cur_tok + tok > max_tokens):
                out.append(cur)
                cur, cur_tok = "", 0
            cur, cur_tok = cur + part, cur_tok + tok
        return out + ([cur] if cur else [])

    def split(piece: str, level: int) -> list[str]:
        if fits(piece):
            return [piece]
        if level < 2:
            rx = _SENTENCE_END if level == 0 else _WORD_END
            cuts = [0] + [m.start() for m in rx.finditer(piece) if 0 < m.start() < len(piece)] + [len(piece)]
            parts = [piece[a:b] for a, b in zip(cuts, cuts[1:]) if a < b]
            if len(parts) > 1:
                return pack([q for part in parts for q in split(part, level + 1)])
            return split(piece, level + 1)
        return by_tokens(piece)

    def by_tokens(piece: str) -> list[str]:
        out = []
        for start in range(0, len(piece), max_chars):
            out.extend(v05_tokens.split_by_tokens(piece[start:start + max_chars], max_tokens))
        return out

    # The packed counts are sums of parts; the exact count of every piece is
    # checked here, and a piece that still lands over is cut on token boundaries.
    return [q for piece in split(line, 0) for q in ([piece] if fits(piece) else by_tokens(piece))]


def paginate_text(text: str, max_chars: int = TEXT_PAGE_MAX_CHARS,
                  max_tokens: int | None = None, stats: dict | None = None) -> list[str]:
    """Split extracted text into page-sized blocks for the text modality.

    "One request = one page" (§2): the text arm only ever saw page-sized input
    (see TEXT_PAGE_MAX_CHARS for the measured envelope), so a spreadsheet dump
    or a long markdown file is paginated rather than sent whole. Paragraphs are
    packed greedily; a paragraph over the bound is broken on line boundaries.

    A single line over the bound: under v0.4 (no ``max_tokens``) it is emitted as
    its own page, which ``extract_text_page`` filters and reports (§5.4). Under
    v0.5 it is SPLIT, never refused (text extracted from a PDF often arrives as
    one long line): at sentence ends, then whitespace, then on
    token boundaries (``_split_long_line``).

    ``max_tokens`` (v0.5: contract.V05_TEXT_PAGE_MAX_TOKENS) adds a bound in
    tokens of the served tokenizer; a page satisfies both. Units are counted
    once and summed with an allowance per joiner. ``stats`` (a dict) receives
    ``split_oversize_line`` and ``data_uri_payloads_removed``.
    """
    if not text.strip():
        return []
    from . import v05_tokens
    if max_tokens and V05_STRIP_DATA_URIS:
        text, removed = strip_data_uris(text)
        if stats is not None and removed:
            stats["data_uri_payloads_removed"] = stats.get("data_uri_payloads_removed", 0) + removed
    cost = (lambda u: v05_tokens.text_tokens(u)) if max_tokens else (lambda u: 0)
    limit = max_tokens or 0

    def _pack(units: list[str], joiner: str) -> list[str]:
        pages: list[str] = []
        cur, cur_tok = "", 0
        for u in units:
            # A unit over the character bound is split below whatever it costs.
            tok = cost(u) if len(u) <= max_chars else limit + 1
            candidate = u if not cur else cur + joiner + u
            cand_tok = tok if not cur else cur_tok + len(joiner) + tok
            if len(candidate) <= max_chars and (not max_tokens or cand_tok <= limit):
                cur, cur_tok = candidate, cand_tok
                continue
            if cur:
                pages.append(cur)
            if len(u) <= max_chars and (not max_tokens or tok <= limit):
                cur, cur_tok = u, tok
            else:
                # Too big even alone — break it down a level.
                if joiner == "\n\n":
                    pages.extend(_pack(u.split("\n"), "\n"))
                elif max_tokens:
                    pages.extend(_split_long_line(u, max_chars, limit))
                    if stats is not None:
                        stats["split_oversize_line"] = stats.get("split_oversize_line", 0) + 1
                else:
                    pages.append(u)  # single oversize line: filtered downstream
                cur, cur_tok = "", 0
        if cur:
            pages.append(cur)
        return pages

    return [p for p in _pack(_PARA_SPLIT.split(text.strip()), "\n\n") if p.strip()]
