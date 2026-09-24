"""Bounded document-level summary for the structured pipeline.

The structured path extracts one page at a time, and each page's 5-key output
carries a PAGE summary. Until this module the document ``summary`` was
``assembly.join_doc_summary()`` — every page summary joined with newlines —
so it grew linearly with page count: 1.1 MB for a 3,501-page handbook, which
blew a downstream reranker's 10,240-token window and made a whole retrieval
collection unsearchable (platform ops report, 2026-09-23). The pre-cutover
server never had that problem because its legacy pipeline made ONE
document-level call over the first ~8,000-token section of the cleaned
markdown (``structured.generate_summary``), which produced 600-750-char
abstracts even for 594-page PDFs.

Two ways to build the document summary for a multi-page document, chosen by
SHREW_DOC_SUMMARY_MODE:

* ``deterministic`` (DEFAULT, Tyler 2026-09-23: no extra model calls) —
  composed from what the page calls already produced. Selection rule, exact:
    1. the first extracted page's summary (the title page / opening page);
    2. then the summaries of pages carrying a chunk whose section_type is
       ``abstract``, then ``introduction``, then ``conclusion`` — each group
       in page order, a page taken once, identical summaries taken once;
       — for documents longer than SHREW_DOC_SUMMARY_LONG_DOC_PAGES (30) pages,
       ``abstract`` pages only: manuals label body pages ``introduction``;
    3. split into sentences; a sentence is dropped as a near-duplicate unless
       >= 50% of its content tokens are new relative to the sentences already
       kept (DEDUPE_MIN_NOVELTY; the first sentence is always kept);
    4. joined with a single space and cut at the budget on a SENTENCE
       boundary (the first sentence that does not fit ends the selection).
  Every other page summary is left out: the result is page 1's abstract plus
  the document's own overview pages, never a walk over the whole document.
* ``model`` — the pre-cutover legacy step: ONE text call with the same prompt
  (``pipelines/summarize_document.json``), the same section rule
  (``_section_document(markdown, max_tokens=8000)[0]``) and the same
  generation params, plus the hard output budget.
* ``summary_of_summaries`` (Tyler 2026-09-23) — ONE call to the same served
  model over the page summaries in page order (the whole document's coverage,
  cut at ~8,000 tokens), the legacy prompt plus a sentence saying the input
  is per-page summaries of one document.
* ``contract_sos`` — HIERARCHICAL, through the model's NORMAL text-page
  contract (sentinel system prompt, raw text user turn, structured_page decode
  params), taking each call's ``summary`` key and stopping the stream as soon
  as that key closes. Page summaries are packed in page order into blocks of
  <= 11,000 chars (the text arm's trained bound, ~35 pages) -> one call each;
  the block summaries are packed and summarized again until one root call
  remains. ~90 short calls for a 3,000-page book; one call when everything
  fits one block. Stays inside what the served OCR model was trained to do.
  If the tree returns null it FALLS BACK to ``deterministic`` (logged;
  processing_log ``mode_used`` = "deterministic_fallback").
* ``off`` — summary is None for multi-page documents.

MEASURED 2026-09-23 (E6 on an MI100): with the served Granite OCR model
(shrew-ocr-preview) BOTH ``model`` and ``summary_of_summaries`` return null —
the model answers a free-form summary prompt in its trained 5-key JSON, which
this module rejects. They only become meaningful behind a general-purpose
summary model reached through a separate URL knob, which is NOT built and
would be a new dependency. Use ``deterministic`` (or, once measured,
``contract_sos``) with this model.

Model modes: max_tokens 4096 (the legacy value); a completion that hits it is
a runaway and fails; otherwise the answer is cut to the budget on a sentence
boundary. Input windows are measured with a CJK-safe token estimate (CJK
chars count one token each) so a Chinese document cannot overflow 32k.

Rules common to all modes:

* single-page documents are untouched — the page summary already IS the
  document summary, and the output stays byte-identical to before;
* on ANY failure of the document-level step (transport error, empty or
  truncated completion, degenerate output, over budget) the summary is
  ``None`` and the failure is logged at ERROR — never a concatenation, never a
  silent truncation;
* per-page summaries stay exactly where they are (``pages[].page_summary``).

Knobs:
    SHREW_DOC_SUMMARY_MAX_CHARS  hard budget in CHARACTERS (default 4000).
        Characters, not tokens, because every consumer measures the field in
        characters (loader2 caps at ~4k chars; DB rows; rerank inputs are
        truncated by length before tokenization) and because the token count
        of the same string differs across the model, the reranker and the
        embedder. 4000 chars is ~1,000-1,300 tokens of English (~2.5-4k of
        CJK) — well inside a 10,240-token rerank pair with room for the query.
    SHREW_DOC_SUMMARY_MODE       "deterministic" (default) | "contract_sos"
                                 | "summary_of_summaries" | "model" | "off".
    SHREW_DOC_SUMMARY_BLOCK_CHARS  contract_sos block size (default 11,000).
    Per request, /v1/convert and /v1/convert/stream accept ``summary_mode`` and
    ``summary_max_chars`` (same enum, 200-20,000 chars; invalid -> HTTP 400).
    An invalid env value is logged and the default is used.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .generation import get_generation_params
from .structured import _load_pipeline, _section_document
from .structured_page import (MAX_TOKENS, STREAM_WALL_CLOCK_S, TEXT_PAGE_MAX_CHARS, RepetitionGuard,
                              context_limit_chars,
                              build_text_messages, is_degenerate, parse_json_lenient)

logger = logging.getLogger(__name__)

MODES = ("deterministic", "contract_sos", "summary_of_summaries", "model", "off")
# Bounds for SHREW_DOC_SUMMARY_MAX_CHARS and the per-request summary_max_chars.
# The floor keeps at least a real sentence or two; the ceiling keeps the field
# a bounded abstract (20,000 chars is ~5k English tokens, still inside a
# 10,240-token rerank pair) — larger than that is the bug this module fixes.
MIN_MAX_CHARS, MAX_MAX_CHARS = 200, 20000


def validate_mode(value) -> str:
    v = str(value).strip().lower()
    if v not in MODES:
        raise ValueError(f"summary mode must be one of {', '.join(MODES)}; got {value!r}")
    return v


def validate_max_chars(value) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"summary max chars must be an integer; got {value!r}") from None
    if not MIN_MAX_CHARS <= v <= MAX_MAX_CHARS:
        raise ValueError(f"summary max chars must be between {MIN_MAX_CHARS} and "
                         f"{MAX_MAX_CHARS}; got {v}")
    return v


def _env(name: str, default, validate):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return validate(raw)
    except ValueError as e:
        logger.error("%s ignored (%s) — using %r", name, e, default)
        return default


DOC_SUMMARY_MAX_CHARS = _env("SHREW_DOC_SUMMARY_MAX_CHARS", 4000, validate_max_chars)
DOC_SUMMARY_MODE = _env("SHREW_DOC_SUMMARY_MODE", "deterministic", validate_mode)
# contract_sos: block calls in flight per level (the VLM gate still bounds totals).
DOC_SUMMARY_CONCURRENCY = int(os.environ.get("SHREW_DOC_SUMMARY_CONCURRENCY", "4"))


def _block_chars() -> int:
    """contract_sos block size (SHREW_DOC_SUMMARY_BLOCK_CHARS, default the text
    arm's trained 11,000-char bound). Clamped to the hard context capacity; a
    value above the trained bound is allowed but logged, since it leaves the
    distribution the model was trained on."""
    raw = os.environ.get("SHREW_DOC_SUMMARY_BLOCK_CHARS")
    if raw is None:
        return TEXT_PAGE_MAX_CHARS
    try:
        v = int(raw)
        if v < 1000:
            raise ValueError("below 1000")
    except ValueError as e:
        logger.error("SHREW_DOC_SUMMARY_BLOCK_CHARS=%r ignored (%s) — using %d",
                     raw, e, TEXT_PAGE_MAX_CHARS)
        return TEXT_PAGE_MAX_CHARS
    cap = context_limit_chars()
    if v > cap:
        logger.error("SHREW_DOC_SUMMARY_BLOCK_CHARS=%d exceeds the context capacity; "
                     "clamped to %d", v, cap)
        return cap
    if v > TEXT_PAGE_MAX_CHARS:
        logger.warning("SHREW_DOC_SUMMARY_BLOCK_CHARS=%d is above the text arm's trained "
                       "%d-char bound", v, TEXT_PAGE_MAX_CHARS)
    return v
# Overview section types, in selection priority after the first page.
OVERVIEW_SECTION_TYPES = ("abstract", "introduction", "conclusion")
# Long documents use `abstract` pages only (plus page 1). Measured 2026-09-23/24:
# on an 18-page paper the introduction/conclusion pages were the paper's own
# overview; on pages 1-50 of a reference manual the model labels many BODY
# pages `introduction` (logarithms, complex numbers, quadratic equations...),
# and the summary degraded into a page list. The two measurements bracket the
# threshold to (18, 50] pages — they do not pin it — so the default is their
# geometric mean, ~30. SHREW_DOC_SUMMARY_LONG_DOC_PAGES overrides it.
LONG_DOC_PAGES = int(os.environ.get("SHREW_DOC_SUMMARY_LONG_DOC_PAGES", "30"))
LONG_DOC_SECTION_TYPES = ("abstract",)
# The legacy step's input window: the first ~8,000 approx-tokens of markdown.
DOC_SUMMARY_INPUT_TOKENS = 8000


# Legacy call's output cap (structured.generate_summary used 4096). The prompt
# asks for <=200 tokens; a completion that hits this cap is a runaway and fails.
DOC_SUMMARY_MAX_TOKENS = 4096

_SOS_PREAMBLE = (
    "\n\nThe input below is the sequence of per-page summaries of ONE document, "
    "in page order, one per line. Summarize the document as a whole — its "
    "purpose, scope and main content — not the individual pages."
)
_CJK = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")


def _approx_tokens_safe(text: str) -> int:
    """Token estimate that does not undercount CJK.

    The codebase's ``_approx_tokens`` is chars/4, right for English and ~4x
    low for Chinese/Japanese/Korean, where a character is ~1 token. An input
    window sized with chars/4 would overflow the 32k context on a CJK
    document. Count CJK characters as one token each, the rest as chars/4.
    """
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk) // 4


def _cap_tokens(text: str, max_tokens: int) -> str:
    """Cut ``text`` to ``max_tokens`` (safe estimate) at a line boundary."""
    if _approx_tokens_safe(text) <= max_tokens:
        return text
    out, used = [], 0
    for line in text.split("\n"):
        t = _approx_tokens_safe(line) + 1
        if used + t > max_tokens:
            break
        out.append(line)
        used += t
    return "\n".join(out)


def _cut_to_budget(text: str, budget: int) -> str | None:
    """Leading whole sentences within ``budget`` chars; None if none fits."""
    if len(text) <= budget:
        return text
    out, used = [], 0
    for sent in _sentences(text):
        add = (1 if out else 0) + len(sent)
        if used + add > budget:
            break
        out.append(sent)
        used += add
    return " ".join(out) or None


def summarize_document(source_text: str, client, *, source: str = "markdown",
                       max_chars: int | None = None,
                       doc_id: str | None = None) -> tuple[str | None, dict]:
    """ONE document-level summary call.

    ``source="markdown"`` (mode ``model``): the legacy step — the first
    ~8,000-token section of the cleaned markdown, the legacy prompt.
    ``source="page_summaries"`` (mode ``summary_of_summaries``): the joined
    page summaries in page order, cut at ~8,000 tokens, the legacy prompt plus
    one sentence saying the input is per-page summaries of one document.

    Returns ``(summary_or_None, report)``.
    """
    budget = DOC_SUMMARY_MAX_CHARS if max_chars is None else max_chars
    report = {"status": None, "chars": 0, "max_chars": budget, "input_chars": 0,
              "input_tokens_est": 0, "calls": 1, "seconds": 0.0, "error": None}

    def fail(reason: str):
        report["status"] = "failed"
        report["error"] = reason
        logger.error("doc summary FAILED for %s: %s — returning summary: null "
                     "(the per-page summaries are NOT concatenated in its place)",
                     doc_id or "document", reason)
        return None, report

    if source == "markdown":
        section = _section_document(source_text, max_tokens=DOC_SUMMARY_INPUT_TOKENS)[0]["content"]
    else:
        section = source_text
    section = _cap_tokens(section, DOC_SUMMARY_INPUT_TOKENS)
    report["input_chars"] = len(section)
    report["input_tokens_est"] = _approx_tokens_safe(section)
    if not section.strip():
        report["calls"] = 0
        return fail("no document text to summarize")

    pipeline = _load_pipeline("summarize_document")
    system = pipeline["generation_system_prompt"]
    if source == "page_summaries":
        system += _SOS_PREAMBLE
    user_msg = pipeline["generation_user_template"].replace("{section_content}", section)
    params = get_generation_params(client.model, "structured")
    t0 = time.time()
    try:
        result = client.chat_completion(
            [{"role": "system", "content": system},
             {"role": "user", "content": user_msg}],
            max_tokens=DOC_SUMMARY_MAX_TOKENS,
            temperature=params["temperature"],
            extra_params=params["extra_params"],
        )
        choice = result["choices"][0]
        text = (choice["message"].get("content") or "").strip().strip('"').strip()
        finish = choice.get("finish_reason")
    except Exception as e:  # transport, HTTP, malformed response
        report["seconds"] = round(time.time() - t0, 2)
        return fail(f"{type(e).__name__}: {e}")
    report["seconds"] = round(time.time() - t0, 2)

    if finish == "length":
        return fail(f"completion hit the {DOC_SUMMARY_MAX_TOKENS}-token cap (runaway)")
    if not text:
        return fail("empty completion")
    if is_degenerate(text):
        return fail("degenerate (repetitive) completion")
    if text.lstrip("`").lstrip().startswith(("{", "[", "json")):
        # The served OCR model's trained contract is 5-key JSON; if it answers
        # the summary prompt in that shape it has not produced an abstract.
        return fail("model returned structured JSON, not a plain-text summary")
    cut = _cut_to_budget(text, budget)
    if cut is None:
        return fail(f"no sentence of the {len(text)}-char completion fits the "
                    f"{budget}-char budget (SHREW_DOC_SUMMARY_MAX_CHARS)")
    if len(cut) < len(text):
        logger.warning("doc summary for %s cut from %d to %d chars at a sentence "
                       "boundary (budget %d)", doc_id or "document", len(text), len(cut), budget)
        report["cut_from_chars"] = len(text)

    report["status"] = "ok"
    report["chars"] = len(cut)
    logger.info("doc summary (%s) for %s: %d chars from a %d-char input in %.1fs",
                source, doc_id or "document", len(cut), len(section), report["seconds"])
    return cut, report


# Near-duplicate filter for the deterministic mode. Page summaries of the
# pages it selects (title page, abstract, introduction, conclusion) often
# restate the whole document in different words — measured on an 18-page
# paper: three "This research paper evaluates…" openers. Pairwise sentence
# similarity cannot separate those paraphrases (content-word overlap <= 0.56)
# from genuinely different sentences of one page (0.17-0.22), so the rule is
# CUMULATIVE NOVELTY instead: a sentence is kept only if at least
# DEDUPE_MIN_NOVELTY of its content tokens are not already in the kept
# sentences. Content tokens: lowercase Latin/digit words of 3+ chars minus
# stopwords, plus character bigrams of CJK runs (CJK has no spaces). The
# document's first sentence is always kept. On the paper this keeps 890 of
# 1,296 chars with one sentence each from the introduction and conclusion.
DEDUPE_MIN_NOVELTY = 0.5
_STOPWORDS = frozenset(
    "a an the of in on and or to for with by at from as is are was were be been "
    "this that these those it its their which while even into across over under "
    "than more most such not no do does also has have had".split())
_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u9fff\uac00-\ud7af\u3040-\u30ff]+")


def _content_tokens(sentence: str) -> set:
    low = sentence.lower()
    toks = {w for w in _WORD.findall(low) if len(w) > 2 and w not in _STOPWORDS}
    for run in _CJK_RUN.findall(sentence):
        toks |= {run[i:i + 2] for i in range(max(1, len(run) - 1))}
    return toks


def _novelty(tokens: set, kept: set) -> float:
    if not tokens:
        return 0.0          # nothing checkable (e.g. a bare number): not new
    return len(tokens - kept) / len(tokens)


_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s*")


def _sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_END.split(text)) if s]


def compose_document_summary(doc: dict, *, max_chars: int | None = None) -> tuple[str | None, dict]:
    """Deterministic document summary from the page summaries (no model call).

    Returns ``(summary_or_None, report)``. See the module docstring for the
    exact selection rule.
    """
    budget = DOC_SUMMARY_MAX_CHARS if max_chars is None else max_chars
    report = {"status": None, "mode": "deterministic", "chars": 0,
              "max_chars": budget, "pages_used": [], "error": None}
    summaries = {p["page"]: (p.get("page_summary") or "").strip()
                 for p in doc.get("pages") or [] if p.get("ok")}
    ordered = [pg for pg in sorted(summaries) if summaries[pg]]
    if not ordered:
        report["status"] = "failed"
        report["error"] = "no page produced a summary"
        logger.error("doc summary FAILED for %s: %s — returning summary: null",
                     doc.get("doc_id") or "document", report["error"])
        return None, report

    types_by_page: dict[int, set] = {}
    for c in doc.get("chunks") or []:
        for pg in c.get("pages") or [c.get("page")]:
            if pg is not None:
                types_by_page.setdefault(pg, set()).add(c.get("section_type"))
    long_doc = len(doc.get("pages") or []) > LONG_DOC_PAGES
    overview = LONG_DOC_SECTION_TYPES if long_doc else OVERVIEW_SECTION_TYPES
    report["overview_types"] = list(overview)
    candidates = [ordered[0]]
    for st in overview:
        candidates += [pg for pg in ordered if st in types_by_page.get(pg, ())]

    # Sentence stream over the selected pages, near-duplicates dropped, cut at
    # the budget on a sentence boundary (the first sentence that does not fit
    # ends the selection).
    parts: list[str] = []
    kept_tokens: set = set()
    seen_pages: set = set()
    seen_text: set = set()
    used = 0
    report["sentences_dropped_as_duplicate"] = 0
    full = False
    for pg in candidates:
        text = summaries[pg]
        if pg in seen_pages or text in seen_text:
            continue
        seen_pages.add(pg)
        seen_text.add(text)
        took = False
        for sent in _sentences(text):
            toks = _content_tokens(sent)
            if parts and _novelty(toks, kept_tokens) < DEDUPE_MIN_NOVELTY:
                report["sentences_dropped_as_duplicate"] += 1
                continue
            add = (1 if parts else 0) + len(sent)
            if used + add > budget:
                full = True
                break
            parts.append(sent)
            kept_tokens |= toks
            used += add
            took = True
        if took:
            report["pages_used"].append(pg)
        if full:
            break

    if not parts:
        report["status"] = "failed"
        report["error"] = (f"the first page summary has no sentence that fits the "
                           f"{budget}-char budget (SHREW_DOC_SUMMARY_MAX_CHARS)")
        logger.error("doc summary FAILED for %s: %s — returning summary: null",
                     doc.get("doc_id") or "document", report["error"])
        return None, report
    summary = " ".join(parts)
    report["status"] = "ok"
    report["chars"] = len(summary)
    return summary, report


_SUMMARY_VALUE = re.compile(r'"summary"\s*:\s*("(?:[^"\\]|\\.)*"|null)')
SUMMARY_CAPTURED = "summary_captured"


class _SummaryCapture:
    """``on_delta`` that stops the stream as soon as the top-level ``summary``
    value has been emitted. The trained contract emits ``metadata`` first and
    ``summary`` second; everything after it (chunks transcribing the input)
    is discarded, so waiting for it would only cost tokens."""

    def __init__(self, repetition: RepetitionGuard):
        self.repetition = repetition
        self.value = None
        self.found = False

    def __call__(self, chunk: str, accumulated: str):
        if not self.found:
            m = _SUMMARY_VALUE.search(accumulated)
            if m:
                self.found = True
                raw = m.group(1)
                self.value = None if raw == "null" else json.loads(raw)
                return SUMMARY_CAPTURED
        return self.repetition(chunk, accumulated)


def _contract_call(text: str, client) -> tuple[str | None, dict]:
    """ONE text-page-contract call over ``text``; returns (summary_or_None, info)."""
    info = {"input_chars": len(text), "output_chars": 0, "seconds": 0.0, "error": None}
    messages = build_text_messages(text)
    params = get_generation_params(client.model, "structured_page")
    t0 = time.time()
    try:
        if hasattr(client, "chat_completion_stream"):
            capture = _SummaryCapture(RepetitionGuard(attempt="doc summary"))
            result = client.chat_completion_stream(
                messages, max_tokens=MAX_TOKENS, temperature=params["temperature"],
                extra_params=params["extra_params"], on_delta=capture,
                wall_clock_s=STREAM_WALL_CLOCK_S)
            choice = result["choices"][0]
            out = choice["message"].get("content") or ""
            if capture.found:
                value = capture.value
            else:
                parsed, _ = parse_json_lenient(out)
                if not isinstance(parsed, dict):
                    info["error"] = (f"no summary key in the contract output "
                                     f"(finish={choice.get('finish_reason')})")
                    value = None
                else:
                    value = parsed.get("summary")
        else:
            result = client.chat_completion(
                messages, max_tokens=MAX_TOKENS, temperature=params["temperature"],
                extra_params=params["extra_params"])
            out = result["choices"][0]["message"].get("content") or ""
            parsed, _ = parse_json_lenient(out)
            if not isinstance(parsed, dict):
                info["error"] = "contract output did not parse"
                value = None
            else:
                value = parsed.get("summary")
    except Exception as e:  # transport, HTTP, malformed response
        info["seconds"] = round(time.time() - t0, 2)
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info
    info["seconds"] = round(time.time() - t0, 2)
    info["output_chars"] = len(out)
    if info["error"]:
        return None, info
    if not isinstance(value, str) or not value.strip():
        info["error"] = "the contract's summary key was empty or null"
        return None, info
    value = value.strip()
    if is_degenerate(value):
        info["error"] = "degenerate (repetitive) summary"
        return None, info
    return value, info


def _pack_blocks(items: list[str], cap: int) -> list[str]:
    """Pack ``items`` in order into newline-joined blocks of at most ``cap``
    chars. An item longer than ``cap`` on its own is cut at a line/char bound
    (page summaries are ~300 chars; this only guards a pathological one)."""
    blocks, cur, size = [], [], 0
    for it in items:
        it = it[:cap]
        add = len(it) + (1 if cur else 0)
        if cur and size + add > cap:
            blocks.append("\n".join(cur))
            cur, size = [], 0
            add = len(it)
        cur.append(it)
        size += add
    if cur:
        blocks.append("\n".join(cur))
    return blocks


# EXPERIMENT (off by default): one line prepended to the ROOT call's input when
# the root summarizes block summaries (level > 0), e.g. "Section summaries of one
# document, in order:". It is input text, so the call stays inside the trained
# contract. Measured on 50 pages the root echoed block 1 instead of integrating
# the blocks; this tests whether framing the input changes that.
ROOT_PREFIX = os.environ.get("SHREW_DOC_SUMMARY_ROOT_PREFIX", "")


def summarize_via_contract(page_summaries: str, client, *, max_chars: int | None = None,
                           doc_id: str | None = None,
                           root_prefix: str | None = None) -> tuple[str | None, dict]:
    """Mode ``contract_sos``: HIERARCHICAL summary through the model's NORMAL
    text-page contract (sentinel system prompt, raw text user turn,
    structured_page decode params); each call's ``summary`` key is taken and
    the stream stops as soon as that key closes.

    Level 0: page summaries, in page order, packed into blocks of at most
    TEXT_PAGE_MAX_CHARS (11,000 chars — the text arm's trained distribution
    bound; ~35 page summaries) -> one call per block. Each further level packs
    the previous level's summaries the same way, until one block remains; the
    call over that block is the ROOT, cut to the char budget on a sentence
    boundary. A document whose page summaries fit one block is a single call.
    A failed non-root call drops that block (logged at WARNING and counted); a
    level where every call fails, or a failed root call, returns None.
    """
    budget = DOC_SUMMARY_MAX_CHARS if max_chars is None else max_chars
    report = {"status": None, "mode": "contract_sos", "chars": 0, "max_chars": budget,
              "calls": 0, "seconds": 0.0, "levels": [], "error": None}
    t0 = time.time()

    def fail(reason: str):
        report["status"] = "failed"
        report["error"] = reason
        report["seconds"] = round(time.time() - t0, 2)
        logger.error("doc summary FAILED for %s: %s — returning summary: null "
                     "(the per-page summaries are NOT concatenated in its place)",
                     doc_id or "document", reason)
        return None, report

    items = [ln.strip() for ln in page_summaries.split("\n") if ln.strip()]
    if not items:
        return fail("no page summaries to summarize")
    tree = [] if os.environ.get("SHREW_DOC_SUMMARY_DEBUG") == "1" else None
    block_chars = _block_chars()
    report["block_chars"] = block_chars
    level = 0
    while True:
        blocks = _pack_blocks(items, block_chars)
        is_root = len(blocks) == 1
        prefix = ROOT_PREFIX if root_prefix is None else root_prefix
        if is_root and level > 0 and prefix:
            blocks = [prefix.strip() + "\n" + blocks[0]]
            report["root_prefix"] = prefix.strip()
        workers = max(1, min(DOC_SUMMARY_CONCURRENCY, len(blocks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda b: _contract_call(b, client), blocks))
        report["calls"] += len(blocks)
        ok = [v for v, _ in results if v is not None]
        errors = [info["error"] for v, info in results if v is None]
        lvl = {"level": level, "inputs": len(items), "calls": len(blocks), "ok": len(ok),
               "dropped": len(errors), "root": is_root,
               "seconds": round(sum(i["seconds"] for _, i in results), 2)}
        report["levels"].append(lvl)
        if tree is not None:
            tree.append([{"input_chars": i["input_chars"], "summary": v, "error": i["error"]}
                         for v, i in results])
        for e in errors:
            if not is_root:
                logger.warning("doc summary for %s: level-%d block dropped: %s",
                               doc_id or "document", level, e)
        if is_root:
            if not ok:
                if tree is not None:
                    report["tree"] = tree
                return fail(f"root call failed: {errors[0]}")
            value = ok[0]
            break
        if not ok:
            if tree is not None:
                report["tree"] = tree
            return fail(f"every level-{level} call failed (first: {errors[0]})")
        items = ok
        level += 1
        if level > 6:  # each level shrinks ~30x; 6 levels cover any real document
            return fail("summary tree did not converge")

    if tree is not None:
        report["tree"] = tree
    report["seconds"] = round(time.time() - t0, 2)
    cut = _cut_to_budget(value, budget)
    if cut is None:
        return fail(f"no sentence of the {len(value)}-char root summary fits the "
                    f"{budget}-char budget (SHREW_DOC_SUMMARY_MAX_CHARS)")
    report["status"] = "ok"
    report["chars"] = len(cut)
    logger.info("doc summary (contract_sos) for %s: %d chars; tree %s; %d calls in %.1fs",
                doc_id or "document", len(cut),
                "/".join(str(l["calls"]) for l in report["levels"]), report["calls"],
                report["seconds"])
    return cut, report


def _joined_page_summaries(doc: dict) -> str:
    return "\n".join(
        (p.get("page_summary") or "").strip()
        for p in sorted(doc.get("pages") or [], key=lambda p: p["page"])
        if p.get("ok") and (p.get("page_summary") or "").strip())


def apply_document_summary(doc: dict, markdown: str, client, *,
                           max_chars: int | None = None, mode: str | None = None) -> dict:
    """Replace the joined per-page summary on a multi-page ``doc`` in place.

    Single-page documents (exactly one extracted page) keep their page summary
    unchanged. Returns the processing-log entry.
    """
    ok_pages = [p for p in doc.get("pages") or [] if p.get("ok")]
    if len(ok_pages) <= 1:
        return {"status": "single_page", "chars": len(doc.get("doc_summary") or "")}
    # Per-request override (validated by the server) wins over the env default.
    mode = DOC_SUMMARY_MODE if mode is None else validate_mode(mode)
    if max_chars is not None:
        max_chars = validate_max_chars(max_chars)
    if mode == "off":
        doc["doc_summary"] = None
        return {"status": "disabled", "mode": "off", "chars": 0}
    if mode == "model":
        summary, report = summarize_document(markdown, client, source="markdown",
                                             max_chars=max_chars, doc_id=doc.get("doc_id"))
        report["mode"] = "model"
    elif mode == "contract_sos":
        summary, report = summarize_via_contract(_joined_page_summaries(doc), client,
                                                 max_chars=max_chars, doc_id=doc.get("doc_id"))
        report["mode_used"] = "contract_sos"
        if summary is None:
            # The tree failed (logged at ERROR above). Fall back to the zero-call
            # composer rather than returning null: a bounded deterministic
            # abstract beats none. If that fails too, the summary is null.
            logger.warning("doc summary for %s: contract_sos returned null (%s) — "
                           "falling back to deterministic", doc.get("doc_id") or "document",
                           report.get("error"))
            summary, det = compose_document_summary(doc, max_chars=max_chars)
            report = {**det, "mode": "contract_sos", "contract_sos": report,
                      "mode_used": "deterministic_fallback" if summary is not None else "none"}
    elif mode == "summary_of_summaries":
        summary, report = summarize_document(_joined_page_summaries(doc), client,
                                             source="page_summaries",
                                             max_chars=max_chars, doc_id=doc.get("doc_id"))
        report["mode"] = "summary_of_summaries"
    else:
        summary, report = compose_document_summary(doc, max_chars=max_chars)
    report.setdefault("mode_used", report.get("mode") or mode)
    if summary is None:
        report["mode_used"] = "none"
    doc["doc_summary"] = summary
    return report
