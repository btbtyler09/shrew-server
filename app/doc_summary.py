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
    3. joined with a single space and cut at the budget on a SENTENCE
       boundary (a partially fitting page summary contributes its leading
       whole sentences; a page summary that fits nowhere ends the selection).
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
* ``off`` — summary is None for multi-page documents.

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
    SHREW_DOC_SUMMARY_MODE       "deterministic" (default) | "summary_of_summaries"
                                 | "model" | "off".
"""
from __future__ import annotations

import logging
import os
import re
import time

from .generation import get_generation_params
from .structured import _load_pipeline, _section_document
from .structured_page import is_degenerate

logger = logging.getLogger(__name__)

DOC_SUMMARY_MAX_CHARS = int(os.environ.get("SHREW_DOC_SUMMARY_MAX_CHARS", "4000"))
DOC_SUMMARY_MODE = os.environ.get("SHREW_DOC_SUMMARY_MODE", "deterministic").strip().lower()
# Overview section types, in selection priority after the first page.
OVERVIEW_SECTION_TYPES = ("abstract", "introduction", "conclusion")
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
    candidates = [ordered[0]]
    for st in OVERVIEW_SECTION_TYPES:
        candidates += [pg for pg in ordered if st in types_by_page.get(pg, ())]

    parts: list[str] = []
    seen_pages: set = set()
    seen_text: set = set()
    used = 0
    for pg in candidates:
        text = summaries[pg]
        if pg in seen_pages or text in seen_text:
            continue
        seen_pages.add(pg)
        seen_text.add(text)
        sep = 1 if parts else 0
        if used + sep + len(text) <= budget:
            parts.append(text)
            used += sep + len(text)
            report["pages_used"].append(pg)
            continue
        # Partial fit: the leading whole sentences that fit, then stop.
        fitted = []
        for sent in _sentences(text):
            add = (1 if (parts or fitted) else 0) + len(sent)
            if used + add > budget:
                break
            fitted.append(sent)
            used += add
        if fitted:
            parts.append(" ".join(fitted))
            report["pages_used"].append(pg)
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


def apply_document_summary(doc: dict, markdown: str, client, *,
                           max_chars: int | None = None) -> dict:
    """Replace the joined per-page summary on a multi-page ``doc`` in place.

    Single-page documents (exactly one extracted page) keep their page summary
    unchanged. Returns the processing-log entry.
    """
    ok_pages = [p for p in doc.get("pages") or [] if p.get("ok")]
    if len(ok_pages) <= 1:
        return {"status": "single_page", "chars": len(doc.get("doc_summary") or "")}
    mode = DOC_SUMMARY_MODE
    if mode == "off":
        doc["doc_summary"] = None
        return {"status": "disabled", "mode": "off", "chars": 0}
    if mode == "model":
        summary, report = summarize_document(markdown, client, source="markdown",
                                             max_chars=max_chars, doc_id=doc.get("doc_id"))
        report["mode"] = "model"
    elif mode == "summary_of_summaries":
        page_summaries = "\n".join(
            (p.get("page_summary") or "").strip()
            for p in sorted(doc.get("pages") or [], key=lambda p: p["page"])
            if p.get("ok") and (p.get("page_summary") or "").strip())
        summary, report = summarize_document(page_summaries, client, source="page_summaries",
                                             max_chars=max_chars, doc_id=doc.get("doc_id"))
        report["mode"] = "summary_of_summaries"
    else:
        if mode != "deterministic":
            logger.error("unknown SHREW_DOC_SUMMARY_MODE=%r — using deterministic", mode)
        summary, report = compose_document_summary(doc, max_chars=max_chars)
    doc["doc_summary"] = summary
    return report
