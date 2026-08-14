"""Fallback extraction: a generic VLM rescues pages shrew-ocr fails.

When a page fails the primary path (repetition loop, wall-clock grind,
truncation, empty completion), an optional fallback model is asked for the same
5-key JSON using the teacher-lineage prompt below. Off unless FALLBACK_VLM_URL
is set; every rescued page is flagged ``fallback: true`` so downstream can
filter or de-rank — fallback output is outside the validated e4 distribution.

PROMPT LINEAGE (merged 2026-08-14; keep in sync with the training repo):
  * output shape + semantic_chunks rules — the 5-key dataset-generation prompt
    (shrew_ocr/rl/gen_text_labels_27b.py TEXT_SYSTEM)
  * image transcription + figure/table/artifact/handwriting rules — the image
    teacher spec (shrew_ocr/structured_eval/spec.py SYSTEM)
  * bbox convention VERBATIM from spec.py as synced to
    vhr-bench/spec/REGION_CONVENTION.md (commit fda2962: 0-4 unit padding,
    notes/SOURCE lines inside the box, caption excluded, no edge may slice a
    glyph). REGION_CONVENTION.md is authoritative; if this file disagrees with
    it, this file is wrong.

INPUT RESOLUTION: generic VLMs (Qwen-VL dynamic resolution, Gemma pan-and-scan)
accept arbitrary-resolution input and tile it themselves — the e4 bucket
pinpoints are Granite/LlavaNext architecture config and do NOT transfer. What
transfers is the principle: scale the native render by measured glyph height so
body text is resolvable, cap the long edge for API sanity, and let the model's
own preprocessor do the rest. No luminance/CLAHE enhancement — that is
student-specific training distribution, not a generic-VLM improvement.
"""

import logging
import os

from PIL import Image

from .structured_page import (
    ZLIB_GATE_RATIO,
    parse_json_lenient,
    validate_schema,
    zlib_ratio,
    SECTION_ENUM,
)

logger = logging.getLogger("shrew.fallback")

FALLBACK_URL = os.environ.get("FALLBACK_VLM_URL", "").strip()
FALLBACK_MODEL = os.environ.get("FALLBACK_VLM_MODEL", "").strip()
FALLBACK_API_KEY = os.environ.get("FALLBACK_API_KEY") or None
FALLBACK_MAX_TOKENS = int(os.environ.get("FALLBACK_MAX_TOKENS", "12000"))
FALLBACK_TIMEOUT_S = int(os.environ.get("FALLBACK_TIMEOUT", "600"))
# Target glyph height in the image the fallback model sees. 10 px matches the
# resolvability floor measured for the e4 buckets; generic towers are usually at
# least as capable, and sending more pixels than needed only burns tokens.
FALLBACK_GLYPH_TARGET = float(os.environ.get("FALLBACK_GLYPH_TARGET", "10"))
FALLBACK_MAX_LONG_EDGE = int(os.environ.get("FALLBACK_MAX_LONG_EDGE", "4096"))


def enabled() -> bool:
    return bool(FALLBACK_URL)


def make_fallback_client():
    """VLMClient for the fallback endpoint, or None when unconfigured."""
    if not enabled():
        return None
    from .vlm_client import VLMClient
    return VLMClient(FALLBACK_URL, FALLBACK_MODEL or "fallback",
                     api_key=FALLBACK_API_KEY, default_timeout=FALLBACK_TIMEOUT_S)


FALLBACK_SYSTEM = """You convert a single document page image into a structured JSON object with five keys: metadata, summary, semantic_chunks, figures, tables. Transcribe only what is visibly present on THIS page. Do not summarize away content, infer, or fabricate.

- metadata: {title, authors, organization, year, doc_type} — a field's value ONLY if it physically appears on this page; otherwise null. Never guess from context. authors is a list. year is a 4-digit string, only if a publication/revision date is printed. doc_type one of [Research Paper, Technical Manual, Technical Report, News Article, Proposal, Contract, Calculation Package, Planning Document, Acceptance Report, Regulation, Book], only if confidently inferable from this page; else null.
- summary: a concise 1-2 sentence summary of what this page contains. Unlike metadata, you SYNTHESIZE this.
- semantic_chunks: self-contained retrieval units in natural reading order — each {chunk_id ("c1","c2",...), title (short descriptive title you write), content (the transcribed text of the unit), keywords (3-6), section_type (one of: abstract, introduction, methodology, results, discussion, conclusion, technical_content, appendix)}. For two-column layouts, read the entire left column top-to-bottom, then the right column. Merge tiny fragments into coherent units; do NOT put table grids in chunk content.
- figures: one entry per genuine visual on the page; each {bbox: [x1,y1,x2,y2], caption: the printed caption line exactly, or "" if none}.
- tables: one entry per genuine data grid; each {bbox: [x1,y1,x2,y2], html: a complete HTML <table>, caption: the printed caption or null}.

Chunk content rules:
- Use LaTeX for ALL math: $...$ inline, $$...$$ for display equations (with \\tag{N} if numbered). Greek letters, subscripts, superscripts always in LaTeX even in prose ($J_n$, $a_0$, $10^{-6}$). Inline citations as <sup>1</sup>.
- Lists as markdown (-, 1.) preserving items and nesting. Also use a list (one item per row) for content that is visually columnar but read item-by-item — label–value pairs without a header row, key–value form fields, document indices / tables-of-contents, reference or exhibit lists, and numbered/lettered enumerations. Keep each row's identifier (e.g. "10.5", "(a)") as the start of its item. For a two-column "label → description" layout where the second column is prose, emit one list item per row as "label — description" (do NOT split a row into disconnected paragraphs, and do NOT make it a table).
- Code as fenced code with exact code + language tag; no line-number gutters.

Tables — use ONLY for a genuine DATA GRID: content read by a row×column relationship, with a clear header row (or header column) AND at least two genuine data columns. Emit exact cell values from the image, rowspan/colspan for merged cells and multi-row headers. Represent the table's TRUE structure — do not force a rectangular grid. Transcribe only cells visible in the image; never fabricate rows/columns/cells. Do NOT use a table for label–value pairs without a header, key–value form fields, indices/tables-of-contents, reference/exhibit lists, or any linear enumeration that merely happens to be column-aligned — those are lists in a chunk.

Figures — emit ONLY for a genuine visual: photograph, chart/plot/graph, diagram, drawing/illustration, map, or schematic. Do NOT transcribe text inside the figure (axis labels, legends, data values, annotations) into chunks.

Artifacts — do NOT emit as figures: company logos, branding/letterhead, signatures, stamps/seals, watermarks, decorative rules/dividers/borders, icons/bullets/dingbats, barcodes/QR codes. For signatures and stamps/seals, transcribe their legible text inline in chunk content (the signed name; stamp text such as "APPROVED 2024-01-03"); if illegible use [illegible signature] / [illegible stamp]. Skip purely decorative/non-textual artifacts entirely.

Handwriting — transcribe only printed/typeset text. Ignore handwritten margin notes, annotations, and handwritten form-field entries. (Signatures are the only exception, per above.)

The bbox of a figure or table is [x1, y1, x2, y2] with coordinates normalized to a 0-1000 grid over the page (x rightward, y downward, origin at the top-left corner). The bbox covers the object and every mark that makes it readable — axis ticks and their labels, axis titles, legends, colorbars, scale bars, panel labels — and it also covers the object's own NOTES / SOURCE / key / footnote lines even when those sit just outside its frame (a "NOTES: CDIC = ..." or "SOURCE: Cheng Li, Foreign Policy, 2012" line printed under a chart is part of the figure; a footnote row inside a table's own ruling is part of the table). It excludes only the caption line, which is carried in `caption` instead. Keep the box TIGHT: 0-4 grid units of padding is what is wanted, and a tight edge that loses nothing is correct. Padding above 4 units is never required, and a box is never wrong for being tight alone. The invariant is content, not margin: no edge may slice a glyph, a tick, or a rule.

General: skip page headers/footers (page numbers, running heads, repeated banners). If a sentence is cut off at the page boundary, include it as-is. If text is illegible, use [illegible] rather than guessing. Return ONLY the JSON object — no commentary, no fences.

Leader lines — runs of repeated dots/periods/middots/dashes/underscores that connect a label to a page number or value (common in tables-of-contents, indices, lists-of-figures, exhibit/reference lists, form fields) — are DECORATIVE fill. NEVER reproduce the leader characters, even when leaders separate several columns within one row. Emit each entry as its meaningful parts joined by ' — ', dropping every leader run: e.g. '- Introduction — 14'; for a multi-column row '- Brown v. Piper — 91 U. S. 37 — 2'; for a form field 'Serial No. —'. Use just the label if no value follows."""

FALLBACK_USER = "Extract the structured document from this page image per the rules. Return only the JSON object."


def prepare_image_fallback(img: Image.Image, glyph_px: float | None):
    """Scale the native render for a generic VLM: glyph-targeted, never
    upscaled (interpolation invents no detail), long edge capped."""
    w, h = img.size
    scale = 1.0
    if glyph_px and glyph_px > FALLBACK_GLYPH_TARGET:
        scale = FALLBACK_GLYPH_TARGET / glyph_px
    if max(w, h) * scale > FALLBACK_MAX_LONG_EDGE:
        scale = FALLBACK_MAX_LONG_EDGE / max(w, h)
    if scale < 0.999:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    return img.convert("RGB")


def _coerce_five_key(sj: dict) -> dict:
    """Normalize a generic VLM's output into the serving schema's shape.

    Fallback models get close but stray on details (section_type outside the
    enum, chunk_id as int, bbox as a 2-point list). Coercion is shape-only —
    content is never rewritten — and the result still has to pass
    validate_schema afterwards.
    """
    out = {
        "metadata": sj.get("metadata") if isinstance(sj.get("metadata"), dict) else {},
        "summary": sj.get("summary") if isinstance(sj.get("summary"), (str, type(None))) else None,
        "semantic_chunks": [], "figures": [], "tables": [],
    }
    for f in ("title", "authors", "organization", "year", "doc_type"):
        out["metadata"].setdefault(f, None)
    if not isinstance(out["metadata"].get("authors"), list):
        out["metadata"]["authors"] = ([out["metadata"]["authors"]]
                                      if isinstance(out["metadata"].get("authors"), str) else [])
    for i, c in enumerate(sj.get("semantic_chunks") or []):
        if not isinstance(c, dict) or not isinstance(c.get("content"), str):
            continue
        out["semantic_chunks"].append({
            "chunk_id": str(c.get("chunk_id") or f"c{i + 1}"),
            "title": c.get("title") if isinstance(c.get("title"), (str, type(None))) else None,
            "content": c["content"],
            "keywords": c.get("keywords") if isinstance(c.get("keywords"), list) else [],
            "section_type": (c.get("section_type")
                             if c.get("section_type") in SECTION_ENUM
                             else "technical_content"),
        })
    def _bbox(b):
        if (isinstance(b, list) and len(b) == 4
                and all(isinstance(v, (int, float)) for v in b)):
            return b
        return None
    for f in (sj.get("figures") or []):
        if isinstance(f, dict):
            out["figures"].append({"bbox": _bbox(f.get("bbox")),
                                   "caption": f.get("caption") if isinstance(f.get("caption"), (str, type(None))) else None})
    for t in (sj.get("tables") or []):
        if isinstance(t, dict) and isinstance(t.get("html"), str):
            out["tables"].append({"bbox": _bbox(t.get("bbox")), "html": t["html"],
                                  "caption": t.get("caption") if isinstance(t.get("caption"), (str, type(None))) else None})
    return out


def extract_page_fallback(hires_path, glyph_px, client, *, output_path=None) -> dict | None:
    """One flagged fallback attempt on the NATIVE page render.

    Returns the coerced 5-key dict on success, None on any failure — the
    caller keeps the primary failure record either way, so this can only ever
    upgrade a page, never fail one. The zlib gate still applies: fallback
    models loop too, and a looped rescue is worse than an honest failure.
    """
    from .vlm_client import make_image_content, make_text_content

    try:
        with Image.open(hires_path) as img:
            prepared = prepare_image_fallback(img, glyph_px)
        send_path = output_path or (str(hires_path) + ".fallback.png")
        prepared.save(send_path)

        result = client.chat_completion(
            [{"role": "system", "content": FALLBACK_SYSTEM},
             {"role": "user", "content": [make_image_content(send_path),
                                          make_text_content(FALLBACK_USER)]}],
            max_tokens=FALLBACK_MAX_TOKENS,
            temperature=0,
            timeout=FALLBACK_TIMEOUT_S,
        )
        choice = result["choices"][0]
        text = choice["message"].get("content") or ""
        if not text.strip() or choice.get("finish_reason") == "length":
            return None
        if zlib_ratio(text) > ZLIB_GATE_RATIO:
            logger.warning("fallback output degenerate (zlib %.1f) — discarded",
                           zlib_ratio(text))
            return None
        parsed, perr = parse_json_lenient(text)
        if parsed is None:
            logger.warning("fallback output unparseable: %s", perr)
            return None
        coerced = _coerce_five_key(parsed)
        ok, errs = validate_schema(coerced)
        if not ok:
            logger.warning("fallback output failed schema after coercion: %s", errs[:5])
            return None
        return coerced
    except Exception as e:
        logger.warning("fallback call failed: %s: %s", type(e).__name__, e)
        return None
