"""shrew-ocr-preview pipeline orchestrator.

One model, two modalities, one output schema (SHREW_OCR_PREVIEW.md §2):

  * pdf/image/office -> rasterize at 200 DPI -> preprocess.prepare_image
    (luminance, 100 DPI) -> extract_page                  [image modality]
  * text/csv/spreadsheet -> the deterministic extractor for that format
    (text_extract/spreadsheet_extract/msg_extract) -> paginate ->
    extract_text_page                                     [text modality]

Both produce the same per-page 5-key result, so assembly.py does all the
cross-page aggregation/stitching/cropping either way. This is the single-model
successor to run_pipeline_vlm: no docling, no editor, no multi-stage.

Two renderings of the same doc record:
  * structured (default) - markdown + the full structured.json
  * raw                  - a flat readable text document (title/metadata/
                           summary header, then chunks, tables as flat_text,
                           figures as captions), stage-3 keys omitted

The model client is injectable so tests never hit a live endpoint.
"""

import base64
import difflib
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from . import rasterizer
from .assembly import assemble_document, build_table_composite
from .models import PipelineResult
from .pipeline import CancelledException
from .preprocess import prepare_image
from .rasterizer import classify_file, prepare_pages
from .structured_page import extract_page, extract_text_page, paginate_text
from .text_extract import extract_text
from .vlm_client import VLMClient

# Classes with a deterministic extractor: these go through the text modality
# rather than being rasterized.
TEXT_CLASSES = {"text", "csv", "spreadsheet"}

# Formats whose deterministic parse is already a faithful structured rendering:
# openpyxl gives real cell values as a GFM table, the MAPI/email readers give
# real header fields and body. In `raw` mode these are returned as-is — asking
# the model to re-emit them costs a round-trip per page and can only flatten
# structure it already has.
#
# NOT included: html/md/txt/rtf. "Messy HTML/markdown/code/plain text" is
# exactly what the text modality was built for (§2), so raw mode runs those
# through the model and renders the result as flat text — a scraped page's nav
# chrome and boilerplate is what the model is there to resolve.
RAW_DETERMINISTIC_EXTENSIONS = {
    ".xlsx", ".xlsm", ".xls", ".ods",   # spreadsheets
    ".csv",                              # tabular
    ".eml", ".msg",                      # email
}


def _b64(path) -> str | None:
    """Read a file's bytes and base64-encode them (no data-uri prefix).

    None-safe: returns None if path is falsy or the file doesn't exist.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _page_result(page_no: int, res: dict) -> dict:
    """Project an extract_* result into the shape assemble_document wants,
    keeping the gate outcome so processing_log can report §5 health metrics."""
    return {
        "page": page_no,
        "ok": res["ok"],
        "data": res["data"],
        "status": res["status"],
        "schema_coerced": res.get("schema_coerced", False),
        "degenerate": res.get("degenerate", False),
    }


# The trained model-input geometry. Every page in the corpus was rendered at
# 200 DPI and halved to 100 DPI — uniformly 1700x2200 -> 850x1100, i.e. a long
# edge of 1100 (verified across the test split: no other size occurs).
TRAINED_LONG_EDGE = 1100


def _model_input_src_dpi(size, config, input_class: str) -> int:
    """The src_dpi to hand prepare_image so its output lands at the trained scale.

    prepare_image scales by a pure ratio target_dpi/src_dpi — correct only when
    the source really was rendered at src_dpi. That holds for pdf/office, which
    we rasterize ourselves at config.high_dpi.

    It does NOT hold for an uploaded bitmap: rasterizer.prepare_image_pages
    passes those through at their original size, so a 400-DPI scan arrives at
    3400x4400 and the fixed halving yields 1700x2200 — four times the trained
    pixel area, well out of distribution, and slow enough to hit the request
    timeout. Derive an effective DPI from the long edge instead.

    Never upscales: a small upload (a cropped figure, a thumbnail) has no detail
    to recover, so it is enhanced at its native size rather than blown up.
    """
    if input_class != "image":
        return config.high_dpi
    long_edge = max(size)
    if long_edge <= TRAINED_LONG_EDGE:
        return config.low_dpi  # ratio 1.0 — enhance in place, don't upscale
    return max(1, round(config.low_dpi * long_edge / TRAINED_LONG_EDGE))


def _process_one_page(page_no: int, hires_path, config, output_dir: str, client,
                       input_class: str = "pdf") -> dict:
    """Prepare the model input image for one page and run extraction on it."""
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    with Image.open(hires_path) as img:
        src_dpi = _model_input_src_dpi(img.size, config, input_class)
        enh = prepare_image(img, config.low_dpi, src_dpi)
    model_png = os.path.join(pages_dir, f"page_{page_no:04d}_model.png")
    enh.save(model_png)

    return _page_result(page_no, extract_page(model_png, client))


def _process_one_text_page(page_no: int, text: str, client) -> dict:
    """Run extraction on one paginated block of extracted text."""
    return _page_result(page_no, extract_text_page(text, client))


def _count_rows(html: str) -> int:
    return (html or "").count("<tr")


# Per-attempt cap for the refinement model call. The verified-good case (RFQ
# real-table composite) completes well under this; anything slower is the
# model enumerating filler from a padded form-like region, and the row-append
# fallback is already in hand.
_REFINE_TIMEOUT_S = 120


def make_table_refiner(hires_images: dict, output_dir: str, client, stats: dict):
    """Build the stitch_tables refine hook: re-extract a detected cross-page
    table from a composite of its two fragment crops.

    The composite (fragment crops heavily padded onto a full page-sized white
    canvas — exactly trained geometry) goes through the standard
    image-modality path: prepare_image then extract_page, same
    sentinel/sampling/gates as any page. The model resolves the seam properly
    (a row split across the break, rowspan structure), which row
    concatenation cannot. ONLY the re-extraction's tables are used; whatever
    metadata/summary/chunks/figures the model reads into the synthetic image
    are discarded.

    Declines (returns None -> deterministic row-append fallback) when the
    fragments cannot fit a page canvas at native text scale (never shrinks —
    sub-scale text empirically stalls the model), when extraction fails any
    gate, or when the re-extracted table has fewer rows than the larger
    fragment (the known dropped-trailing-tables failure mode). On success
    returns (html, crop_path); the hires composite doubles as the merged
    table's crop.
    """
    def refine(prev: dict, nxt: dict):
        stats["attempted"] = stats.get("attempted", 0) + 1
        prev_hires = hires_images.get(prev["page"])
        next_hires = hires_images.get(nxt["page"])
        if not prev_hires or not next_hires:
            return None

        with Image.open(prev_hires) as pi, Image.open(next_hires) as ni:
            composite = build_table_composite(pi, prev["bbox"], ni, nxt["bbox"])
        if composite is None:
            return None

        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        hires_path = os.path.join(crops_dir, f"{prev['table_id']}_stitched.png")
        composite.save(hires_path)

        # Standard 200->100 halving for rasterized pages — but the canvas
        # inherits the source page's pixel size, and for an uploaded scan that
        # is its native resolution (a 400-DPI scan gives a 3400x4400 canvas).
        # Normalize by long edge exactly like _model_input_src_dpi does, never
        # upscaling, so the model input lands at trained scale.
        eff_src = max(200, round(100 * max(composite.size) / TRAINED_LONG_EDGE))
        model_png = os.path.join(crops_dir, f"{prev['table_id']}_stitched_model.png")
        prepare_image(composite, 100, eff_src).save(model_png)

        # Refinement is an optional enhancement — it must never be able to
        # fail the document. Any transport error (timeout included) means
        # fall back to the row-append. The timeout is deliberately tighter
        # than the default: a composite that sends the model into a slow
        # enumeration (seen live on a form-heavy page) is not worth waiting
        # 300s for when a good deterministic merge is already in hand.
        try:
            res = extract_page(model_png, client, timeout=_REFINE_TIMEOUT_S)
        except Exception:
            return None
        if not res["ok"] or not res["data"]:
            return None
        tables = [t for t in res["data"].get("tables", []) if t.get("html")]
        if not tables:
            return None
        best = max(tables, key=lambda t: _count_rows(t["html"]))
        # The model may legitimately return FEWER rows than the two fragments
        # combined (it joins the seam row) — but fewer than the larger single
        # fragment means content was dropped.
        if _count_rows(best["html"]) < max(_count_rows(prev.get("html", "")),
                                            _count_rows(nxt.get("html", ""))):
            return None
        stats["model_refined"] = stats.get("model_refined", 0) + 1
        return best["html"], hires_path

    return refine


def build_structured_json(doc: dict, total_pages: int) -> dict:
    """Build the back-compat structured.json shape (plus a new `tables` key)
    from a doc record produced by assemble_document.
    """
    metadata = dict(doc["metadata"])
    metadata["type"] = doc["metadata"].get("doc_type")
    metadata["id"] = doc["doc_id"]
    metadata["file_path"] = doc["file_path"]
    metadata["source_pages"] = total_pages
    metadata["num_chunks"] = len(doc["chunks"])

    # `format` describes `data`, so it is only meaningful when a crop exists.
    # Text-modality pages have no render to crop from and the model returns
    # null bboxes there (§2), so every figure/table on that path is caption-only
    # — claiming "png" with data: null would advertise an image that cannot exist.
    tables = []
    for t in doc["tables"]:
        data = _b64(t.get("crop_path"))
        tables.append({
            "table_id": t["table_id"],
            "page": t["page"],
            "pages": t.get("pages", [t["page"]]),
            # Long tables are split into <=3-page segments so no single entry
            # outgrows an embedding/model context. Segments of one logical
            # table are linked both directions and each repeats the header
            # row: `continues` names the previous segment, `continued_by` the
            # next (null for ordinary tables / chain ends).
            "continues": t.get("continues"),
            "continued_by": t.get("continued_by"),
            "bbox": t["bbox"],
            "caption": t["caption"],
            "html": t["html"],
            "flat_text": t["flat_text"],
            "format": "png" if data else None,
            "data": data,
        })

    images = []
    for i, f in enumerate(doc["figures"], start=1):
        data = _b64(f.get("crop_path"))
        images.append({
            "index": i,
            "data": data,
            "format": "png" if data else None,
            "caption": f["caption"],
            "page": f["page"],
            "bbox": f["bbox"],
        })

    return {
        "metadata": metadata,
        "summary": doc["doc_summary"],
        "semantic_chunks": doc["chunks"],
        "tables": tables,
        "images": images,
    }


def _by_page(items: list) -> dict:
    """Bucket units (chunks/tables/figures, each carrying a `page`) by page."""
    out: dict = {}
    for it in items:
        out.setdefault(it["page"], []).append(it)
    return out


def _norm_caption(s) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(".")


# A figure/table caption is very often ALSO emitted as a semantic chunk — 73.5%
# of ground-truth captions are (measured over 6k test labels), so it is a
# trained convention, not a model defect: the caption is both page text (a
# chunk, so it is retrievable) and figure metadata (a bbox, so it is croppable).
# We never drop it from structured_json.semantic_chunks — that is the eval'd
# surface. These helpers only de-duplicate the RENDERED markdown/raw views.
_CAPTION_MATCH_RATIO = 0.9


def _chunk_matches_caption(chunk: dict, caption_norm: str) -> bool:
    if not caption_norm:
        return False
    for field in (chunk.get("content"), chunk.get("title")):
        cn = _norm_caption(field)
        if cn and difflib.SequenceMatcher(None, cn, caption_norm).ratio() >= _CAPTION_MATCH_RATIO:
            return True
    return False


def _figure_placements(doc: dict) -> tuple[dict, set]:
    """Map a caption-placeholder chunk to the figure it echoes.

    The model emits a figure's caption both as a figures[] entry (with a bbox,
    so it can be cropped) and as a semantic_chunk (73.5% of GT captions do
    this). That chunk's position in reading order is the only within-page
    signal we have for WHERE the figure belongs — the 5-key output carries no
    other ordering. So we render the image ref at that chunk's position instead
    of dumping it at the page bottom.

    Returns (placements, placed): placements maps chunk_id -> figure; placed is
    the set of figure_ids that found a home inline. First match wins; each chunk
    and each figure is used at most once.
    """
    placements: dict = {}
    placed: set = set()
    used_chunks: set = set()
    for f in doc["figures"]:
        if not f.get("crop_path"):
            continue  # no image to link — nothing to place
        cap = _norm_caption(f.get("caption"))
        if not cap:
            continue
        for ch in doc["chunks"]:
            if ch["chunk_id"] in used_chunks:
                continue
            if _chunk_matches_caption(ch, cap):
                placements[ch["chunk_id"]] = f
                used_chunks.add(ch["chunk_id"])
                placed.add(f["figure_id"])
                break
    return placements, placed


def _caption_placeholder_chunk_ids(doc: dict) -> set[str]:
    """chunk_ids whose content/title is essentially just a figure/table caption."""
    caps = [_norm_caption(f.get("caption")) for f in doc["figures"]]
    caps += [_norm_caption(t.get("caption")) for t in doc["tables"]]
    caps = [c for c in caps if c]
    ids: set[str] = set()
    for ch in doc["chunks"]:
        if any(_chunk_matches_caption(ch, cap) for cap in caps):
            ids.add(ch["chunk_id"])
    return ids


def synthesize_markdown(doc: dict) -> str:
    """Deterministic reconstruction of a document's markdown, one block per
    page wrapped in ``<page N> ... </page N>`` tags (matching the legacy
    pipeline's output shape). Within each page: its chunks as headed sections
    (verbatim — this is the retrieval-eval'd surface), then its tables (html
    verbatim), then a ``![caption](img:index)`` reference per figure.

    When a figure's caption already appears as a chunk on the page, the image
    ref drops the redundant caption (``![](img:N)``) rather than printing it a
    second time; the image link and the chunk both survive.

    The per-page 5-key model gives no within-page ordering, so a page's
    chunks/tables/figures are emitted in that fixed order; a chunk stitched
    across a page break is emitted under its origin page.
    """
    fig_index = {f["figure_id"]: i for i, f in enumerate(doc["figures"], start=1)}
    chunks_by_page = _by_page(doc["chunks"])
    tables_by_page = _by_page(doc["tables"])
    figures_by_page = _by_page(doc["figures"])
    placements, placed = _figure_placements(doc)

    blocks = []
    for page in doc["pages"]:
        p = page["page"]
        parts = []
        for chunk in chunks_by_page.get(p, []):
            fig = placements.get(chunk["chunk_id"])
            if fig is not None:
                # This chunk is a figure's caption echoed in reading order.
                # Render the image ref HERE, where the figure belongs, and drop
                # the chunk's own text (it is just the caption again). The
                # caption rides along as the image's alt text.
                caption = fig.get("caption") or ""
                parts.append(f"![{caption}](img:{fig_index[fig['figure_id']]})")
                continue
            title = chunk.get("title")
            if title:
                parts.append(f"## {title}")
            content = chunk.get("content") or ""
            if content:
                parts.append(content)
        for t in tables_by_page.get(p, []):
            parts.append(t["html"])
        for f in figures_by_page.get(p, []):
            if f["figure_id"] in placed:
                continue  # already emitted inline at its placeholder chunk
            caption = f.get("caption") or ""
            if f.get("crop_path"):
                # No placeholder chunk to anchor it — fall back to the page end.
                parts.append(f"![{caption}](img:{fig_index[f['figure_id']]})")
            else:
                # No crop (null bbox, or a text-modality page with nothing to
                # crop from). An ![](img:N) ref here would point at an image
                # that isn't in the response.
                parts.append(f"[Figure: {caption or 'untitled'}]")
        inner = "\n\n".join(parts)
        blocks.append(f"<page {p}>\n{inner}\n</page {p}>")

    return "\n\n".join(blocks).strip()


_META_LABELS = (
    ("title", "Title"),
    ("authors", "Authors"),
    ("organization", "Organization"),
    ("year", "Year"),
    ("doc_type", "Type"),
)

# Fixed section order. Every one is emitted on every document, even when empty,
# so a consumer can split on /^# / and rely on finding the same headers rather
# than probing for which ones happen to be present.
RAW_SECTIONS = ("Metadata", "Summary", "Content", "Tables", "Figures")

_EMPTY_SECTION = "(none)"

# Chunk content is arbitrary markdown and routinely contains its own headings —
# a title page comes back with a literal "# <document title>" in the body. Left
# alone those collide with the section headers above and break the /^# / split
# that makes fixed sections worth having. Demote every heading inside a section
# body one level (capped at h6) so nothing but a section starts at h1.
_HEADING_RX = re.compile(r"^(#{1,5})(\s)", re.M)


def _demote_headings(text: str) -> str:
    return _HEADING_RX.sub(r"#\1\2", text)


def render_raw_text(doc: dict) -> str:
    """Flatten a doc record into a plain readable text document.

    The `raw` rendering, in fixed sections:

        # Metadata   title/authors/organization/year/type, one per line
        # Summary    the joined document summary
        # Content    chunks, in <page N> blocks
        # Tables     each table's flat_text, tagged with its page
        # Figures    each figure's caption, tagged with its page

    Tables and figures live in their own sections rather than inline so the
    layout is deterministic, and each entry carries "(page N)" so page
    locality survives the move. No base64, no HTML — flat_text is the
    linearized "Header | Header / cell | cell" projection.
    """
    meta = doc.get("metadata") or {}
    sections: dict[str, str] = {}

    # ── Metadata ────────────────────────────────────────────────────────────
    meta_lines = []
    for key, label in _META_LABELS:
        value = meta.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        meta_lines.append(f"{label}: {value}")
    meta_lines.append(f"Pages: {len(doc.get('pages') or [])}")
    sections["Metadata"] = "\n".join(meta_lines)

    # ── Summary ─────────────────────────────────────────────────────────────
    sections["Summary"] = doc.get("doc_summary") or ""

    # ── Content ─────────────────────────────────────────────────────────────
    # Content is chunk *content* only — no chunk titles (the model's section
    # labels add noise to a flat reading), and caption-placeholder chunks are
    # skipped because their caption already appears once in the Tables/Figures
    # sections below. Real prose only.
    chunks_by_page = _by_page(doc["chunks"])
    placeholder_ids = _caption_placeholder_chunk_ids(doc)
    blocks = []
    for page in doc["pages"]:
        p = page["page"]
        parts: list[str] = []
        for chunk in chunks_by_page.get(p, []):
            if chunk["chunk_id"] in placeholder_ids:
                continue
            if chunk.get("content"):
                parts.append(_demote_headings(chunk["content"]))
        if parts:
            blocks.append(f"<page {p}>\n" + "\n\n".join(parts) + f"\n</page {p}>")
    sections["Content"] = "\n\n".join(blocks)

    # ── Tables ──────────────────────────────────────────────────────────────
    table_blocks = []
    for i, t in enumerate(doc["tables"], start=1):
        tpages = t.get("pages", [t["page"]])
        where = (f"page {t['page']}" if len(tpages) == 1
                 else f"pages {tpages[0]}–{tpages[-1]}")
        head = f"## Table {i} ({where})"
        if t.get("caption"):
            head += f" — {t['caption']}"
        # flat_text already leads with the caption when there is one, so use
        # it verbatim under the heading.
        table_blocks.append(f"{head}\n\n{t['flat_text'] or ''}".rstrip())
    sections["Tables"] = "\n\n".join(table_blocks)

    # ── Figures ─────────────────────────────────────────────────────────────
    figure_blocks = []
    for i, f in enumerate(doc["figures"], start=1):
        caption = f.get("caption") or "untitled"
        figure_blocks.append(f"## Figure {i} (page {f['page']}) — {caption}")
    sections["Figures"] = "\n\n".join(figure_blocks)

    out = []
    for name in RAW_SECTIONS:
        body = sections.get(name, "").strip()
        out.append(f"# {name}\n\n{body or _EMPTY_SECTION}")
    return "\n\n".join(out).strip()


def gate_metrics(page_results: list[dict]) -> dict:
    """§3/§5 health signals, per document.

    first-pass parse-fail rate, zlib-hit rate and coerced-retry success rate
    are the preview's live health signals; drift against the §6 baselines
    (parse-fail ~0.010, degeneration ~0.005) is the early warning that the
    serving input distribution shifted.
    """
    n = len(page_results)
    coerced = sum(1 for pr in page_results if pr.get("schema_coerced"))
    degenerate = sum(1 for pr in page_results if pr.get("degenerate"))
    statuses = [pr.get("status") for pr in page_results]
    # A page needed the retry tier iff it was coerced or ended in any failure.
    retried = coerced + sum(1 for s in statuses
                            if s in {"failed", "degenerate", "overlong_failed",
                                     "empty_completion"})
    return {
        "pages": n,
        "first_pass_ok": sum(1 for s in statuses if s == "ok"),
        "schema_coerced": coerced,
        "degenerate": degenerate,
        "oversize_filtered": sum(1 for s in statuses if s == "oversize"),
        "empty_completions": sum(1 for s in statuses if s == "empty_completion"),
        "overlong_failed": sum(1 for s in statuses if s == "overlong_failed"),
        "first_pass_fail_rate": round((n - sum(1 for s in statuses if s == "ok")) / n, 4) if n else 0.0,
        "degeneration_rate": round(degenerate / n, 4) if n else 0.0,
        "coerce_success_rate": round(coerced / retried, 4) if retried else None,
    }


def _raw_deterministic(file_path, output_dir, config, input_class,
                        start_time, progress) -> PipelineResult:
    """raw mode for the text family: return the deterministic parse as-is.

    No model call — see the call site for why. `total_pages` is 1 because the
    extractor produces one continuous document, not paginated output.
    """
    if progress is not None:
        progress.emit(10, f"Reading {input_class} input...")

    clean_markdown = extract_text(file_path, input_class, output_dir)

    with open(os.path.join(output_dir, "clean.md"), "w", encoding="utf-8") as f:
        f.write(clean_markdown)

    if progress is not None:
        progress.emit(95, "Document extracted")

    return PipelineResult(clean_markdown, {}, {
        "total_pages": 1,
        "total_figures": 0,
        "total_time_seconds": round(time.time() - start_time, 2),
        "failed_pages": 0,
        "modality": "deterministic",
        "gates": gate_metrics([]),
    })


def _extract_text_pages(file_path, output_dir, config, input_class, client, progress):
    """Text modality: deterministic extractor -> paginate -> per-page model call."""
    if progress is not None:
        progress.emit(5, f"Reading {input_class} input...")

    text = extract_text(file_path, input_class, output_dir)
    blocks = paginate_text(text)

    if progress is not None:
        progress.emit(10, f"Extracting pages (0/{len(blocks)})...")

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, config.vlm_concurrency)) as pool:
        futures = {
            pool.submit(_process_one_text_page, i, block, client): i
            for i, block in enumerate(blocks, start=1)
        }
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if progress is not None:
                progress.emit(10 + int(75 * done / max(1, len(blocks))),
                              f"Extracting pages ({done}/{len(blocks)})...")
                if progress.is_cancelled():
                    for pending in futures:
                        pending.cancel()
                    raise CancelledException()

    return [results[i] for i in sorted(results)], len(blocks), text


def run_structured_pipeline(file_path, output_dir, config, *, progress=None,
                             client=None, raw=False) -> PipelineResult:
    """Turn an uploaded file into a shrew-ocr-preview PipelineResult.

    Image modality (pdf/image/office):
      1. Rasterize pages (display + hires PNGs).
      2. Per page (concurrent, ascending-order collection): build the 100-DPI
         enhanced model input from the hires render, run extract_page.

    Text modality (text/csv/spreadsheet):
      1. Run the format's deterministic extractor (xlsx via openpyxl, .msg via
         the MAPI reader, html via bs4, ...).
      2. Paginate to page-sized blocks, run extract_text_page on each.

    Either way: assemble_document stitches/aggregates the per-page 5-key
    results into one doc record (cropping figures/tables from the hires images
    when there are any), then the doc record is rendered either as markdown +
    structured.json, or — with raw=True — as a flat text document.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    input_class = classify_file(file_path)

    if raw and os.path.splitext(file_path)[1].lower() in RAW_DETERMINISTIC_EXTENSIONS:
        return _raw_deterministic(file_path, output_dir, config, input_class,
                                   start_time, progress)

    client = client or VLMClient(
        base_url=config.vlm_url, model=config.vlm_model, api_key=config.api_key,
    )

    if progress is not None and progress.is_cancelled():
        raise CancelledException()

    hires_images: dict[int, str] | None = None

    if input_class in TEXT_CLASSES:
        page_results, total_pages, _ = _extract_text_pages(
            file_path, output_dir, config, input_class, client, progress,
        )
    else:
        with rasterizer.RASTERIZE_LOCK:
            page_images, total_pages, page_dims = prepare_pages(
                file_path, output_dir,
                low_dpi=config.low_dpi, high_dpi=config.high_dpi,
                page_range=config.page_range,
            )

        page_numbers = sorted(page_images.keys())
        n_pages = len(page_numbers)

        if progress is not None:
            progress.emit(5, f"Extracting pages (0/{n_pages})...")

        page_results_map: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, config.vlm_concurrency)) as pool:
            futures = {
                pool.submit(
                    _process_one_page, pno, page_images[pno][1], config, output_dir,
                    client, input_class,
                ): pno
                for pno in page_numbers
            }
            pages_done = 0
            for fut in as_completed(futures):
                pno = futures[fut]
                page_results_map[pno] = fut.result()
                pages_done += 1
                if progress is not None:
                    pct = 5 + int(80 * pages_done / n_pages)
                    progress.emit(pct, f"Extracting pages ({pages_done}/{n_pages})...")
                # Abort on client disconnect: cancel pending pages and stop.
                if progress is not None and progress.is_cancelled():
                    for pending in futures:
                        pending.cancel()
                    raise CancelledException()

        # Collect in ascending page order regardless of completion order.
        page_results = [page_results_map[pno] for pno in page_numbers]
        hires_images = {n: str(hires) for n, (disp, hires) in page_images.items()}

    with open(file_path, "rb") as f:
        doc_id = hashlib.sha256(f.read()).hexdigest()[:16]

    # Model-refined table stitching needs the hires renders — image modality
    # only (text-modality tables have null bboxes and never stitch anyway).
    stitch_stats: dict = {}
    table_refiner = (make_table_refiner(hires_images, output_dir, client, stitch_stats)
                     if hires_images else None)

    doc = assemble_document(
        doc_id, file_path, input_class, page_results,
        hires_images=hires_images, stitch=True,
        crops_dir=os.path.join(output_dir, "crops"),
        table_refiner=table_refiner,
    )

    if raw:
        # Flat text rendering: no structured.json, so the server omits the
        # stage-3 keys rather than returning empty ones.
        clean_markdown = render_raw_text(doc)
        structured_json = {}
    else:
        structured_json = build_structured_json(doc, total_pages)
        clean_markdown = synthesize_markdown(doc)

    if progress is not None:
        progress.emit(95, "Document assembled")

    elapsed = time.time() - start_time
    processing_log = {
        "total_pages": total_pages,
        "total_figures": len(doc["figures"]),
        "total_time_seconds": round(elapsed, 2),
        "failed_pages": sum(1 for pr in page_results if not pr["ok"]),
        "modality": "text" if input_class in TEXT_CLASSES else "image",
        "gates": gate_metrics(page_results),
        "table_stitch": {
            "attempted": stitch_stats.get("attempted", 0),
            "model_refined": stitch_stats.get("model_refined", 0),
        },
    }

    return PipelineResult(clean_markdown, structured_json, processing_log)
