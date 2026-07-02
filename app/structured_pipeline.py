"""v2 structured pipeline orchestrator.

Rasterize -> per-page model call (structured_page.extract_page) -> assemble
-> build_structured_json / synthesize_markdown. This is the single-model
successor to run_pipeline_vlm: no docling, no editor, no multi-stage —
one 9B-class model produces the 5-key extraction per page, and assembly.py
does all the cross-page aggregation/stitching/cropping.

The model client is injectable so tests never hit a live endpoint.
"""

import base64
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from . import rasterizer
from .assembly import assemble_document
from .models import PipelineResult
from .preprocess import prepare_image
from .rasterizer import classify_file, prepare_pages
from .structured_page import extract_page
from .vlm_client import VLMClient


def _b64(path) -> str | None:
    """Read a file's bytes and base64-encode them (no data-uri prefix).

    None-safe: returns None if path is falsy or the file doesn't exist.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _process_one_page(page_no: int, hires_path, config, output_dir: str, client) -> dict:
    """Prepare the model input image for one page and run extraction on it.

    Returns {"page": page_no, "ok": bool, "data": dict|None} — the shape
    assemble_document expects for page_results.
    """
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    with Image.open(hires_path) as img:
        enh = prepare_image(img, config.low_dpi, config.high_dpi)
    model_png = os.path.join(pages_dir, f"page_{page_no:04d}_model.png")
    enh.save(model_png)

    res = extract_page(model_png, client)
    return {"page": page_no, "ok": res["ok"], "data": res["data"]}


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

    tables = []
    for t in doc["tables"]:
        tables.append({
            "table_id": t["table_id"],
            "page": t["page"],
            "bbox": t["bbox"],
            "caption": t["caption"],
            "html": t["html"],
            "flat_text": t["flat_text"],
            "format": "png",
            "data": _b64(t.get("crop_path")),
        })

    images = []
    for i, f in enumerate(doc["figures"], start=1):
        images.append({
            "index": i,
            "data": _b64(f.get("crop_path")),
            "format": "png",
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


def synthesize_markdown(doc: dict) -> str:
    """Deterministic reconstruction of a document's markdown, page order.

    Chunks (already page-ascending) become headed sections, followed by a
    Tables section with each table's html verbatim, followed by a Figures
    section with one `![caption](img:index)` reference per figure — the
    same indexing build_structured_json uses for `images`.
    """
    parts = []

    for chunk in doc["chunks"]:
        title = chunk.get("title")
        if title:
            parts.append(f"## {title}\n\n")
        parts.append(f"{chunk.get('content') or ''}\n\n")

    if doc["tables"]:
        parts.append("## Tables\n\n")
        for t in doc["tables"]:
            parts.append(f"{t['html']}\n\n")

    if doc["figures"]:
        parts.append("## Figures\n\n")
        for i, f in enumerate(doc["figures"], start=1):
            caption = f.get("caption") or ""
            parts.append(f"![{caption}](img:{i})\n\n")

    return "".join(parts).strip()


def run_structured_pipeline(file_path, output_dir, config, *, progress=None, client=None) -> PipelineResult:
    """Turn an uploaded file into a v2 structured PipelineResult.

    1. Rasterize pages (display + hires PNGs).
    2. Per page (concurrent, ascending-order collection): build the
       100-DPI enhanced model input from the hires render and run
       structured_page.extract_page on it.
    3. assemble_document stitches/aggregates the per-page 5-key results
       into one doc record, cropping figures/tables from the hires images.
    4. build_structured_json / synthesize_markdown turn the doc record into
       the server's response shapes.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    with rasterizer.RASTERIZE_LOCK:
        page_images, total_pages, page_dims = prepare_pages(
            file_path, output_dir,
            low_dpi=config.low_dpi, high_dpi=config.high_dpi,
            page_range=config.page_range,
        )

    client = client or VLMClient(
        base_url=config.vlm_url, model=config.vlm_model, api_key=config.api_key,
    )

    page_numbers = sorted(page_images.keys())
    n_pages = len(page_numbers)

    if progress is not None:
        progress.emit(5, f"Extracting pages (0/{n_pages})...")

    page_results_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, config.vlm_concurrency)) as pool:
        futures = {
            pool.submit(
                _process_one_page, pno, page_images[pno][1], config, output_dir, client,
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

    # Collect in ascending page order regardless of completion order.
    page_results = [page_results_map[pno] for pno in page_numbers]

    with open(file_path, "rb") as f:
        doc_id = hashlib.sha256(f.read()).hexdigest()[:16]

    hires_images = {n: str(hires) for n, (disp, hires) in page_images.items()}

    doc = assemble_document(
        doc_id, file_path, classify_file(file_path), page_results,
        hires_images=hires_images, stitch=True,
        crops_dir=os.path.join(output_dir, "crops"),
    )

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
    }

    return PipelineResult(clean_markdown, structured_json, processing_log)
