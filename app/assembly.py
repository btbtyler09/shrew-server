"""Assemble per-page 5-key VLM extractions into one canonical document record.

This module owns metadata aggregation, doc-summary joining, and
chunk/table/figure numbering + provenance. Cross-page chunk stitching merges
a page's first prose chunk into the previous page's last chunk when the text
clearly continues across the page break (assembly-spec §3). When hires page
images and a crops directory are supplied, figure/table bboxes are cropped
from the hires render (crop_bbox) and table HTML is linearized to flat_text
(table_flat_text) for embedding/search.
"""

import os
import re
from collections import Counter

from bs4 import BeautifulSoup
from PIL import Image

_TERMINAL_END = re.compile(r'[.!?:;]["\')\]]*$')
_LIST_LINE = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+', re.M)
_WS = re.compile(r'\s+')


def crop_bbox(hires_img, bbox, *, min_area_frac: float = 0.015):
    """Crop a 0-1000 normalized bbox out of a hires page image.

    Clamps coordinates to [0, 1000], returns None for degenerate boxes
    (x1<=x0 or y1<=y0 after clamping) or boxes below min_area_frac of the
    page area (measured in normalized space), otherwise scales the bbox to
    the hires image's pixel dimensions and returns the cropped image.
    """
    x0, y0, x1, y1 = (max(0, min(1000, c)) for c in bbox)
    if x1 <= x0 or y1 <= y0:
        return None
    if ((x1 - x0) * (y1 - y0)) / (1000 * 1000) < min_area_frac:
        return None

    W, H = hires_img.size
    left = x0 / 1000 * W
    right = x1 / 1000 * W
    upper = y0 / 1000 * H
    lower = y1 / 1000 * H
    return hires_img.crop((int(left), int(upper), int(right), int(lower)))


def table_flat_text(html: str, caption) -> str:
    """Linearize a table's HTML (+ optional caption) into plain text.

    First line (if caption is truthy) is the caption; one line per <tr>
    follows, with th/td cell texts joined by " | ", all whitespace
    normalized (collapsed internal runs, stripped).
    """
    lines = []
    if caption:
        lines.append(_WS.sub(" ", caption).strip())

    soup = BeautifulSoup(html, "html.parser")
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        cell_texts = [_WS.sub(" ", cell.get_text()).strip() for cell in cells]
        lines.append(" | ".join(cell_texts))

    return "\n".join(lines)


def _is_prose(text: str) -> bool:
    return len(_LIST_LINE.findall(text)) < 2


def _starts_like_continuation(text: str) -> bool:
    head = text.lstrip()[:1]
    return bool(head) and (head.islower() or head.isdigit())


def stitch_pages(pages_chunks: list[tuple[int, list[dict]]]) -> list[dict]:
    """Merge a page's first prose chunk into the previous page's last chunk
    when the content clearly continues across the page break.

    pages_chunks: list of (page, [chunk-dicts for that page]) in ascending
    page order, chunk dicts already carrying chunk_id/title/content/keywords/
    section_type/page. Merge conditions (assembly-spec §3): the previous
    assembled chunk's last page is page-1, both contents are prose (not
    list-like), the previous content does not end in terminal punctuation,
    and the next content starts lowercase/digit. Merged chunks keep the
    earlier chunk_id/title and union keywords (order-preserving).
    """
    merged = []
    for page, chunks in pages_chunks:
        for i, src in enumerate(chunks):
            c = {**src, "pages": [page]}
            prev = merged[-1] if merged else None
            if (i == 0 and prev is not None and prev["pages"][-1] == page - 1
                    and _is_prose(prev["content"]) and _is_prose(c["content"])
                    and not _TERMINAL_END.search(prev["content"].rstrip())
                    and _starts_like_continuation(c["content"])):
                prev["content"] = prev["content"].rstrip() + " " + c["content"].lstrip()
                prev["pages"].append(page)
                prev["keywords"] = list(dict.fromkeys(prev["keywords"] + c["keywords"]))
                continue
            merged.append(c)
    return merged


def aggregate_metadata(page_metas: list[dict]) -> dict:
    """Combine per-page metadata dicts into one document-level metadata dict.

    title/authors/organization/year: first non-empty value in page order.
    doc_type: majority vote among non-null values; ties break to the
    earliest page (i.e. the first value achieving the max count).
    """
    out = {k: None for k in ("title", "authors", "organization", "year", "doc_type")}
    for k in ("title", "authors", "organization", "year"):
        for m in page_metas:
            v = m.get(k)
            if v:
                out[k] = v
                break
    votes = [m["doc_type"] for m in page_metas if m.get("doc_type")]
    if votes:
        counts = Counter(votes)
        best = max(counts.values())
        out["doc_type"] = next(v for v in votes if counts[v] == best)
    return out


def join_doc_summary(page_summaries: list) -> str:
    """Join non-empty per-page summaries in order with newlines."""
    return "\n".join(s for s in page_summaries if s)


def assemble_document(doc_id, file_path, source, page_results,
                       hires_images=None, *, stitch=True, crops_dir=None) -> dict:
    """Build the canonical document record from ordered per-page results.

    page_results: list of {"page": int, "ok": bool, "data": dict|None} in
    ascending page order. `data` (when ok) is the 5-key extraction dict
    {metadata, summary, semantic_chunks, figures, tables}.
    """
    ok_pages = [pr for pr in page_results if pr.get("ok")]

    metadata = aggregate_metadata([pr["data"]["metadata"] for pr in ok_pages])
    doc_summary = join_doc_summary([pr["data"]["summary"] for pr in ok_pages])

    pages = []
    for pr in page_results:
        if pr.get("ok"):
            pages.append({
                "page": pr["page"],
                "page_summary": pr["data"]["summary"],
                "hires_px": None,
                "ok": True,
            })
        else:
            pages.append({
                "page": pr["page"],
                "page_summary": None,
                "hires_px": None,
                "ok": False,
            })

    pages_chunks = []
    for pr in ok_pages:
        page = pr["page"]
        page_chunks = []
        for n, chunk in enumerate(pr["data"]["semantic_chunks"], start=1):
            page_chunks.append({
                "chunk_id": f"p{page}_c{n}",
                "pages": [page],
                "page": page,
                "title": chunk.get("title"),
                "content": chunk.get("content"),
                "keywords": chunk.get("keywords", []),
                "section_type": chunk.get("section_type"),
            })
        pages_chunks.append((page, page_chunks))

    if stitch:
        chunks = stitch_pages(pages_chunks)
    else:
        chunks = [c for _, page_chunks in pages_chunks for c in page_chunks]

    tables = []
    for pr in ok_pages:
        page = pr["page"]
        for i, table in enumerate(pr["data"]["tables"], start=1):
            html = table.get("html", "")
            tables.append({
                "table_id": f"{doc_id}_p{page}_t{i}",
                "page": page,
                "bbox": table.get("bbox"),
                "caption": table.get("caption"),
                "html": html,
                "flat_text": table_flat_text(html, table.get("caption")),
                "crop_path": None,
            })

    figures = []
    for pr in ok_pages:
        page = pr["page"]
        for i, figure in enumerate(pr["data"]["figures"], start=1):
            figures.append({
                "figure_id": f"{doc_id}_p{page}_f{i}",
                "page": page,
                "bbox": figure.get("bbox"),
                "caption": figure.get("caption"),
                "crop_path": None,
            })

    if hires_images and crops_dir:
        os.makedirs(crops_dir, exist_ok=True)
        pages_by_no = {p["page"]: p for p in pages}
        page_tables = {}
        page_figures = {}
        for table in tables:
            page_tables.setdefault(table["page"], []).append(table)
        for figure in figures:
            page_figures.setdefault(figure["page"], []).append(figure)

        for pr in ok_pages:
            page = pr["page"]
            hires_path = hires_images.get(page)
            if hires_path is None:
                continue
            with Image.open(hires_path) as hires_img:
                hires_img.load()
                W, H = hires_img.size
                pages_by_no[page]["hires_px"] = [W, H]

                for table in page_tables.get(page, []):
                    if table["bbox"] is None:
                        continue
                    crop = crop_bbox(hires_img, table["bbox"])
                    if crop is not None:
                        crop_path = os.path.join(crops_dir, f"{table['table_id']}.png")
                        crop.save(crop_path)
                        table["crop_path"] = crop_path

                for figure in page_figures.get(page, []):
                    if figure["bbox"] is None:
                        continue
                    crop = crop_bbox(hires_img, figure["bbox"])
                    if crop is not None:
                        crop_path = os.path.join(crops_dir, f"{figure['figure_id']}.png")
                        crop.save(crop_path)
                        figure["crop_path"] = crop_path

    return {
        "doc_id": doc_id,
        "file_path": file_path,
        "source": source,
        "metadata": metadata,
        "doc_summary": doc_summary,
        "pages": pages,
        "chunks": chunks,
        "tables": tables,
        "figures": figures,
    }
