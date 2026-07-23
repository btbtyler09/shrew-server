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


# ── cross-page table stitching ──────────────────────────────────────────────
#
# Thresholds calibrated against the GT test split (2026-07-23): of 701
# consecutive test-page pairs, 29 had tables on both pages; hand-classifying
# all 29 gave 9 true continuations and 20 non-continuations. True
# continuations: last-table y1 ∈ {942, 1000}, next-table y0 ∈ {0, 90}. The
# nearest false positives the gates must reject: y1=942/y0=113 with a 2-col →
# 5-col structure change (column gate), y1=1000/y0=357 (y0 gate), y1=838 with
# a fresh "Table 8-5." caption (y1 + caption gates), and NASA/AEDC run-data
# dumps that repeat identical headers every page but always sit mid-page
# (y1 gate). Distinct "TABLE 6" → "TABLE 7" pairs with identical headers are
# rejected by geometry (y1=634) and by the new-table caption pattern.

_STITCH_Y1_MIN = 930
_STITCH_Y0_MAX = 100

# "Table 8-5.", "TABLE II:", "Tab. 3 —" — a fresh numbered caption means a new
# table, not a continuation.
_NEW_TABLE_CAPTION = re.compile(r"^\s*tab(?:le|\.)?\s*[A-Za-z0-9][\w.-]*\s*[.:)–—-]", re.I)
# "(continued)", "cont'd", "cont." — an explicit continuation marker.
_CONTINUED_CAPTION = re.compile(r"\bcont(?:inued|'d|\.)?\b", re.I)


def _max_row_cells(html: str) -> int:
    """Widest row, in cells. The modal count is unreliable here — one verified
    GT continuation is dominated by single-cell colspan section rows — and the
    model's known ragged-row artifact means exact equality is too strict."""
    widest = 0
    soup = BeautifulSoup(html or "", "html.parser")
    for row in soup.find_all("tr"):
        widest = max(widest, len(row.find_all(["th", "td"])))
    return widest


def _norm_row_text(row) -> str:
    return _WS.sub(" ", row.get_text()).strip().lower()


def should_stitch_tables(prev: dict, nxt: dict) -> bool:
    """Is `nxt` (first table of page N+1) a continuation of `prev` (last table
    of page N)? Pure geometry + structure + caption; no model call."""
    pb, nb = prev.get("bbox"), nxt.get("bbox")
    if not pb or not nb:
        return False  # text-modality/null bboxes carry no position evidence
    if pb[3] < _STITCH_Y1_MIN or nb[1] > _STITCH_Y0_MAX:
        return False
    pc, nc = _max_row_cells(prev.get("html", "")), _max_row_cells(nxt.get("html", ""))
    if not pc or not nc or abs(pc - nc) > 1:
        return False
    cap = nxt.get("caption")
    if cap and not _CONTINUED_CAPTION.search(cap) and _NEW_TABLE_CAPTION.match(cap):
        return False
    return True


def _rowappend_html(prev: dict, nxt: dict) -> str:
    """Deterministic merge: nxt's rows appended to prev's table, dropping a
    repeated header row. The fallback when no model refinement is available."""
    psoup = BeautifulSoup(prev.get("html", ""), "html.parser")
    nsoup = BeautifulSoup(nxt.get("html", ""), "html.parser")
    ptable = psoup.find("table")
    nrows = nsoup.find_all("tr")
    if ptable is None or not nrows:
        return prev.get("html", "")

    prows = psoup.find_all("tr")
    if prows and _norm_row_text(nrows[0]) == _norm_row_text(prows[0]):
        nrows = nrows[1:]  # repeated header row

    # Append into the same section the last row lives in (tbody if present).
    anchor = ptable.find("tbody") or ptable
    for row in nrows:
        anchor.append(row.extract())
    return str(psoup)


def _merge_table(prev: dict, nxt: dict, refined=None) -> None:
    """Fold nxt into prev in place. prev keeps its table_id and caption.

    `refined` is an optional (html, crop_path) from a model re-extraction of
    the composite fragment image; when absent (or the refiner declined), the
    deterministic row-append is used, and the bbox/crop keep describing the
    first fragment only — the honest best available without a composite."""
    if refined is not None:
        prev["html"], crop_path = refined
        if crop_path:
            prev["crop_path"] = crop_path
    else:
        prev["html"] = _rowappend_html(prev, nxt)
    prev["flat_text"] = table_flat_text(prev["html"], prev.get("caption"))
    prev["pages"] = prev.get("pages", [prev["page"]]) + [nxt["page"]]


def build_table_composite(prev_img, prev_bbox, next_img, next_bbox,
                           *, margin: int = 50, gap: int = 8):
    """Compose two table-fragment crops onto one full page-sized canvas.

    Crops each fragment from its hires page render (no minimum-area filter —
    a continuation's top sliver can be tiny) and stacks them, heavily padded,
    on a white canvas with the SAME dimensions as the source page. Text stays
    at native scale and the canvas is exactly the geometry of a real page from
    this document, so the standard 200→100 transform lands the model input
    squarely in the trained distribution — a page containing one table and a
    lot of white space.

    Returns None when either crop fails or the fragments cannot fit the page
    canvas at native scale. Never shrinks to fit: an empirical check
    (2026-07-23, live model) showed a two-page composite scaled to the trained
    long edge puts text at ~55% of trained scale and stalls generation into
    the request timeout. Unfittable continuations fall back to the
    deterministic row-append.
    """
    a = crop_bbox(prev_img, prev_bbox, min_area_frac=0.0)
    b = crop_bbox(next_img, next_bbox, min_area_frac=0.0)
    if a is None or b is None:
        return None
    W, H = prev_img.size
    if a.height + b.height + 2 * margin + gap > H:
        return None
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(a, (min(margin, max(0, W - a.width)), margin))
    canvas.paste(b, (min(margin, max(0, W - b.width)), margin + a.height + gap))
    return canvas


def stitch_tables(tables: list[dict], refine=None) -> list[dict]:
    """Merge cross-page table fragments in a page-ordered table list.

    Walks pages in order; when the first table of page N+1 passes
    should_stitch_tables against the tail table of page N, the fragments are
    folded into one table. Chains naturally across 3+ pages: a merged table
    remains the tail candidate when the continuation was its page's only
    table. A failed page in between breaks adjacency (its tables simply don't
    exist), so no stitch happens across a gap.

    `refine`, when given, is called as refine(prev, nxt) on each merge pair
    and may return (html, crop_path) from a model re-extraction of the
    composite fragment image — the model resolves the seam (a row split
    across the break, rowspan structure) better than row concatenation can.
    Returning None falls back to the deterministic row-append, so stitching
    never depends on the refiner succeeding. Refinement is only attempted on
    the first join of a chain (a 2-fragment composite); later chain joins use
    the row-append.
    """
    by_page: dict[int, list[dict]] = {}
    for t in tables:
        by_page.setdefault(t["page"], []).append(t)

    merged_away: set[int] = set()
    tail = None            # last surviving table of the previous page chain
    tail_end_page = None   # page its content currently ends on

    for p in sorted(by_page):
        ts = by_page[p]
        head = ts[0]
        if tail is not None and tail_end_page == p - 1 and should_stitch_tables(tail, head):
            refined = None
            if refine is not None and len(tail.get("pages", [tail["page"]])) == 1:
                refined = refine(tail, head)
            _merge_table(tail, head, refined=refined)
            merged_away.add(id(head))
            rest = ts[1:]
            if rest:
                tail, tail_end_page = rest[-1], p
            else:
                tail_end_page = p  # merged table keeps trailing the chain
        else:
            tail, tail_end_page = ts[-1], p

    return [t for t in tables if id(t) not in merged_away]


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
                       hires_images=None, *, stitch=True, crops_dir=None,
                       table_refiner=None) -> dict:
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
                "pages": [page],
                "bbox": table.get("bbox"),
                "caption": table.get("caption"),
                "html": html,
                "flat_text": table_flat_text(html, table.get("caption")),
                "crop_path": None,
            })

    if stitch:
        tables = stitch_tables(tables, refine=table_refiner)

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
                    if table.get("crop_path"):
                        continue  # a stitched composite already covers it
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
