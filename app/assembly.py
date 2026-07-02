"""Assemble per-page 5-key VLM extractions into one canonical document record.

This module owns metadata aggregation, doc-summary joining, and
chunk/table/figure numbering + provenance. Cross-page chunk stitching and
hires crop population (flat_text for tables, crop_path for tables/figures)
are filled in by later tasks; the seams are marked below.
"""

from collections import Counter


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

    # TODO(task5): stitch_pages when stitch=True
    chunks = []
    for pr in ok_pages:
        page = pr["page"]
        for n, chunk in enumerate(pr["data"]["semantic_chunks"], start=1):
            chunks.append({
                "chunk_id": f"p{page}_c{n}",
                "pages": [page],
                "page": page,
                "title": chunk.get("title"),
                "content": chunk.get("content"),
                "keywords": chunk.get("keywords", []),
                "section_type": chunk.get("section_type"),
            })

    tables = []
    for pr in ok_pages:
        page = pr["page"]
        for i, table in enumerate(pr["data"]["tables"], start=1):
            tables.append({
                "table_id": f"{doc_id}_p{page}_t{i}",
                "page": page,
                "bbox": table.get("bbox"),
                "caption": table.get("caption"),
                "html": table.get("html", ""),
                "flat_text": "",
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
