from app.assembly import aggregate_metadata, join_doc_summary, assemble_document


def _page(page, chunks=(), summary="s", meta=None, figures=(), tables=()):
    return {"page": page, "ok": True, "data": {
        "metadata": meta or {"title": None, "authors": [], "organization": None,
                             "year": None, "doc_type": None},
        "summary": summary, "semantic_chunks": list(chunks),
        "figures": list(figures), "tables": list(tables)}}


def test_metadata_first_non_null_and_doctype_majority():
    metas = [
        {"title": None, "authors": [], "organization": None, "year": None, "doc_type": "report"},
        {"title": "T", "authors": ["A"], "organization": None, "year": "2019", "doc_type": "paper"},
        {"title": "X", "authors": [], "organization": "Org", "year": None, "doc_type": "paper"},
    ]
    m = aggregate_metadata(metas)
    assert m["title"] == "T" and m["authors"] == ["A"]
    assert m["organization"] == "Org" and m["year"] == "2019"
    assert m["doc_type"] == "paper"          # majority 2-1


def test_doctype_tie_breaks_to_earliest_page():
    metas = [{"doc_type": "report"}, {"doc_type": "paper"}]
    assert aggregate_metadata(metas)["doc_type"] == "report"


def test_doc_summary_joins_non_null_in_order():
    assert join_doc_summary(["a", None, "b"]) == "a\nb"


def test_chunk_ids_and_provenance():
    doc = assemble_document("d1", "/f.pdf", "arxiv", [
        _page(1, chunks=[{"chunk_id": "1", "title": "t", "content": "Alpha.",
                          "keywords": [], "section_type": "introduction"}]),
        _page(2, chunks=[{"chunk_id": "1", "title": "u", "content": "Beta.",
                          "keywords": [], "section_type": "results"}]),
    ], stitch=False)
    assert [c["chunk_id"] for c in doc["chunks"]] == ["p1_c1", "p2_c1"]
    assert doc["chunks"][1]["pages"] == [2]


def test_failed_page_listed_but_contributes_nothing():
    doc = assemble_document("d1", "/f.pdf", "arxiv",
                            [_page(1), {"page": 2, "ok": False, "data": None}], stitch=False)
    assert doc["pages"][1] == {"page": 2, "page_summary": None, "hires_px": None, "ok": False}
    assert all(c["pages"] == [1] for c in doc["chunks"])
