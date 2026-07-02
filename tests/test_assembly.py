from pathlib import Path

from PIL import Image

from app.assembly import (
    aggregate_metadata,
    join_doc_summary,
    assemble_document,
    crop_bbox,
    table_flat_text,
)


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


def test_stitch_merges_midsentence_prose():
    doc = assemble_document("d", "/f.pdf", "arxiv", [
        _page(1, chunks=[{"chunk_id": "1", "title": "t", "keywords": ["k1"],
                          "section_type": "technical_content",
                          "content": "The reactor operates at a pressure of"}]),
        _page(2, chunks=[{"chunk_id": "1", "title": "", "keywords": ["k2"],
                          "section_type": "technical_content",
                          "content": "155 bar under normal conditions."}]),
    ])
    assert len(doc["chunks"]) == 1
    c = doc["chunks"][0]
    assert c["chunk_id"] == "p1_c1" and c["pages"] == [1, 2]
    assert c["content"] == "The reactor operates at a pressure of 155 bar under normal conditions."
    assert c["keywords"] == ["k1", "k2"]


def test_no_stitch_when_terminal_punctuation():
    doc = assemble_document("d", "/f.pdf", "arxiv", [
        _page(1, chunks=[{"chunk_id": "1", "title": "t", "keywords": [],
                          "section_type": "results", "content": "It ends here."}]),
        _page(2, chunks=[{"chunk_id": "1", "title": "u", "keywords": [],
                          "section_type": "results", "content": "and continues"}]),
    ])
    assert len(doc["chunks"]) == 2


def test_no_stitch_when_next_starts_uppercase():
    # prev mid-sentence ("...pressure of") but next starts "The result..." -> no merge
    doc = assemble_document("d", "/f.pdf", "arxiv", [
        _page(1, chunks=[{"chunk_id": "1", "title": "t", "keywords": [],
                          "section_type": "technical_content",
                          "content": "The reactor operates at a pressure of"}]),
        _page(2, chunks=[{"chunk_id": "1", "title": "u", "keywords": [],
                          "section_type": "technical_content",
                          "content": "The result was catastrophic."}]),
    ])
    assert len(doc["chunks"]) == 2


def test_no_stitch_across_failed_page_gap():
    # pages 1 and 3 present (page 2 failed) -> no merge even if texts line up
    doc = assemble_document("d", "/f.pdf", "arxiv", [
        _page(1, chunks=[{"chunk_id": "1", "title": "t", "keywords": [],
                          "section_type": "technical_content",
                          "content": "The reactor operates at a pressure of"}]),
        {"page": 2, "ok": False, "data": None},
        _page(3, chunks=[{"chunk_id": "1", "title": "u", "keywords": [],
                          "section_type": "technical_content",
                          "content": "155 bar under normal conditions."}]),
    ])
    assert len(doc["chunks"]) == 2


def test_no_stitch_when_list_like():
    # next content = "1. item one\n2. item two\n3. item three" -> digit-led list
    # head is a digit (clears _starts_like_continuation) but _is_prose still
    # rejects it (>=2 list-line matches) -> no merge
    doc = assemble_document("d", "/f.pdf", "arxiv", [
        _page(1, chunks=[{"chunk_id": "1", "title": "t", "keywords": [],
                          "section_type": "technical_content",
                          "content": "The reactor operates at a pressure of"}]),
        _page(2, chunks=[{"chunk_id": "1", "title": "u", "keywords": [],
                          "section_type": "technical_content",
                          "content": "1. item one\n2. item two\n3. item three"}]),
    ])
    assert len(doc["chunks"]) == 2


def test_crop_scales_from_hires_dims():
    img = Image.new("RGB", (2000, 1000))
    out = crop_bbox(img, [100, 200, 600, 700])
    assert out.size == (1000, 500)          # (600-100)/1000*2000, (700-200)/1000*1000


def test_crop_clamps_out_of_bounds():
    img = Image.new("RGB", (1000, 1000))
    out = crop_bbox(img, [-50, 0, 1200, 500])
    assert out.size == (1000, 500)


def test_crop_drops_degenerate_and_tiny():
    img = Image.new("RGB", (1000, 1000))
    assert crop_bbox(img, [500, 500, 400, 600]) is None      # x1 < x0
    assert crop_bbox(img, [0, 0, 30, 30]) is None            # 0.09% area


def test_flat_text_linearizes_rows():
    html = "<table><tr><th>Name</th><th>Qty</th></tr><tr><td>bolt</td><td>4</td></tr></table>"
    assert table_flat_text(html, "Parts list") == "Parts list\nName | Qty\nbolt | 4"


def test_assemble_document_produces_crops_and_hires_px(tmp_path):
    hires_path = tmp_path / "p1.png"
    Image.new("RGB", (2000, 1000)).save(hires_path)

    doc = assemble_document(
        "d1", "/f.pdf", "arxiv",
        [_page(1,
               tables=[{"bbox": [100, 200, 600, 700], "caption": "Parts list",
                        "html": "<table><tr><th>Name</th><th>Qty</th></tr>"
                                "<tr><td>bolt</td><td>4</td></tr></table>"}],
               figures=[{"bbox": [100, 200, 600, 700], "caption": "A figure"}])],
        hires_images={1: str(hires_path)},
        crops_dir=str(tmp_path),
        stitch=False,
    )

    assert doc["pages"][0]["hires_px"] == [2000, 1000]

    table = doc["tables"][0]
    assert table["flat_text"] == "Parts list\nName | Qty\nbolt | 4"
    assert table["crop_path"] == str(tmp_path / f"{table['table_id']}.png")
    assert Path(table["crop_path"]).is_file()

    figure = doc["figures"][0]
    assert figure["crop_path"] == str(tmp_path / f"{figure['figure_id']}.png")
    assert Path(figure["crop_path"]).is_file()
