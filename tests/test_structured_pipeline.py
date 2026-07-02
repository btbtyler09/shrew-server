import base64
import json
from pathlib import Path

from PIL import Image

from app.assembly import assemble_document
from app.models import PipelineConfig
from app.structured_pipeline import (
    build_structured_json,
    run_structured_pipeline,
    synthesize_markdown,
)


class FakeClient:
    """Same shape as structured_page's tests — a queue of (text, finish_reason)
    replies popped one per chat_completion call. Never touches the network."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.model = "shrew-9b"

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                         timeout=None, extra_params=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        text, finish_reason = self.replies.pop(0)
        return {"choices": [{"finish_reason": finish_reason,
                              "message": {"content": text}}]}


GOOD_PAGE_JSON = json.dumps({
    "metadata": {"title": "Doc Title", "authors": ["A"], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "Page 1 summary.",
    "semantic_chunks": [{"chunk_id": "1", "title": "Intro",
                          "content": "Some intro content.", "keywords": [],
                          "section_type": "introduction"}],
    "figures": [{"bbox": [100, 200, 900, 900], "caption": "A diagram"}],
    "tables": [],
})


def _make_config(**overrides):
    cfg = PipelineConfig(vlm_url="http://unused", vlm_model="shrew-9b")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _page(page, chunks=(), summary="s", meta=None, figures=(), tables=()):
    return {"page": page, "ok": True, "data": {
        "metadata": meta or {"title": None, "authors": [], "organization": None,
                              "year": None, "doc_type": None},
        "summary": summary, "semantic_chunks": list(chunks),
        "figures": list(figures), "tables": list(tables)}}


# ── end-to-end wiring on a synthetic 1-page image ───────────────────────────


def test_run_structured_pipeline_wiring(tmp_path):
    doc_path = tmp_path / "doc.png"
    Image.new("RGB", (1700, 2200), "white").save(doc_path)
    out_dir = tmp_path / "out"

    client = FakeClient([(GOOD_PAGE_JSON, "stop")])
    config = _make_config()

    result = run_structured_pipeline(str(doc_path), str(out_dir), config, client=client)

    sj = result.structured_json
    assert set(sj.keys()) == {"metadata", "summary", "semantic_chunks", "tables", "images"}
    assert sj["metadata"]["type"] == "report"
    assert sj["metadata"]["id"]
    assert sj["metadata"]["num_chunks"] == 1
    assert sj["metadata"]["source_pages"] == 1

    assert len(sj["images"]) == 1
    img = sj["images"][0]
    assert img["data"] is not None
    # decoding should not raise — it's real base64 png bytes
    base64.b64decode(img["data"])

    crops_dir = out_dir / "crops"
    crop_files = list(crops_dir.glob("*.png"))
    assert len(crop_files) == 1

    assert result.processing_log["total_pages"] == 1
    assert result.processing_log["failed_pages"] == 0

    # model input got saved under pages/
    model_pngs = list((out_dir / "pages").glob("*_model.png"))
    assert len(model_pngs) == 1


def test_run_structured_pipeline_failed_page_contributes_nothing(tmp_path):
    doc_path = tmp_path / "doc.png"
    Image.new("RGB", (1700, 2200), "white").save(doc_path)
    out_dir = tmp_path / "out"

    # extract_page resamples once on parse failure then gives up (2 replies).
    client = FakeClient([("not json", "stop"), ("still not json", "stop")])
    config = _make_config()

    result = run_structured_pipeline(str(doc_path), str(out_dir), config, client=client)

    assert result.structured_json["semantic_chunks"] == []
    assert result.processing_log["failed_pages"] >= 1


# ── build_structured_json / synthesize_markdown on a hand-built doc ────────


def _two_page_doc(tmp_path):
    hires_path = tmp_path / "p1.png"
    Image.new("RGB", (2000, 1000)).save(hires_path)

    return assemble_document(
        "docid123", "/f.pdf", "arxiv",
        [
            _page(1,
                  chunks=[{"chunk_id": "1", "title": "Intro", "keywords": [],
                           "section_type": "introduction", "content": "Intro text."}],
                  figures=[{"bbox": [100, 200, 600, 700], "caption": "Fig one"}]),
            _page(2,
                  chunks=[{"chunk_id": "1", "title": "Results", "keywords": [],
                           "section_type": "results", "content": "Result text."}],
                  tables=[{"bbox": [100, 200, 600, 700], "caption": "Tbl one",
                           "html": "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"}]),
        ],
        hires_images={1: str(hires_path), 2: str(hires_path)},
        crops_dir=str(tmp_path / "crops"),
        stitch=False,
    )


def test_build_structured_json_tables_and_images(tmp_path):
    doc = _two_page_doc(tmp_path)
    sj = build_structured_json(doc, total_pages=2)

    assert sj["metadata"]["id"] == "docid123"
    assert sj["metadata"]["file_path"] == "/f.pdf"
    assert sj["metadata"]["source_pages"] == 2
    assert sj["metadata"]["num_chunks"] == 2
    assert sj["summary"] == doc["doc_summary"]
    assert sj["semantic_chunks"] == doc["chunks"]

    assert len(sj["tables"]) == 1
    table = sj["tables"][0]
    assert table["html"] == doc["tables"][0]["html"]
    assert table["flat_text"] == doc["tables"][0]["flat_text"]
    assert table["format"] == "png"
    assert table["data"] is not None
    base64.b64decode(table["data"])

    assert len(sj["images"]) == 1
    image = sj["images"][0]
    assert image["index"] == 1
    assert image["caption"] == "Fig one"
    assert image["data"] is not None
    base64.b64decode(image["data"])


def test_build_structured_json_image_without_crop_keeps_caption():
    doc = assemble_document(
        "d2", "/f.pdf", "arxiv",
        [_page(1, figures=[{"bbox": None, "caption": "Uncropped fig"}])],
        stitch=False,
    )
    sj = build_structured_json(doc, total_pages=1)
    assert len(sj["images"]) == 1
    assert sj["images"][0]["data"] is None
    assert sj["images"][0]["caption"] == "Uncropped fig"


def test_synthesize_markdown_contains_chunks_tables_and_figure_refs(tmp_path):
    doc = _two_page_doc(tmp_path)
    md = synthesize_markdown(doc)

    assert "Intro text." in md
    assert "Result text." in md
    assert "<table>" in md
    assert "![Fig one](img:1)" in md
