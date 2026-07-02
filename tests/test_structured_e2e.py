"""Multi-page mock end-to-end smoke test for the v2 structured pipeline.

Exercises the FULL pipeline — real PDF rasterization, prepare_image
enhancement, per-page extraction, cross-page assembly, figure/table crop
writing, and structured.json/markdown synthesis — with only the model
client mocked. No network endpoint is ever contacted.

Multi-page input: a genuine 2-page PDF built with Pillow's multi-page PDF
writer (``Image.save(..., save_all=True, append_images=[...])``). This was
verified to work reliably in this environment: pypdfium2 (via
``rasterizer.classify_file`` / ``prepare_pages``) correctly classifies it as
"pdf" and rasterizes it to 2 pages (see
``test_two_page_pdf_classifies_and_rasterizes``), so the PNG-fallback
mentioned in the task spec was not needed.

Fake client note: ``run_structured_pipeline`` processes pages concurrently
(``vlm_concurrency=2``), so a plain "pop one reply per call" fake (as used
in ``test_structured_pipeline.py`` for single-page tests) can't guarantee
which canned reply lands on which page — thread scheduling decides call
order, not submission order. Instead ``PageAwareFakeClient`` below keys the
reply on the mean brightness of the page image embedded in the request:
page 1 is rendered near-white, page 2 near-black, and ``prepare_image``'s
grayscale + CLAHE pipeline preserves that separation, so each thread
deterministically gets the reply for the page it is actually processing.
"""

import base64
import io
import json

import numpy as np
from PIL import Image

from app.models import PipelineConfig
from app.rasterizer import classify_file, prepare_pages
from app.structured_pipeline import run_structured_pipeline

PAGE1_JSON = json.dumps({
    "metadata": {"title": "Doc Title", "authors": ["A"], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "Page 1 summary.",
    "semantic_chunks": [{"chunk_id": "1", "title": "Intro",
                          "content": "Intro chunk content.", "keywords": [],
                          "section_type": "introduction"}],
    "figures": [{"bbox": [100, 200, 900, 900], "caption": "A diagram"}],
    "tables": [],
})

PAGE2_JSON = json.dumps({
    "metadata": {"title": "Doc Title", "authors": ["A"], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "Page 2 summary.",
    "semantic_chunks": [{"chunk_id": "1", "title": "Results",
                          "content": "Result chunk content.", "keywords": [],
                          "section_type": "results"}],
    "figures": [],
    "tables": [{"bbox": [50, 50, 800, 400], "caption": "A table",
                "html": "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"}],
})


class PageAwareFakeClient:
    """Dispatches canned replies by mean brightness of the request image
    rather than by call order, so it stays correct under concurrent
    per-page dispatch (see module docstring)."""

    def __init__(self, bright_reply, dark_reply, threshold=128):
        self.bright_reply = bright_reply
        self.dark_reply = dark_reply
        self.threshold = threshold
        self.calls = []
        self.model = "shrew-9b"

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                         timeout=None, extra_params=None):
        data_uri = messages[1]["content"][0]["image_url"]["url"]
        b64_data = data_uri.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("L")
        mean = float(np.asarray(img).mean())
        self.calls.append(mean)
        text, finish_reason = self.bright_reply if mean > self.threshold else self.dark_reply
        return {"choices": [{"finish_reason": finish_reason,
                              "message": {"content": text}}]}


def _make_two_page_pdf(tmp_path, page1_gray=250, page2_gray=30):
    """A 2-page PDF: page 1 near-white, page 2 near-black."""
    p1 = Image.new("RGB", (1700, 2200), (page1_gray,) * 3)
    p2 = Image.new("RGB", (1700, 2200), (page2_gray,) * 3)
    pdf_path = tmp_path / "doc.pdf"
    p1.save(str(pdf_path), save_all=True, append_images=[p2])
    return pdf_path


def test_two_page_pdf_classifies_and_rasterizes(tmp_path):
    pdf_path = _make_two_page_pdf(tmp_path)
    assert classify_file(str(pdf_path)) == "pdf"

    out_dir = tmp_path / "precheck"
    page_images, total_pages, page_dims = prepare_pages(
        str(pdf_path), str(out_dir), low_dpi=100, high_dpi=200,
    )
    assert total_pages == 2
    assert sorted(page_images.keys()) == [1, 2]


def test_run_structured_pipeline_two_page_pdf_e2e(tmp_path):
    pdf_path = _make_two_page_pdf(tmp_path)
    out_dir = tmp_path / "out"

    fake = PageAwareFakeClient(
        bright_reply=(PAGE1_JSON, "stop"),
        dark_reply=(PAGE2_JSON, "stop"),
    )
    config = PipelineConfig(
        vlm_url="http://unused", vlm_model="shrew-9b",
        vlm_concurrency=2, low_dpi=100, high_dpi=200,
    )

    result = run_structured_pipeline(str(pdf_path), str(out_dir), config, client=fake)

    assert len(fake.calls) == 2
    assert result.processing_log["total_pages"] == 2
    assert result.processing_log["failed_pages"] == 0

    sj = result.structured_json

    # 2 chunks, one per page, each carrying page provenance.
    chunks = sj["semantic_chunks"]
    assert len(chunks) == 2
    by_page = {c["page"]: c for c in chunks}
    assert set(by_page) == {1, 2}
    assert by_page[1]["pages"] == [1]
    assert by_page[2]["pages"] == [2]
    assert by_page[1]["content"] == "Intro chunk content."
    assert by_page[2]["content"] == "Result chunk content."

    # 1 table, from page 2, with a real base64 crop written to disk.
    tables = sj["tables"]
    assert len(tables) == 1
    table = tables[0]
    assert table["page"] == 2
    assert table["data"] is not None
    base64.b64decode(table["data"])
    crop_path = out_dir / "crops" / f"{table['table_id']}.png"
    assert crop_path.exists()

    # 1 figure, from page 1, with base64 data.
    images = sj["images"]
    assert len(images) == 1
    fig = images[0]
    assert fig["page"] == 1
    assert fig["data"] is not None
    base64.b64decode(fig["data"])

    # markdown carries both chunks' content.
    md = result.clean_markdown
    assert "Intro chunk content." in md
    assert "Result chunk content." in md
