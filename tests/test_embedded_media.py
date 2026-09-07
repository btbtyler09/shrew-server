"""Regression: embedded spreadsheet media through the non-raw structured
pipeline (GitLab #8).

run_structured_pipeline() appends embedded-workbook pictures to doc["figures"]
*after* assemble_document() — which is the only thing that stamps figure_id.
Those figures had none, so synthesize_markdown()'s

    fig_index = {f["figure_id"]: i for i, f in enumerate(doc["figures"], 1)}

KeyError'd and every spreadsheet-with-media conversion failed under
shrew-server 0.3.8. The fix assigns each a deterministic, unique id and keeps
the media in both the markdown and the structured image projection.
"""
import json
import threading

import pytest
from openpyxl import Workbook

from app.models import PipelineConfig
from app.structured_pipeline import _validate_document, run_structured_pipeline

# Satisfies both the text-page schema and the media-caption call (extract_page
# reads figures[0].caption). One reply serves every call the run makes.
GOOD = json.dumps({
    "metadata": {"title": "Sheet", "authors": [], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "s",
    "semantic_chunks": [{"chunk_id": "1", "title": "T", "content": "body",
                         "section_type": "introduction"}],
    "figures": [{"bbox": [0, 0, 10, 10], "caption": "A red square"}],
    "tables": [],
})


class _AlwaysClient:
    """Returns the same page for every call. Thread-safe for the text pool;
    never touches the network."""
    model = "shrew-9b"

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                        timeout=None, extra_params=None):
        with self._lock:
            self.calls += 1
        return {"choices": [{"finish_reason": "stop",
                             "message": {"content": GOOD}}]}


def _xlsx_with_embedded_image(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    from openpyxl.drawing.image import Image as XLImage
    png = tmp_path / "logo.png"
    PIL.new("RGB", (24, 24), (200, 30, 30)).save(png)
    wb = Workbook()
    ws = wb.active
    ws.title = "Front"
    for row in [("Quarter", "Revenue"), ("Q1", 10), ("Q2", 20)]:
        ws.append(row)
    ws.add_image(XLImage(str(png)), "D2")
    p = tmp_path / "book.xlsx"
    wb.save(p)
    return str(p)


def test_embedded_media_completes_and_is_retained(tmp_path):
    path = _xlsx_with_embedded_image(tmp_path)
    cfg = PipelineConfig(vlm_url="http://unused", vlm_model="shrew-9b")

    # 0.3.8 raised KeyError('figure_id') from synthesize_markdown here.
    result = run_structured_pipeline(path, str(tmp_path / "out"), cfg,
                                     client=_AlwaysClient())

    images = result.structured_json["images"]
    embedded = [f for f in images if "Embedded image" in (f["caption"] or "")]
    assert len(embedded) == 1, images
    # media preserved: crop payload, null page, null bbox
    assert embedded[0]["data"], "embedded media lost its crop payload"
    assert embedded[0]["page"] is None and embedded[0]["bbox"] is None
    # markdown retains it as an image ref rather than dropping it
    assert f"(img:{embedded[0]['index']})" in result.clean_markdown


def test_validate_document_requires_unique_figure_ids():
    ok = {"figures": [{"figure_id": "d_p1_f1"}, {"figure_id": "d_embedded_f1"}]}
    _validate_document(ok)  # no raise

    with pytest.raises(ValueError, match="figure_id"):
        _validate_document({"figures": [{"figure_id": "d_p1_f1"}, {"caption": "x"}]})

    with pytest.raises(ValueError, match="duplicate"):
        _validate_document({"figures": [{"figure_id": "dup"}, {"figure_id": "dup"}]})
