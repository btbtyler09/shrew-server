"""Regression tests from the post-#8 review of "record field one step assumes
another step guaranteed" defects (GitLab #23). Three confirmed instances:

1. validate_schema() checked key PRESENCE but not TYPE for chunk content/title/
   keywords and figure/table caption. The first pass runs unenforced, so a
   model emitting ``"caption": 42`` or ``"content": null`` passed the gate and
   crashed assemble_document() (TypeError in table_flat_text / _is_prose).
   fallback.py already re-types exactly these fields — only the primary door
   was unguarded. Now a bad-typed page is a "schema" verdict, i.e. it takes
   the existing retry/fallback ladder instead of killing the conversion.
2. Embedded JPEG/GIF workbook pictures were written with their original
   bytes, yet build_structured_json() labels every figure ``format: "png"``.
   Now every extracted picture is normalised to a real PNG.
3. fidelity scanned a table's flat_text (whose first line IS the caption) but
   only rewrote html/flat_text — never ``caption`` — so a corrected identifier
   shipped next to its uncorrected spelling in the same record.
"""
import json
import os

import pytest
from openpyxl import Workbook

from app.assembly import table_flat_text
from app.fidelity import apply_corrections, check_document
from app.spreadsheet_extract import extract_spreadsheet_media
from app.structured_page import _gate

GOOD = {
    "metadata": {"title": "T", "authors": [], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "s",
    "semantic_chunks": [{"chunk_id": "1", "title": "A", "content": "body",
                         "keywords": [], "section_type": "introduction"}],
    "figures": [{"bbox": None, "caption": "Fig"}],
    "tables": [{"bbox": None, "caption": "Tab",
                "html": "<table><tr><td>x</td></tr></table>"}],
}


def _verdict(page):
    _parsed, verdict, _err = _gate(json.dumps(page), "stop")
    return verdict


def _with(**patch):
    page = json.loads(json.dumps(GOOD))
    for path, value in patch.items():
        coll, field = path.split("__")
        page[coll][0][field] = value
    return page


# ── 1. type gate ─────────────────────────────────────────────────────────────


def test_good_page_still_ok():
    assert _verdict(GOOD) == "ok"


@pytest.mark.parametrize("patch", [
    {"semantic_chunks__content": None},
    {"semantic_chunks__content": ["a", "b"]},
    {"semantic_chunks__title": 7},
    {"semantic_chunks__keywords": "not-a-list"},
    {"figures__caption": ["Fig 1"]},
    {"figures__caption": 42},
    {"tables__caption": 42},
    {"tables__caption": {"en": "Tab"}},
], ids=lambda p: next(iter(p)))
def test_bad_field_type_is_a_schema_verdict_not_a_crash(patch):
    assert _verdict(_with(**patch)) == "schema"


@pytest.mark.parametrize("patch", [
    {"figures__caption": None},         # null caption is legal
    {"tables__caption": None},
    {"semantic_chunks__title": None},   # null title is legal
], ids=lambda p: next(iter(p)))
def test_legal_nulls_are_not_over_rejected(patch):
    assert _verdict(_with(**patch)) == "ok"


def test_missing_keywords_is_not_over_rejected():
    page = json.loads(json.dumps(GOOD))
    del page["semantic_chunks"][0]["keywords"]
    assert _verdict(page) == "ok"


# ── 2. embedded media normalised to PNG ──────────────────────────────────────


def test_embedded_jpeg_and_gif_ship_as_real_png(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    from openpyxl.drawing.image import Image as XLImage
    jpg = tmp_path / "photo.jpg"
    PIL.new("RGB", (32, 32), (10, 200, 30)).save(jpg, "JPEG")
    gif = tmp_path / "anim.gif"
    PIL.new("P", (16, 16)).save(gif, "GIF")
    wb = Workbook()
    ws = wb.active
    ws.title = "S"
    ws.append(("a",))
    ws.add_image(XLImage(str(jpg)), "B2")
    ws.add_image(XLImage(str(gif)), "B10")
    p = tmp_path / "b.xlsx"
    wb.save(p)

    media = extract_spreadsheet_media(str(p), str(tmp_path))
    assert len(media) == 2
    for m in media:
        assert m["path"].endswith(".png"), m["path"]
        with open(m["path"], "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n", "not a real PNG"
        with PIL.open(m["path"]) as img:
            assert img.format == "PNG"


# ── 3. table caption corrected in sync with flat_text ───────────────────────


def test_table_caption_corrected_in_sync_with_flat_text():
    html = "<table><tr><td>Inst</td><td>Type</td></tr></table>"
    cap = "DCR instance mapping DCRrectifierControl_PH"  # extra R
    doc = {"doc_id": "d", "doc_summary": "", "metadata": {"title": None},
           "pages": [{"page": 1}], "chunks": [], "figures": [],
           "tables": [{"table_id": "t1", "page": 1, "pages": [1], "bbox": None,
                       "caption": cap, "html": html,
                       "flat_text": table_flat_text(html, cap)}]}
    source = "The instance is DCRectifierControl_PH in section 7."
    report = check_document(doc, source)
    assert apply_corrections(doc, report) >= 1
    t = doc["tables"][0]
    assert "DCRectifierControl_PH" in t["caption"]
    assert "DCRrectifierControl_PH" not in t["caption"]
    # paired representations must agree
    assert t["caption"] == t["flat_text"].splitlines()[0]
