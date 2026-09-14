"""v0.3.13 (GitLab #26): the model emits a per-figure `description` (93% of
training labels, present in every raw E4/E6 generation inspected) and assembly
dropped it — 0 of ~1,300 figure records across 200 docs/arm carried one. It
was generated at full cost and never reached the output or the retrieval
index. Additive field: carried through assembly, multi-page docs, the
structured images[] projection, the fidelity layer (a corrected caption next
to an uncorrected description would be the v0.3.10 paired-field desync again),
the type gate, and the fallback coercer.
"""
import json

import pytest

from app.assembly import assemble_document, table_flat_text
from app.fallback import _coerce_five_key
from app.fidelity import apply_corrections, check_document
from app.structured_page import _gate
from app.structured_pipeline import build_structured_json


def _page(page, figs):
    return {"page": page, "ok": True, "status": "ok", "data": {
        "metadata": {"title": None, "authors": [], "organization": None,
                     "year": None, "doc_type": None},
        "summary": "s", "semantic_chunks": [], "tables": [], "figures": figs}}


def test_description_survives_assembly_and_multi_page():
    doc = assemble_document("d", "f.pdf", "pdf", [
        _page(1, [{"bbox": [0, 0, 100, 100], "caption": "Fig. 1",
                   "description": "Cross-section of a twist drill showing flute geometry."}]),
        _page(2, [{"bbox": [0, 0, 50, 50], "caption": "Fig. 2",
                   "description": "Load vs deflection curve for a simply supported beam."},
                  {"bbox": None, "caption": "Fig. 3"}]),  # no description emitted
    ], hires_images=None, stitch=False)
    figs = {f["figure_id"]: f for f in doc["figures"]}
    assert figs["d_p1_f1"]["description"].startswith("Cross-section")
    assert figs["d_p2_f1"]["description"].startswith("Load vs deflection")
    assert "description" in figs["d_p2_f2"] and figs["d_p2_f2"]["description"] is None


def test_description_reaches_structured_images_projection():
    doc = assemble_document("d", "f.pdf", "pdf", [
        _page(1, [{"bbox": [0, 0, 100, 100], "caption": "Fig. 1",
                   "description": "A gear train."}])], hires_images=None, stitch=False)
    sj = build_structured_json(doc, total_pages=1)
    assert sj["images"][0]["description"] == "A gear train."
    assert sj["images"][0]["caption"] == "Fig. 1"


def test_fidelity_corrects_description_in_sync_with_caption():
    # the same corrupted identifier in caption AND description must be fixed in both
    doc = {"doc_id": "d", "doc_summary": "", "metadata": {"title": None},
           "pages": [{"page": 1}], "chunks": [], "tables": [],
           "figures": [{"figure_id": "d_p1_f1", "page": 1, "bbox": None,
                        "caption": "DCRrectifierControl_PH wiring",
                        "description": "Shows the DCRrectifierControl_PH block.",
                        "crop_path": None}]}
    source = "The DCRectifierControl_PH block is described in section 7."
    apply_corrections(doc, check_document(doc, source))
    f = doc["figures"][0]
    assert "DCRectifierControl_PH" in f["caption"]
    assert "DCRectifierControl_PH" in f["description"]
    assert "DCRrectifierControl_PH" not in f["description"]


def test_non_string_description_is_a_schema_verdict():
    page = {"metadata": {"title": "T", "authors": [], "organization": None,
                         "year": None, "doc_type": "report"}, "summary": "s",
            "semantic_chunks": [], "tables": [],
            "figures": [{"bbox": None, "caption": "Fig", "description": 42}]}
    _parsed, verdict, _err = _gate(json.dumps(page), "stop")
    assert verdict == "schema"
    page["figures"][0]["description"] = None  # null is legal
    assert _gate(json.dumps(page), "stop")[1] == "ok"


def test_fallback_coercer_keeps_description():
    out = _coerce_five_key({"metadata": {}, "summary": None, "semantic_chunks": [],
                            "tables": [],
                            "figures": [{"bbox": [0, 0, 1, 1], "caption": "c",
                                         "description": "a valve assembly"},
                                        {"bbox": None, "caption": "x", "description": 7}]})
    assert out["figures"][0]["description"] == "a valve assembly"
    assert out["figures"][1]["description"] is None  # bad type coerced to null, not crash
