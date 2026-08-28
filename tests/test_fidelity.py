"""Deterministic-source fidelity check (vocab cross-check).

Origin: a user's workspace agent reported a "typo" — DCRrectifierControl_PH
(extra R) — that did not exist in the source Word document. Reproduced at home
(2026-08-28, shrew fine-tune, real pipeline path, temp 0, deterministic): on an
identifier-dense 9pt table page the transcription itself was faithful, but the
model's GENERATED summary blended the caption ("DCR instance mapping") with the
type name and emitted DCRRectifierStatus_PH / DCRRectifierControl_PH. Copying
is safe; paraphrasing identifiers is the hazard.

The check: build a vocabulary of identifier-shaped tokens from a deterministic
extraction of the source (office → LibreOffice text, born-digital PDF → text
layer), then flag identifier-shaped tokens in the pipeline output that don't
exist in the source. Flag-only: results ride processing_log, nothing is
rewritten.
"""
import json

import pytest

from app.fidelity import (
    check_output,
    edit_distance,
    extract_vocab,
    iter_output_texts,
)


# Verbatim from the reproduced corrupt first-pass output (probe 2026-08-28,
# ident_d200_p9 pp=0): the summary paraphrase, not the table cells.
CORRUPT_SUMMARY = (
    "7.15 Control Object Instances table for the PH substation, mapping DCR "
    "instance names to DCRRectifierStatus_PH, DCRRectifierControl_PH, and "
    "ACLineSegmentCtl_PH types and locations."
)

SOURCE_TEXT = """7.15 Control Object Instances
Table 95 - DCR instance mapping (PH substation)
Instance Type Location
DCRectifierStatus01 DCRectifierStatus_PH Bay 2, unit 1
ACLineSegmentCtl02 ACLineSegmentCtl_PH Bay 3, unit 2
DCRectifierControl03 DCRectifierControl_PH Bay 4, unit 3
"""


# ── vocabulary extraction ───────────────────────────────────────────────────


def test_extract_vocab_picks_identifier_shaped_tokens():
    vocab = extract_vocab(SOURCE_TEXT)
    assert "DCRectifierControl_PH" in vocab
    assert "ACLineSegmentCtl_PH" in vocab
    assert "DCRectifierStatus01" in vocab  # digits count as identifier-shaped


def test_extract_vocab_ignores_prose_words():
    vocab = extract_vocab(SOURCE_TEXT)
    for w in ("Instance", "Type", "Location", "Control", "Object", "unit", "Bay"):
        assert w not in vocab, f"plain word {w!r} must not enter the vocab"


def test_extract_vocab_empty_source():
    assert extract_vocab("") == set()
    assert extract_vocab(None) == set()


# ── flagging ────────────────────────────────────────────────────────────────


def test_flags_the_reproduced_corruption_with_closest_match():
    vocab = extract_vocab(SOURCE_TEXT)
    flagged = check_output(CORRUPT_SUMMARY, vocab)
    tokens = {f["token"]: f for f in flagged}
    assert "DCRRectifierControl_PH" in tokens
    assert "DCRRectifierStatus_PH" in tokens
    f = tokens["DCRRectifierControl_PH"]
    assert f["closest"] == "DCRectifierControl_PH"
    assert f["distance"] == 1


def test_users_reported_form_is_flagged_too():
    # The exact form from the user report: lowercase r inserted.
    vocab = extract_vocab(SOURCE_TEXT)
    flagged = check_output("instance DCRrectifierControl_PH is misconfigured", vocab)
    assert flagged and flagged[0]["token"] == "DCRrectifierControl_PH"
    assert flagged[0]["closest"] == "DCRectifierControl_PH"
    assert flagged[0]["distance"] == 1


def test_faithful_output_flags_nothing():
    vocab = extract_vocab(SOURCE_TEXT)
    ok = ("The table maps DCRectifierControl03 (type DCRectifierControl_PH) "
          "to Bay 4, and ACLineSegmentCtl_PH instances to their bays.")
    assert check_output(ok, vocab) == []


def test_prose_only_output_flags_nothing_even_if_absent_from_source():
    """Plain English the model adds (summaries!) must never be flagged —
    only identifier-shaped tokens are checked."""
    vocab = extract_vocab(SOURCE_TEXT)
    out = "This section describes rectifier control equipment configuration."
    assert check_output(out, vocab) == []


def test_identifier_with_no_near_match_is_still_flagged_without_closest():
    vocab = extract_vocab(SOURCE_TEXT)
    flagged = check_output("see FooBarBazQuux_99 for details", vocab)
    assert flagged and flagged[0]["token"] == "FooBarBazQuux_99"
    assert flagged[0]["closest"] is None


def test_empty_vocab_flags_nothing():
    """No deterministic source (scanned PDF, image upload) → the check is
    unavailable, not a firehose of false flags."""
    assert check_output(CORRUPT_SUMMARY, set()) == []


def test_flagged_tokens_are_deduplicated():
    vocab = extract_vocab(SOURCE_TEXT)
    out = "DCRRectifierControl_PH here and DCRRectifierControl_PH there"
    flagged = check_output(out, vocab)
    assert len([f for f in flagged if f["token"] == "DCRRectifierControl_PH"]) == 1


# ── the wider precision-token family: same flavor, other classes ────────────


ENG_SOURCE = """3.2 Protocol Mapping per IEC-61850-7-4
The SCADA gateway polls DNP3 outstations. Register base 0x1A2B, offset 0x0040.
Fasteners: M8x1.25 socket head, torque 22.5 Nm (see Table 95).
Firmware v2.14.3 required. Section 7.15 lists instances.
"""


def test_acronym_near_miss_is_flagged():
    """SCDA is not a word the model may invent — it's a mangled SCADA."""
    vocab = extract_vocab(ENG_SOURCE)
    flagged = check_output("The SCDA gateway configuration", vocab)
    assert flagged and flagged[0]["token"] == "SCDA"
    assert flagged[0]["closest"] == "SCADA"


def test_unmangled_acronym_absent_from_source_is_not_flagged():
    """An acronym the model legitimately adds (e.g. naming a concept the doc
    never abbreviates) is not evidence of corruption — only near-misses are."""
    vocab = extract_vocab(ENG_SOURCE)
    assert check_output("configured via MODBUS registers", vocab) == []


def test_standard_reference_corruption_is_flagged():
    vocab = extract_vocab(ENG_SOURCE)
    flagged = check_output("per IEC-61850-7-5 clause 12", vocab)
    tokens = {f["token"]: f for f in flagged}
    assert "IEC-61850-7-5" in tokens
    assert tokens["IEC-61850-7-5"]["closest"] == "IEC-61850-7-4"


def test_dotted_section_number_corruption_is_flagged():
    vocab = extract_vocab(ENG_SOURCE)
    flagged = check_output("described in section 7.5 of this document", vocab)
    assert any(f["token"] == "7.5" and f["closest"] == "7.15" for f in flagged)


def test_version_and_hex_corruption_are_flagged():
    vocab = extract_vocab(ENG_SOURCE)
    flagged = check_output("requires v2.4.3 at base 0x1A2F", vocab)
    tokens = {f["token"]: f["closest"] for f in flagged}
    assert tokens.get("v2.4.3") == "v2.14.3"
    assert tokens.get("0x1A2F") == "0x1A2B"


def test_part_code_corruption_is_flagged():
    vocab = extract_vocab(ENG_SOURCE)
    flagged = check_output("use M8x1.5 socket head fasteners", vocab)
    assert any(f["token"] == "M8x1.5" and f["closest"] == "M8x1.25"
               for f in flagged)


def test_decimal_value_corruption_is_flagged():
    vocab = extract_vocab(ENG_SOURCE)
    flagged = check_output("torque to 22.4 Nm", vocab)
    assert any(f["token"] == "22.4" and f["closest"] == "22.5" for f in flagged)


def test_aggregation_counts_are_never_flagged():
    """Summaries legitimately compose small integers the source never writes
    ("lists 30 instances", "spans 4 bays") — bare short integers are exempt."""
    vocab = extract_vocab(ENG_SOURCE)
    assert check_output("lists 30 instances across 4 bays and 12 units", vocab) == []


def test_exact_precision_tokens_are_never_flagged():
    vocab = extract_vocab(ENG_SOURCE)
    ok = ("Per IEC-61850-7-4, the SCADA gateway maps DNP3 registers from "
          "0x1A2B with M8x1.25 fasteners torqued to 22.5 Nm, firmware v2.14.3, "
          "see 7.15.")
    assert check_output(ok, vocab) == []


# ── edit distance ───────────────────────────────────────────────────────────


def test_edit_distance_basics():
    assert edit_distance("DCRRectifierControl_PH", "DCRectifierControl_PH") == 1
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("abc", "xyz") == 3


# ── walking pipeline output ─────────────────────────────────────────────────


def _doc():
    return {
        "doc_summary": CORRUPT_SUMMARY,
        "chunks": [{"chunk_id": "p1_c1", "content": "DCRectifierControl03 in Bay 4.",
                    "page": 1}],
        "tables": [{"html": "<table><tr><td>DCRectifierControl_PH</td></tr></table>",
                    "flat_text": "DCRectifierControl_PH", "pages": [1]}],
        "figures": [{"caption": "Wiring of DCRectifeirControl_PH", "page": 2}],
    }


def test_iter_output_texts_walks_summary_chunks_tables_figures():
    fields = dict(iter_output_texts(_doc()))
    assert any("doc_summary" in k for k in fields)
    assert any("chunk" in k for k in fields)
    assert any("table" in k for k in fields)
    assert any("figure" in k for k in fields)


def test_check_document_report_shape():
    from app.fidelity import check_document
    report = check_document(_doc(), SOURCE_TEXT)
    tokens = {f["token"] for f in report["flagged"]}
    # summary corruption caught...
    assert "DCRRectifierControl_PH" in tokens
    # ...and the transposed figure-caption corruption too, with location.
    assert "DCRectifeirControl_PH" in tokens
    where = {f["token"]: f["where"] for f in report["flagged"]}
    assert "figure" in where["DCRectifeirControl_PH"]
    assert report["vocab_size"] > 0
    assert report["checked"] >= 4


def test_check_document_none_source_returns_none():
    from app.fidelity import check_document
    assert check_document(_doc(), None) is None
    assert check_document(_doc(), "   ") is None


# ── pipeline integration ────────────────────────────────────────────────────


def _make_text_pdf(path, lines):
    """Minimal born-digital 1-page PDF with a real text layer (no deps)."""
    text_ops = "BT /F1 12 Tf 72 720 Td 14 TL\n"
    for ln in lines:
        text_ops += f"({ln}) Tj T*\n"
    text_ops += "ET"
    stream = text_ops.encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF").encode()
    path.write_bytes(out)


def test_extract_pdf_text_reads_text_layer(tmp_path):
    from app.rasterizer import extract_pdf_text
    pdf = tmp_path / "src.pdf"
    _make_text_pdf(pdf, ["Table 95 - DCR instance mapping",
                         "DCRectifierControl03 DCRectifierControl_PH Bay 4"])
    text = extract_pdf_text(str(pdf))
    assert "DCRectifierControl_PH" in text
    assert "DCRectifierControl03" in text


def test_pipeline_attaches_fidelity_report_for_pdf(tmp_path):
    """End-to-end: a born-digital PDF whose model summary contains the
    reproduced corruption -> processing_log.fidelity flags it, with the
    correct source spelling as the closest match."""
    corrupt_page = json.dumps({
        "metadata": {"title": None, "authors": [], "organization": None,
                     "year": None, "doc_type": "report"},
        "summary": CORRUPT_SUMMARY,
        "semantic_chunks": [{"chunk_id": "1", "title": "Instances",
                             "content": "DCRectifierControl03 is in Bay 4.",
                             "keywords": [], "section_type": "technical_content"}],
        "figures": [], "tables": [],
    })
    pdf = tmp_path / "doc.pdf"
    _make_text_pdf(pdf, ["7.15 Control Object Instances",
                         "Table 95 - DCR instance mapping (PH substation)",
                         "DCRectifierControl03 DCRectifierControl_PH Bay 4",
                         "DCRectifierStatus01 DCRectifierStatus_PH Bay 2",
                         "ACLineSegmentCtl02 ACLineSegmentCtl_PH Bay 3"])

    from app.structured_pipeline import run_structured_pipeline

    class OneReplyClient:
        model = "shrew-9b"
        def __init__(self):
            self.calls = []
        def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                            timeout=None, extra_params=None):
            self.calls.append(messages)
            return {"choices": [{"finish_reason": "stop",
                                 "message": {"content": corrupt_page}}]}

    from app.models import PipelineConfig
    cfg = PipelineConfig(vlm_url="http://unused", vlm_model="shrew-9b")
    result = run_structured_pipeline(
        str(pdf), str(tmp_path / "out"), cfg, client=OneReplyClient())

    fid = result.processing_log.get("fidelity")
    assert fid is not None, "born-digital PDF must produce a fidelity report"
    tokens = {f["token"]: f for f in fid["flagged"]}
    assert "DCRRectifierControl_PH" in tokens
    assert tokens["DCRRectifierControl_PH"]["closest"] == "DCRectifierControl_PH"
    assert "doc_summary" in tokens["DCRRectifierControl_PH"]["where"]
    # The faithful chunk content flags nothing extra.
    assert "DCRectifierControl03" not in tokens


def test_pipeline_omits_fidelity_for_image_upload(tmp_path):
    """No deterministic source -> no fidelity key (unavailable != all clear)."""
    from PIL import Image as PILImage
    from app.models import PipelineConfig
    from app.structured_pipeline import run_structured_pipeline

    png = tmp_path / "doc.png"
    PILImage.new("RGB", (1700, 2200), "white").save(png)
    ok_page = json.dumps({
        "metadata": {"title": None, "authors": [], "organization": None,
                     "year": None, "doc_type": None},
        "summary": "s", "semantic_chunks": [], "figures": [], "tables": [],
    })

    class C:
        model = "shrew-9b"
        def chat_completion(self, *a, **k):
            return {"choices": [{"finish_reason": "stop",
                                 "message": {"content": ok_page}}]}

    cfg = PipelineConfig(vlm_url="http://unused", vlm_model="shrew-9b")
    result = run_structured_pipeline(str(png), str(tmp_path / "out"), cfg, client=C())
    assert "fidelity" not in result.processing_log


def test_fidelity_env_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("SHREW_FIDELITY", "0")
    from app.structured_pipeline import _fidelity_check
    assert _fidelity_check({}, "/x.pdf", str(tmp_path), "pdf") is None


def test_identifier_stem_of_a_source_token_is_not_flagged():
    """Live probe finding (d200_p9 pp=0.3 clean run): summaries legitimately
    refer to identifier FAMILIES by stem — "the DCRectifierStatus instances" —
    when the source only ever writes suffixed forms (DCRectifierStatus01,
    DCRectifierStatus_PH). A token that is a substring of a source token is
    backed by the source, not corruption."""
    vocab = extract_vocab(SOURCE_TEXT)
    out = ("The table lists DCRectifierStatus and DCRectifierControl "
           "instances plus ACLineSegmentCtl entries.")
    assert check_output(out, vocab) == []


def test_corruption_is_still_flagged_despite_stem_exemption():
    vocab = extract_vocab(SOURCE_TEXT)
    # The corrupt form is a substring of NOTHING in the source.
    flagged = check_output("the DCRRectifierStatus family", vocab)
    assert flagged and flagged[0]["token"] == "DCRRectifierStatus"


# ── deterministic correction ────────────────────────────────────────────────


def test_correction_rewrites_unique_ident_near_miss():
    from app.fidelity import check_document, apply_corrections
    doc = {"doc_summary": CORRUPT_SUMMARY, "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, SOURCE_TEXT)
    n = apply_corrections(doc, report)
    assert n == 2
    assert "DCRectifierControl_PH" in doc["doc_summary"]
    assert "DCRRectifierControl_PH" not in doc["doc_summary"]
    assert "DCRRectifierStatus_PH" not in doc["doc_summary"]
    by_tok = {f["token"]: f for f in report["flagged"]}
    assert by_tok["DCRRectifierControl_PH"]["corrected"] is True


def test_correction_covers_table_html_and_captions():
    """Transcription near-misses in table cells get the same evidence-backed
    repair — this is the user's original story class."""
    from app.fidelity import check_document, apply_corrections
    doc = {"doc_summary": "s",
           "chunks": [{"chunk_id": "1", "page": 1,
                       "content": "See DCRrectifierControl_PH config."}],
           "tables": [{"pages": [1],
                       "html": "<table><tr><td>DCRrectifierControl_PH</td></tr></table>",
                       "flat_text": "DCRrectifierControl_PH"}],
           "figures": [{"caption": "DCRrectifierControl_PH wiring", "page": 1}]}
    report = check_document(doc, SOURCE_TEXT)
    apply_corrections(doc, report)
    assert "DCRectifierControl_PH" in doc["tables"][0]["html"]
    assert "DCRectifierControl_PH" in doc["tables"][0]["flat_text"]
    assert "DCRectifierControl_PH" in doc["figures"][0]["caption"]
    assert "DCRectifierControl_PH" in doc["chunks"][0]["content"]
    assert "DCRrectifier" not in json.dumps(doc)


def test_correction_skips_ambiguous_matches():
    """Two source identifiers equidistant from the corrupt form -> no
    substitution (we cannot know which was meant); the flag stands."""
    from app.fidelity import check_document, apply_corrections
    src = "PumpCtrlUnit_A1 PumpCtrlUnit_B1 wiring diagram"
    doc = {"doc_summary": "configure PumpCtrlUnit_C1 first",
           "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, src)
    n = apply_corrections(doc, report)
    assert n == 0
    assert "PumpCtrlUnit_C1" in doc["doc_summary"]
    f = report["flagged"][0]
    assert f["token"] == "PumpCtrlUnit_C1" and f.get("corrected") is False


def test_correction_respects_word_boundaries():
    """Substitution must not fire inside a longer token."""
    from app.fidelity import check_document, apply_corrections
    doc = {"doc_summary": "XDCRrectifierControl_PHY stays; DCRrectifierControl_PH goes",
           "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, SOURCE_TEXT)
    apply_corrections(doc, report)
    assert "XDCRrectifierControl_PHY" in doc["doc_summary"]
    assert "; DCRectifierControl_PH goes" in doc["doc_summary"]


def test_correction_env_kill_switch(monkeypatch):
    from app.fidelity import check_document, apply_corrections
    monkeypatch.setenv("SHREW_FIDELITY_CORRECT", "0")
    doc = {"doc_summary": CORRUPT_SUMMARY, "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, SOURCE_TEXT)
    n = apply_corrections(doc, report)
    assert n == 0
    assert "DCRRectifierControl_PH" in doc["doc_summary"]


def test_pipeline_corrects_before_rendering(tmp_path):
    """End-to-end: the shipped markdown and structured JSON carry the
    CORRECTED spelling; the report records what was rewritten."""
    corrupt_page = json.dumps({
        "metadata": {"title": None, "authors": [], "organization": None,
                     "year": None, "doc_type": "report"},
        "summary": CORRUPT_SUMMARY,
        "semantic_chunks": [{"chunk_id": "1", "title": "Instances",
                             "content": "Set DCRrectifierControl_PH per spec.",
                             "keywords": [], "section_type": "technical_content"}],
        "figures": [], "tables": [],
    })
    pdf = tmp_path / "doc.pdf"
    _make_text_pdf(pdf, ["7.15 Control Object Instances",
                         "DCRectifierControl03 DCRectifierControl_PH Bay 4",
                         "DCRectifierStatus01 DCRectifierStatus_PH Bay 2",
                         "ACLineSegmentCtl02 ACLineSegmentCtl_PH Bay 3"])

    from app.models import PipelineConfig
    from app.structured_pipeline import run_structured_pipeline

    class C:
        model = "shrew-9b"
        def chat_completion(self, *a, **k):
            return {"choices": [{"finish_reason": "stop",
                                 "message": {"content": corrupt_page}}]}

    cfg = PipelineConfig(vlm_url="http://unused", vlm_model="shrew-9b")
    result = run_structured_pipeline(str(pdf), str(tmp_path / "out"), cfg, client=C())

    assert "DCRrectifierControl_PH" not in result.clean_markdown
    assert "DCRectifierControl_PH" in result.clean_markdown
    blob = json.dumps(result.structured_json)
    assert "DCRrectifierControl_PH" not in blob and "DCRRectifier" not in blob
    fid = result.processing_log["fidelity"]
    corrected = [f for f in fid["flagged"] if f.get("corrected")]
    assert corrected, "report must record the applied corrections"


# ── widened correction: same evidence logic, per-class risk gating ──────────


def test_structured_code_corruption_is_corrected():
    """Versions, hex addresses, part codes, standard refs: letters+digits
    give them identifier-grade specificity, so a unique near-miss is
    evidence-backed and correctable."""
    from app.fidelity import apply_corrections, check_document
    doc = {"doc_summary": ("requires v2.4.3 at base 0x1A2F with M8x1.5 "
                           "fasteners per IEC-61850-7-5"),
           "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, ENG_SOURCE)
    n = apply_corrections(doc, report)
    assert n == 4
    s = doc["doc_summary"]
    assert "v2.14.3" in s and "0x1A2B" in s
    assert "M8x1.25" in s and "IEC-61850-7-4" in s


def test_bare_numeric_values_are_never_corrected():
    """22.4 near 22.5 might be a legitimately different value — rewriting
    numbers is the one unforgivable failure. Flag, never touch."""
    from app.fidelity import apply_corrections, check_document
    doc = {"doc_summary": "torque to 22.4 Nm, see section 7.5",
           "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, ENG_SOURCE)
    n = apply_corrections(doc, report)
    assert n == 0
    assert "22.4" in doc["doc_summary"] and "7.5" in doc["doc_summary"]
    assert {f["token"] for f in report["flagged"]} >= {"22.4", "7.5"}


def test_pure_acronyms_are_never_corrected():
    """Source says HTTP, model says HTTPS: distance 1, and possibly the model
    CORRECTLY adding knowledge — "fixing" it would introduce the error.
    Pure-alpha acronyms stay flag-only."""
    from app.fidelity import apply_corrections, check_document
    doc = {"doc_summary": "per SCDA guidance", "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, ENG_SOURCE)
    assert apply_corrections(doc, report) == 0
    assert "SCDA" in doc["doc_summary"]
    assert report["flagged"][0]["corrected"] is False


def test_case_corruption_is_corrected_to_source_casing():
    """Case flips blow past the edit-distance budget, but a UNIQUE
    case-insensitive match is the safest correction of all."""
    from app.fidelity import apply_corrections, check_document
    doc = {"doc_summary": ("set DCRECTIFIERCONTROL_PH then verify "
                           "dcrectifierstatus_ph output"),
           "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, SOURCE_TEXT)
    n = apply_corrections(doc, report)
    assert n == 2
    assert "DCRectifierControl_PH" in doc["doc_summary"]
    assert "DCRectifierStatus_PH" in doc["doc_summary"]
    assert "DCRECTIFIER" not in doc["doc_summary"]


def test_truncated_identifier_is_flagged_and_corrected():
    """DCRectifierControl_P ends mid-segment — that's truncation, not a stem;
    the old blanket substring exemption wrongly swallowed it."""
    from app.fidelity import apply_corrections, check_document
    doc = {"doc_summary": "wire DCRectifierControl_P to bay 4",
           "chunks": [], "tables": [], "figures": []}
    report = check_document(doc, SOURCE_TEXT)
    assert [f["token"] for f in report["flagged"]] == ["DCRectifierControl_P"]
    n = apply_corrections(doc, report)
    assert n == 1
    assert "DCRectifierControl_PH to bay 4" in doc["doc_summary"]


def test_segment_aligned_stems_remain_exempt():
    """Whole-segment stems are legit family references: DCRectifierStatus
    (of DCRectifierStatus01), RectifierControl (inner segments), v2.14
    (of v2.14.3). None flag."""
    vocab = extract_vocab(SOURCE_TEXT) | extract_vocab(ENG_SOURCE)
    out = ("the DCRectifierStatus family and RectifierControl group need "
           "firmware v2.14")
    assert check_output(out, vocab) == []


def test_ambiguous_code_tie_is_not_corrected():
    from app.fidelity import apply_corrections, check_document
    src = "registers 0xA0B1 and 0xA0B2 are reserved"
    doc = {"doc_summary": "write 0xA0B3 to enable", "chunks": [], "tables": [],
           "figures": []}
    report = check_document(doc, src)
    assert apply_corrections(doc, report) == 0
    assert "0xA0B3" in doc["doc_summary"]
    f = report["flagged"][0]
    assert f["token"] == "0xA0B3" and f["ambiguous"] is True
