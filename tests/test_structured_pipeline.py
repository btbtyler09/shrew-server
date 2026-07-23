import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from app.assembly import assemble_document
from app.models import PipelineConfig
from app.pipeline import CancelledException
from app.structured_pipeline import (
    build_structured_json,
    gate_metrics,
    render_raw_text,
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


class _CancelledProgress:
    """A progress reporter that reports the request as already cancelled."""

    def emit(self, percent, message):
        pass

    def is_cancelled(self):
        return True


def test_run_structured_pipeline_aborts_when_cancelled(tmp_path):
    doc_path = tmp_path / "doc.png"
    Image.new("RGB", (1700, 2200), "white").save(doc_path)
    out_dir = tmp_path / "out"

    client = FakeClient([(GOOD_PAGE_JSON, "stop")])
    config = _make_config()

    with pytest.raises(CancelledException):
        run_structured_pipeline(
            str(doc_path), str(out_dir), config,
            progress=_CancelledProgress(), client=client,
        )


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


def test_no_crop_means_no_format_claim(tmp_path):
    """A figure/table with no crop (null bbox, or a text-modality page with no
    render behind it) must not advertise format "png" alongside data: null."""
    doc = assemble_document(
        "d3", "/f.xlsx", "spreadsheet",
        [_page(1,
               figures=[{"bbox": None, "caption": "Sheet: Allowables"}],
               tables=[{"bbox": None, "caption": "T", "html": "<table></table>"}])],
        stitch=False,
    )
    sj = build_structured_json(doc, total_pages=1)
    assert sj["images"][0]["data"] is None and sj["images"][0]["format"] is None
    assert sj["tables"][0]["data"] is None and sj["tables"][0]["format"] is None
    # ...but the caption and page provenance survive.
    assert sj["images"][0]["caption"] == "Sheet: Allowables"
    assert sj["images"][0]["page"] == 1


def test_markdown_does_not_link_images_that_do_not_exist(tmp_path):
    doc = assemble_document(
        "d4", "/f.xlsx", "spreadsheet",
        [_page(1, chunks=[{"chunk_id": "1", "title": "T", "content": "c",
                            "keywords": [], "section_type": "results"}],
               figures=[{"bbox": None, "caption": "Sheet: Allowables"}])],
        stitch=False,
    )
    md = synthesize_markdown(doc)
    assert "![" not in md, "no image ref without a crop to back it"
    assert "[Figure: Sheet: Allowables]" in md


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


def test_synthesize_markdown_groups_units_under_their_page(tmp_path):
    doc = _two_page_doc(tmp_path)
    md = synthesize_markdown(doc)

    # Each page is wrapped in <page N> ... </page N> tags, in order.
    assert md.index("<page 1>") < md.index("</page 1>") < md.index("<page 2>")
    assert md.rstrip().endswith("</page 2>")

    # The figure (page 1) is inline in page 1's block; the table (page 2) in
    # page 2's block — not dumped into trailing sections.
    page1 = md[md.index("<page 1>"):md.index("</page 1>")]
    page2 = md[md.index("<page 2>"):md.index("</page 2>")]
    assert "Intro text." in page1 and "![Fig one](img:1)" in page1
    assert "<table>" not in page1
    assert "Result text." in page2 and "<table>" in page2
    assert "![Fig one]" not in page2
    # No leftover global sections from the old layout.
    assert "## Tables" not in md and "## Figures" not in md


# ── text modality: deterministic extractor -> model text arm ────────────────


def test_csv_routes_through_the_text_modality_not_rasterization(tmp_path):
    """A csv has a deterministic extractor, so it must reach the model as a
    §2 text-modality request (plain-string user content) and never be
    rasterized."""
    doc_path = tmp_path / "data.csv"
    doc_path.write_text("name,qty\nbolt,4\nnut,8\n")

    client = FakeClient([(GOOD_PAGE_JSON, "stop")])
    result = run_structured_pipeline(
        str(doc_path), str(tmp_path / "out"), _make_config(), client=client,
    )

    assert result.processing_log["modality"] == "text"
    user_content = client.calls[0]["messages"][1]["content"]
    assert isinstance(user_content, str), "text arm takes a raw string"
    assert "bolt" in user_content
    # No rasterization happened.
    assert not (tmp_path / "out" / "pages").exists()
    assert result.structured_json["semantic_chunks"]


def test_large_text_input_is_paginated_into_page_sized_requests(tmp_path):
    """The text arm only ever saw page-sized input, so a big extraction is
    split into several requests rather than sent as one giant message."""
    doc_path = tmp_path / "big.md"
    para = "word " * 500  # ~2500 chars
    doc_path.write_text("\n\n".join([para] * 40))  # ~100k chars

    client = FakeClient([(GOOD_PAGE_JSON, "stop")] * 40)
    result = run_structured_pipeline(
        str(doc_path), str(tmp_path / "out"), _make_config(), client=client,
    )

    assert len(client.calls) > 1, "should have paginated"
    from app.structured_page import TEXT_PAGE_MAX_CHARS
    for call in client.calls:
        assert len(call["messages"][1]["content"]) <= TEXT_PAGE_MAX_CHARS
    assert result.processing_log["total_pages"] == len(client.calls)


def test_pdf_class_still_uses_the_image_modality(tmp_path):
    doc_path = tmp_path / "doc.png"
    Image.new("RGB", (1700, 2200), "white").save(doc_path)

    client = FakeClient([(GOOD_PAGE_JSON, "stop")])
    result = run_structured_pipeline(
        str(doc_path), str(tmp_path / "out"), _make_config(), client=client,
    )

    assert result.processing_log["modality"] == "image"
    assert isinstance(client.calls[0]["messages"][1]["content"], list)


# ── raw rendering ───────────────────────────────────────────────────────────


def test_raw_mode_on_image_input_runs_the_model_and_flattens(tmp_path):
    doc_path = tmp_path / "doc.png"
    Image.new("RGB", (1700, 2200), "white").save(doc_path)

    client = FakeClient([(GOOD_PAGE_JSON, "stop")])
    result = run_structured_pipeline(
        str(doc_path), str(tmp_path / "out"), _make_config(), client=client, raw=True,
    )

    assert len(client.calls) == 1, "raw still uses shrew-ocr for image input"
    assert result.structured_json == {}, "stage-3 keys are omitted in raw mode"
    assert "Doc Title" in result.clean_markdown
    assert "Some intro content." in result.clean_markdown


def test_raw_mode_on_markdown_still_uses_the_model(tmp_path):
    """"Messy HTML/markdown/code/plain text" is what the text modality was
    built for (§2), so raw runs these through the model and flattens the
    result rather than echoing the source file back."""
    doc_path = tmp_path / "page.md"
    doc_path.write_text("# scraped\n\nnav nav nav\n\nreal content here")

    client = FakeClient([(GOOD_PAGE_JSON, "stop")])
    result = run_structured_pipeline(
        str(doc_path), str(tmp_path / "out"), _make_config(), client=client, raw=True,
    )

    assert len(client.calls) == 1, "markdown goes through the model in raw mode"
    assert result.processing_log["modality"] == "text"
    assert "Doc Title" in result.clean_markdown


def test_raw_mode_on_spreadsheet_family_skips_the_model(tmp_path):
    """A csv/xlsx/eml already has a lossless deterministic parse; raw returns
    it directly rather than paying the model to re-emit it."""
    doc_path = tmp_path / "data.csv"
    doc_path.write_text("name,qty\nbolt,4\n")

    client = FakeClient([])  # must never be called
    result = run_structured_pipeline(
        str(doc_path), str(tmp_path / "out"), _make_config(), client=client, raw=True,
    )

    assert client.calls == [], "no model call for deterministic raw extraction"
    assert result.processing_log["modality"] == "deterministic"
    assert result.structured_json == {}
    assert "bolt" in result.clean_markdown


def test_render_raw_text_sections_are_fixed_and_ordered(tmp_path):
    from app.structured_pipeline import RAW_SECTIONS
    doc = _two_page_doc(tmp_path)
    doc["metadata"] = {"title": "The Title", "authors": ["Ada", "Bob"],
                        "organization": "ACME", "year": "2019", "doc_type": "report"}
    doc["doc_summary"] = "A short summary."
    out = render_raw_text(doc)

    # Every section present, in the declared order.
    positions = [out.index(f"# {name}") for name in RAW_SECTIONS]
    assert positions == sorted(positions)
    assert out.startswith("# Metadata")

    assert "Title: The Title" in out and "Authors: Ada, Bob" in out
    assert "Organization: ACME" in out and "Year: 2019" in out and "Type: report" in out
    assert "Pages: 2" in out
    assert "A short summary." in out
    # Content is content-only — no chunk-title headings.
    assert "Intro text." in out and "Result text." in out
    assert "## Intro" not in out and "## Results" not in out


def test_render_raw_text_tables_and_figures_keep_their_page(tmp_path):
    doc = _two_page_doc(tmp_path)
    out = render_raw_text(doc)
    # Moving them out of the page blocks must not lose locality.
    assert "## Table 1 (page 2)" in out
    assert "## Figure 1 (page 1) — Fig one" in out
    assert "<table>" not in out, "tables render as flat text, not HTML"
    assert "![" not in out, "no markdown image refs in the raw rendering"


def test_render_raw_text_emits_empty_sections(tmp_path):
    """Sections are unconditional so a consumer can split on /^# / and rely on
    the same headers every time."""
    from app.structured_pipeline import RAW_SECTIONS
    doc = assemble_document(
        "d9", "/f.pdf", "arxiv",
        [_page(1, summary=None, chunks=[])], stitch=False,
    )
    out = render_raw_text(doc)
    for name in RAW_SECTIONS:
        assert f"# {name}" in out
    assert out.count("(none)") >= 3  # summary, content, tables, figures


def test_render_raw_text_survives_missing_metadata(tmp_path):
    doc = _two_page_doc(tmp_path)
    out = render_raw_text(doc)
    assert out.startswith("# Metadata")
    assert "Pages: 2" in out


# ── §3/§5 gate metrics ──────────────────────────────────────────────────────


def test_gate_metrics_counts_each_outcome():
    m = gate_metrics([
        {"ok": True, "status": "ok"},
        {"ok": True, "status": "ok"},
        {"ok": True, "status": "ok_coerced", "schema_coerced": True},
        {"ok": False, "status": "degenerate", "degenerate": True},
        {"ok": False, "status": "oversize"},
    ])
    assert m["pages"] == 5
    assert m["first_pass_ok"] == 2
    assert m["schema_coerced"] == 1
    assert m["degenerate"] == 1
    assert m["oversize_filtered"] == 1
    assert m["first_pass_fail_rate"] == 0.6
    assert m["degeneration_rate"] == 0.2
    # One of the two pages that took the retry tier came back usable.
    assert m["coerce_success_rate"] == 0.5


def test_gate_metrics_coerce_rate_is_none_when_nothing_retried():
    m = gate_metrics([{"ok": True, "status": "ok"}])
    assert m["coerce_success_rate"] is None
    assert m["first_pass_fail_rate"] == 0.0


def test_processing_log_reports_gates(tmp_path):
    doc_path = tmp_path / "doc.png"
    Image.new("RGB", (1700, 2200), "white").save(doc_path)
    client = FakeClient([(GOOD_PAGE_JSON, "stop")])
    result = run_structured_pipeline(
        str(doc_path), str(tmp_path / "out"), _make_config(), client=client,
    )
    assert result.processing_log["gates"]["first_pass_ok"] == 1
    assert result.processing_log["gates"]["first_pass_fail_rate"] == 0.0


def test_raw_sections_split_reliably_even_with_headings_in_content():
    """Model chunk content routinely contains its own '# Title' (title pages do
    this). Those must not collide with the section headers, or splitting the
    document on /^# / silently produces bogus sections."""
    import re
    from app.structured_pipeline import RAW_SECTIONS
    doc = assemble_document(
        "d10", "/f.pdf", "arxiv",
        [_page(1, chunks=[{"chunk_id": "1", "title": "Title Page",
                            "content": "# A Critical Evaluation\n\n## Subhead\n\nbody",
                            "keywords": [], "section_type": "introduction"}])],
        stitch=False,
    )
    out = render_raw_text(doc)

    found = re.findall(r"^# (.+)$", out, re.M)
    assert found == list(RAW_SECTIONS), f"stray h1 leaked into sections: {found}"
    # The heading survives, one level down — content is not dropped.
    assert "## A Critical Evaluation" in out
    assert "### Subhead" in out
    assert "body" in out


# ── model-input geometry ────────────────────────────────────────────────────


def test_pdf_pages_keep_the_exact_200_to_100_transform():
    """PDFs are rasterized by us at config.high_dpi, so the contract's fixed
    200->100 halving is correct and must not be second-guessed."""
    from app.structured_pipeline import _model_input_src_dpi
    cfg = _make_config()
    assert _model_input_src_dpi((1700, 2200), cfg, "pdf") == cfg.high_dpi
    assert _model_input_src_dpi((1700, 2200), cfg, "office") == cfg.high_dpi


def test_uploaded_scan_is_normalized_to_the_trained_scale():
    """A 400-DPI scan arrives at its native size, so the fixed halving would
    hand the model 4x the trained pixel area."""
    from app.structured_pipeline import TRAINED_LONG_EDGE, _model_input_src_dpi
    cfg = _make_config()
    src_dpi = _model_input_src_dpi((3400, 4400), cfg, "image")
    # prepare_image scales by low_dpi/src_dpi.
    scaled_long_edge = round(4400 * cfg.low_dpi / src_dpi)
    assert scaled_long_edge == TRAINED_LONG_EDGE


def test_a_200dpi_image_upload_still_lands_at_850x1100():
    from app.structured_pipeline import _model_input_src_dpi
    cfg = _make_config()
    assert _model_input_src_dpi((1700, 2200), cfg, "image") == 200


def test_small_uploads_are_never_upscaled():
    """A thumbnail has no detail to recover; blowing it up to 1100 just
    invents pixels."""
    from app.structured_pipeline import _model_input_src_dpi
    cfg = _make_config()
    assert _model_input_src_dpi((400, 500), cfg, "image") == cfg.low_dpi


def test_landscape_scan_normalizes_on_the_long_edge():
    from app.structured_pipeline import TRAINED_LONG_EDGE, _model_input_src_dpi
    cfg = _make_config()
    src_dpi = _model_input_src_dpi((4400, 3400), cfg, "image")
    assert round(4400 * cfg.low_dpi / src_dpi) == TRAINED_LONG_EDGE


def test_oversized_scan_reaches_the_model_at_trained_size(tmp_path):
    """End-to-end: the saved model input must be the trained geometry, not the
    scan's native size."""
    from PIL import Image as PILImage
    doc_path = tmp_path / "scan.png"
    PILImage.new("RGB", (3400, 4400), "white").save(doc_path)
    out_dir = tmp_path / "out"

    client = FakeClient([(GOOD_PAGE_JSON, "stop")])
    run_structured_pipeline(str(doc_path), str(out_dir), _make_config(), client=client)

    model_png = next((out_dir / "pages").glob("*_model.png"))
    with PILImage.open(model_png) as im:
        assert max(im.size) == 1100, f"model input was {im.size}"


# ── caption / chunk de-duplication ──────────────────────────────────────────


def _doc_with_caption_placeholder(tmp_path):
    """Mirrors the trained convention: a figure caption emitted BOTH as a
    figures[] entry and as its own semantic_chunk."""
    hires = tmp_path / "p.png"
    Image.new("RGB", (2000, 1000)).save(hires)
    return assemble_document(
        "capdoc", "/f.pdf", "arxiv",
        [_page(1,
               chunks=[
                   {"chunk_id": "1", "title": "Coordinate System", "keywords": [],
                    "section_type": "technical_content",
                    "content": "The axes are given by the principal basis vectors."},
                   {"chunk_id": "2", "title": "Figure 1. Principal basis vectors.",
                    "keywords": [], "section_type": "technical_content",
                    "content": "Figure 1. Principal basis vectors."},
               ],
               figures=[{"bbox": [100, 200, 600, 700],
                         "caption": "Figure 1. Principal basis vectors."}]),
        ],
        hires_images={1: str(hires)},
        crops_dir=str(tmp_path / "crops"),
        stitch=False,
    )


def test_structured_keeps_placeholder_chunk_but_strips_image_caption(tmp_path):
    """semantic_chunks stays eval-faithful (the placeholder chunk survives),
    while the bottom image ref drops the redundant caption."""
    doc = _doc_with_caption_placeholder(tmp_path)

    # The placeholder chunk is still in the structured data.
    sj = build_structured_json(doc, total_pages=1)
    assert any(c["content"] == "Figure 1. Principal basis vectors."
               for c in sj["semantic_chunks"])

    md = synthesize_markdown(doc)
    # The image ref is placed INLINE at the placeholder chunk's reading-order
    # position, carrying the caption as alt text — not dumped at the page end.
    assert "![Figure 1. Principal basis vectors.](img:1)" in md
    # The placeholder chunk's own text is replaced by the image, so the caption
    # appears exactly once (as the alt), and the real prose chunk is untouched.
    assert md.count("Figure 1. Principal basis vectors.") == 1
    assert "The axes are given by the principal basis vectors." in md
    # It lands where the figure belongs — after the real prose chunk that
    # preceded it, inside the page block (not appended after everything).
    assert md.index("The axes are given") < md.index("![Figure 1")


def test_standalone_figure_keeps_its_caption_in_the_ref(tmp_path):
    """When no chunk echoes the caption, the image ref keeps it (so the figure
    is not left with an empty alt)."""
    hires = tmp_path / "p.png"
    Image.new("RGB", (2000, 1000)).save(hires)
    doc = assemble_document(
        "d", "/f.pdf", "arxiv",
        [_page(1, chunks=[{"chunk_id": "1", "title": "Body", "keywords": [],
                           "section_type": "results", "content": "Unrelated prose."}],
               figures=[{"bbox": [100, 200, 600, 700], "caption": "Fig 7. Turbine map."}])],
        hires_images={1: str(hires)}, crops_dir=str(tmp_path / "crops"), stitch=False,
    )
    md = synthesize_markdown(doc)
    assert "![Fig 7. Turbine map.](img:1)" in md


def test_raw_content_drops_caption_placeholder_chunks(tmp_path):
    """In raw, the caption-echo chunk is dropped from Content; the caption is
    still present once, in the Figures section."""
    doc = _doc_with_caption_placeholder(tmp_path)
    out = render_raw_text(doc)

    content = out[out.index("# Content"):out.index("# Tables")]
    assert "principal basis vectors" in content.lower()  # the real prose chunk
    # ...but the caption-echo chunk is not duplicated into Content.
    assert "Figure 1. Principal basis vectors." not in content
    # The caption lives once, in the Figures section.
    figures = out[out.index("# Figures"):]
    assert "Figure 1. Principal basis vectors." in figures


# ── model-refined table stitching ───────────────────────────────────────────


from app.assembly import build_table_composite
from app.structured_pipeline import make_table_refiner


def test_build_table_composite_pads_to_full_page_geometry(tmp_path):
    a = Image.new("RGB", (1700, 2200), "white")
    b = Image.new("RGB", (1700, 2200), "white")
    # fragments: bottom 30% of page 1, top 20% of page 2
    comp = build_table_composite(a, [100, 700, 900, 1000], b, [100, 0, 900, 200])
    assert comp is not None
    # Heavily padded to EXACTLY the source page dims, so the standard 200→100
    # transform lands the model input squarely in the trained distribution.
    assert comp.size == (1700, 2200)
    assert comp.getpixel((2, 2)) == (255, 255, 255)


def test_build_table_composite_never_shrinks_to_fit():
    """Two near-full-page fragments cannot fit a page canvas at native scale;
    shrinking is forbidden (sub-scale text stalls the model — verified live),
    so the builder declines and the caller falls back to row-append."""
    a = Image.new("RGB", (1700, 2200), "white")
    b = Image.new("RGB", (1700, 2200), "white")
    assert build_table_composite(a, [50, 90, 950, 950], b, [50, 90, 950, 1000]) is None


def _stitchable_pages():
    """Two pages whose tables pass should_stitch_tables."""
    frag1 = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    frag2 = "<table><tr><td>3</td><td>4</td></tr></table>"
    return [
        _page(1, tables=[{"bbox": [100, 800, 900, 1000], "caption": "Parts", "html": frag1}]),
        _page(2, tables=[{"bbox": [100, 0, 900, 150], "caption": None, "html": frag2}]),
    ]


REFINED_TABLE_JSON = json.dumps({
    "metadata": {"title": None, "authors": [], "organization": None,
                 "year": None, "doc_type": None},
    "summary": "s", "semantic_chunks": [], "figures": [],
    "tables": [{"bbox": [50, 50, 950, 950], "caption": "Parts",
                "html": ("<table><tr><th>A</th><th>B</th></tr>"
                          "<tr><td>1</td><td>2</td></tr>"
                          "<tr><td>3</td><td>4</td></tr></table>")}],
})


def _refiner_fixture(tmp_path, replies):
    hires = {}
    for p in (1, 2):
        path = tmp_path / f"h{p}.png"
        Image.new("RGB", (1700, 2200), "white").save(path)
        hires[p] = str(path)
    stats = {}
    client = FakeClient(replies)
    refiner = make_table_refiner(hires, str(tmp_path), client, stats)
    return refiner, stats, client


def test_refined_stitch_uses_model_html_and_composite_crop(tmp_path):
    refiner, stats, client = _refiner_fixture(tmp_path, [(REFINED_TABLE_JSON, "stop")])
    doc = assemble_document("d", "/f.pdf", "arxiv", _stitchable_pages(),
                            table_refiner=refiner)
    assert len(doc["tables"]) == 1
    t = doc["tables"][0]
    assert t["pages"] == [1, 2] and t["caption"] == "Parts"
    # Model html, not row-append: one coherent table with 3 rows.
    assert t["html"].count("<tr") == 3
    # The hires composite became the merged table's crop.
    assert t["crop_path"] and t["crop_path"].endswith("_stitched.png")
    assert Path(t["crop_path"]).exists()
    assert stats == {"attempted": 1, "model_refined": 1}
    # The model call went through the standard image modality (image part).
    assert isinstance(client.calls[0]["messages"][1]["content"], list)


def test_failed_refinement_falls_back_to_rowappend(tmp_path):
    refiner, stats, _ = _refiner_fixture(
        tmp_path, [("not json", "stop"), ("still not json", "stop")])
    doc = assemble_document("d", "/f.pdf", "arxiv", _stitchable_pages(),
                            table_refiner=refiner)
    assert len(doc["tables"]) == 1
    t = doc["tables"][0]
    # Stitched anyway — deterministically.
    assert t["pages"] == [1, 2]
    assert "<td>3</td>" in t["html"] and "<td>1</td>" in t["html"]
    assert not t.get("crop_path")
    assert stats == {"attempted": 1}


def test_refinement_that_drops_rows_is_rejected(tmp_path):
    dropped = json.dumps({
        "metadata": {"title": None, "authors": [], "organization": None,
                     "year": None, "doc_type": None},
        "summary": "s", "semantic_chunks": [], "figures": [],
        "tables": [{"bbox": None, "caption": None,
                    "html": "<table><tr><td>only</td></tr></table>"}],
    })
    refiner, stats, _ = _refiner_fixture(tmp_path, [(dropped, "stop")])
    doc = assemble_document("d", "/f.pdf", "arxiv", _stitchable_pages(),
                            table_refiner=refiner)
    t = doc["tables"][0]
    # 1-row re-extraction < 2-row fragment -> fallback row-append (3 rows).
    assert t["html"].count("<tr") == 3
    assert stats == {"attempted": 1}


def test_oversize_composite_skips_model_call(tmp_path):
    # Full-page fragments on both sides: composite ~4400px tall > 2400 cap.
    frag = "<table><tr><td>x</td><td>y</td></tr></table>"
    pages = [
        _page(1, tables=[{"bbox": [0, 0, 1000, 1000], "caption": None, "html": frag}]),
        _page(2, tables=[{"bbox": [0, 0, 1000, 100], "caption": None, "html": frag}]),
    ]
    # page-1 table spans the whole page (y0=0,y1=1000) and page-2 starts at top
    refiner, stats, client = _refiner_fixture(tmp_path, [])
    doc = assemble_document("d", "/f.pdf", "arxiv", pages, table_refiner=refiner)
    assert client.calls == [], "no model call for an oversize composite"
    assert len(doc["tables"]) == 1 and doc["tables"][0]["pages"] == [1, 2]
    assert stats == {"attempted": 1}
