"""Fallback rescue: teacher-lineage prompt, glyph-aware render, page upgrade.

The fallback exists for pages the e4 model fails (loops, grinds, truncation).
Its contract: it can only ever UPGRADE a page — any fallback failure leaves the
primary failure record untouched — and every rescue is flagged so downstream
can de-rank output that sits outside the validated distribution.
"""
import json

import pytest
from PIL import Image

from app import fallback as fb
from app.fallback import (
    FALLBACK_SYSTEM,
    _coerce_five_key,
    extract_page_fallback,
    prepare_image_fallback,
)
from app.structured_page import validate_schema

GOOD = {
    "metadata": {"title": "T", "authors": ["A"], "organization": None,
                 "year": "2024", "doc_type": "Technical Report"},
    "summary": "s",
    "semantic_chunks": [{"chunk_id": "c1", "title": "Intro", "content": "body",
                         "keywords": ["a"], "section_type": "introduction"}],
    "figures": [{"bbox": [10, 20, 400, 500], "caption": "Figure 1."}],
    "tables": [],
}


# ── prompt lineage ──────────────────────────────────────────────────────────


def test_prompt_carries_the_current_bbox_convention():
    """The bbox rules are specific and versioned (REGION_CONVENTION.md via
    spec.py fda2962). Assert the load-bearing sentences are present so a stale
    or hand-trimmed prompt fails loudly — same trick make_agent_workflow uses."""
    for marker in [
        "normalized to a 0-1000 grid",
        "NOTES / SOURCE / key / footnote lines",
        "excludes only the caption line",
        "0-4 grid units of padding",
        "no edge may slice a glyph, a tick, or a rule",
    ]:
        assert marker in FALLBACK_SYSTEM, f"bbox rule missing: {marker}"


def test_prompt_asks_for_the_serving_contract_shape():
    for marker in ["five keys", "semantic_chunks", "chunk_id",
                   "section_type", "Leader lines"]:
        assert marker in FALLBACK_SYSTEM


# ── glyph-aware render ──────────────────────────────────────────────────────


def test_high_res_scan_downscales_to_glyph_target():
    img = Image.new("RGB", (3400, 4400), "white")
    out = prepare_image_fallback(img, glyph_px=20.0)
    # scale = 10/20 -> half size
    assert out.size == (1700, 2200)


def test_low_res_page_is_never_upscaled():
    img = Image.new("RGB", (1216, 1700), "white")
    out = prepare_image_fallback(img, glyph_px=6.0)
    assert out.size == (1216, 1700), "interpolation invents no detail"


def test_unmeasurable_glyphs_pass_through_at_native():
    img = Image.new("RGB", (1700, 2200), "white")
    assert prepare_image_fallback(img, None).size == (1700, 2200)


def test_long_edge_cap_applies_after_glyph_scaling():
    img = Image.new("RGB", (9000, 12000), "white")
    out = prepare_image_fallback(img, glyph_px=10.0)  # glyph says keep native
    assert max(out.size) == fb.FALLBACK_MAX_LONG_EDGE


# ── coercion ────────────────────────────────────────────────────────────────


def test_coercion_normalizes_stray_shapes_to_the_serving_schema():
    messy = {
        "metadata": {"title": "T", "authors": "Single Author"},  # str, fields missing
        "summary": "s",
        "semantic_chunks": [
            {"chunk_id": 1, "content": "text", "section_type": "overview"},  # int id, bad enum
            {"content": 42},                                                  # not a str: dropped
        ],
        "figures": [{"bbox": [1, 2], "caption": "half a box"}],               # bad bbox -> null
        "tables": [{"html": "<table></table>"}, {"caption": "no html"}],      # second dropped
    }
    out = _coerce_five_key(messy)
    ok, errs = validate_schema(out)
    assert ok, errs
    assert out["metadata"]["authors"] == ["Single Author"]
    assert out["semantic_chunks"][0]["chunk_id"] == "1"
    assert out["semantic_chunks"][0]["section_type"] == "technical_content"
    assert len(out["semantic_chunks"]) == 1
    assert out["figures"][0]["bbox"] is None
    assert len(out["tables"]) == 1


# ── the rescue call ─────────────────────────────────────────────────────────


class FakeFallbackClient:
    def __init__(self, body, finish_reason="stop"):
        self.body = body
        self.finish_reason = finish_reason
        self.calls = []

    def chat_completion(self, messages, max_tokens=None, temperature=None,
                        timeout=None, **kw):
        self.calls.append(messages)
        return {"choices": [{"message": {"content": self.body},
                             "finish_reason": self.finish_reason}]}


@pytest.fixture
def hires(tmp_path):
    p = tmp_path / "page_hires.png"
    Image.new("RGB", (1700, 2200), "white").save(p)
    return str(p)


def test_successful_rescue_returns_validated_five_key(hires, tmp_path):
    client = FakeFallbackClient(json.dumps(GOOD))
    out = extract_page_fallback(hires, 12.0, client,
                                output_path=str(tmp_path / "fb.png"))
    assert out is not None and validate_schema(out)[0]
    sys_msg = client.calls[0][0]
    assert sys_msg["role"] == "system" and "five keys" in sys_msg["content"]


def test_degenerate_fallback_output_is_discarded(hires, tmp_path):
    client = FakeFallbackClient('{"a": "' + "loop " * 3000 + '"}')
    assert extract_page_fallback(hires, 12.0, client,
                                 output_path=str(tmp_path / "fb.png")) is None


def test_truncated_fallback_output_is_discarded(hires, tmp_path):
    client = FakeFallbackClient('{"metadata": {', finish_reason="length")
    assert extract_page_fallback(hires, 12.0, client,
                                 output_path=str(tmp_path / "fb.png")) is None


def test_fallback_transport_error_returns_none(hires, tmp_path):
    class Boom:
        def chat_completion(self, *a, **k):
            raise ConnectionError("down")
    assert extract_page_fallback(hires, 12.0, Boom(),
                                 output_path=str(tmp_path / "fb.png")) is None


# ── pipeline wiring ─────────────────────────────────────────────────────────


def test_failed_page_is_rescued_and_flagged(tmp_path, monkeypatch):
    from app.models import PipelineConfig
    from app.structured_pipeline import run_structured_pipeline

    doc = tmp_path / "page.png"
    Image.new("RGB", (1700, 2200), "white").save(doc)

    class LoopingPrimary:
        model = "shrew-ocr-preview"
        def chat_completion_stream(self, messages, on_delta=None, **kw):
            body = '{"x": "' + "ab" * 8000
            acc = ""
            for i in range(0, len(body), 100):
                acc += body[i:i + 100]
                if on_delta and on_delta(body[i:i + 100], acc):
                    return {"choices": [{"message": {"content": acc},
                                         "finish_reason": "repetition_abort"}]}
            return {"choices": [{"message": {"content": acc},
                                 "finish_reason": "stop"}]}

    import app.structured_pipeline as sp
    monkeypatch.setattr(sp._fb if hasattr(sp, "_fb") else fb,
                        "make_fallback_client", lambda: FakeFallbackClient(json.dumps(GOOD)))
    monkeypatch.setattr(fb, "make_fallback_client",
                        lambda: FakeFallbackClient(json.dumps(GOOD)))

    cfg = PipelineConfig(vlm_url="http://fake", vlm_model="shrew-ocr-preview")
    res = run_structured_pipeline(str(doc), str(tmp_path / "out"), cfg,
                                  client=LoopingPrimary())
    gates = res.processing_log["gates"]
    assert res.processing_log["failed_pages"] == 0
    assert gates["fallback_pages"] == 1
    assert "T" in json.dumps(res.structured_json)


def test_without_fallback_the_failure_stands(tmp_path, monkeypatch):
    from app.models import PipelineConfig
    from app.structured_pipeline import run_structured_pipeline

    doc = tmp_path / "page.png"
    Image.new("RGB", (1700, 2200), "white").save(doc)

    class LoopingPrimary:
        model = "shrew-ocr-preview"
        def chat_completion_stream(self, messages, on_delta=None, **kw):
            body = '{"x": "' + "ab" * 8000
            acc = ""
            for i in range(0, len(body), 100):
                acc += body[i:i + 100]
                if on_delta and on_delta(body[i:i + 100], acc):
                    return {"choices": [{"message": {"content": acc},
                                         "finish_reason": "repetition_abort"}]}
            return {"choices": [{"message": {"content": acc},
                                 "finish_reason": "stop"}]}

    monkeypatch.setattr(fb, "make_fallback_client", lambda: None)
    cfg = PipelineConfig(vlm_url="http://fake", vlm_model="shrew-ocr-preview")
    res = run_structured_pipeline(str(doc), str(tmp_path / "out"), cfg,
                                  client=LoopingPrimary())
    assert res.processing_log["failed_pages"] == 1
    assert res.processing_log["gates"]["fallback_pages"] == 0
