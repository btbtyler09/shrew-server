"""Coercion-density gate: well-formed slop from the enforcement retry fails.

Found live (VHR run 2026-08-14): a looped broadsheet's enforcement retry
returned valid 5-key JSON carrying 206 chars of hallucinated names against
13,948 chars of real page text. It passed every gate and silently poisoned all
43 of its retrieval queries. A coerced rescue that says almost nothing about a
RENDERED page is not a rescue.
"""
import json

from app.structured_page import (
    COERCED_MIN_CHARS,
    _content_chars,
    extract_page,
    extract_text_page,
)

RICH = {
    "metadata": {"title": "T", "authors": [], "organization": None,
                 "year": None, "doc_type": None},
    "summary": "A page about things.",
    "semantic_chunks": [{"chunk_id": "c1", "title": "Body", "section_type": "technical_content",
                         "content": "Real extracted page text. " * 40}],
    "figures": [], "tables": [],
}
# Varied per-chunk content: the real slop was 17 chunks of DIFFERENT
# hallucinated names — it passed the zlib gate, which is exactly why the
# density gate exists. (An overly repetitive fixture would trip zlib first
# and never exercise this code path.)
SLOP = {
    "metadata": {"title": None, "authors": [], "organization": None,
                 "year": None, "doc_type": None},
    "summary": None,
    "semantic_chunks": [{"chunk_id": f"c{i}", "title": None,
                         "section_type": "technical_content",
                         "content": f"委员{chr(0x4e10 + i * 37)}{chr(0x5f00 + i * 53)}为{chr(0x6210 + i * 41)}"}
                        for i in range(17)],
    "figures": [], "tables": [],
}
BROKEN = "this is not json {{{"


class TwoPassClient:
    """First call fails to parse; second (enforcement) returns `retry_body`."""

    model = "shrew-ocr-preview"

    def __init__(self, retry_body):
        self.retry_body = retry_body
        self.calls = 0

    def chat_completion_stream(self, messages, on_delta=None, **kw):
        self.calls += 1
        body = BROKEN if self.calls == 1 else self.retry_body
        return {"choices": [{"message": {"content": body},
                             "finish_reason": "stop"}]}


def test_content_chars_counts_content_not_scaffolding():
    assert _content_chars(SLOP) < 300 < _content_chars(RICH)


def test_coerced_slop_fails_on_image_pages(tmp_path):
    p = tmp_path / "page.png"; p.write_bytes(b"png")
    res = extract_page(str(p), TwoPassClient(json.dumps(SLOP)))
    assert not res["ok"]
    assert res["status"] == "coerced_empty"
    assert res["schema_coerced"] is True   # visible why it was attempted
    assert "slop" in res["error"]


def test_coerced_rich_content_still_passes(tmp_path):
    p = tmp_path / "page.png"; p.write_bytes(b"png")
    res = extract_page(str(p), TwoPassClient(json.dumps(RICH)))
    assert res["ok"] and res["status"] == "ok_coerced"


def test_first_pass_sparse_page_is_exempt(tmp_path):
    """A poster/cover that says little but parses first-pass is legitimate."""
    class OnePass:
        model = "shrew-ocr-preview"
        def chat_completion_stream(self, messages, on_delta=None, **kw):
            return {"choices": [{"message": {"content": json.dumps(SLOP)},
                                 "finish_reason": "stop"}]}
    p = tmp_path / "page.png"; p.write_bytes(b"png")
    res = extract_page(str(p), OnePass())
    assert res["ok"] and res["status"] == "ok"


def test_text_modality_is_exempt():
    """A tiny text segment legitimately coerces to a tiny result."""
    res = extract_text_page("short note", TwoPassClient(json.dumps(SLOP)))
    assert res["ok"] and res["status"] == "ok_coerced"


def test_default_floor_separates_the_measured_case():
    # slop case: 206 chars; dense pages: 5-15k. The floor must sit between.
    assert 206 < COERCED_MIN_CHARS < 5000
