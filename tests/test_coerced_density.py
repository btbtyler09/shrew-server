"""Coercion-density gate: well-formed slop from the enforcement retry fails.

Found live (VHR run 2026-08-14): a looped broadsheet's enforcement retry
returned valid 5-key JSON carrying 206 chars of hallucinated names against
13,948 chars of real page text. It passed every gate and silently poisoned all
43 of its retrieval queries. A coerced rescue that says almost nothing about a
RENDERED page is not a rescue.

Refined (VHR run 2026-08-15): an absolute char floor alone over-fires on
honest sparse pages — handwriting photos and diagram-only pages fail
first-pass on FORM, not volume, then rescue to proportionate (small) content;
204/206 gate kills had <300 chars in the reference arm too. The gate now
requires the coerced output to also be tiny RELATIVE to the first-pass
emission: slop follows a multi-thousand-char grind, honest sparsity doesn't.
"""
import json

from app.structured_page import (
    COERCED_MIN_CHARS,
    COERCED_MIN_RATIO,
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
# The measured slop case: the first pass GROUND ~14k chars of looped junk
# before the enforcement retry "rescued" it with 206 chars of hallucination.
# The relative test keys on that grind, so the fixture must model it.
BROKEN_LONG = ("Committee member listings: " + "name name name {{{ " * 730)
assert len(BROKEN_LONG) > 13000


class TwoPassClient:
    """First call fails to parse; second (enforcement) returns `retry_body`."""

    model = "shrew-ocr-preview"

    def __init__(self, retry_body, first_body=BROKEN):
        self.retry_body = retry_body
        self.first_body = first_body
        self.calls = 0

    def chat_completion_stream(self, messages, on_delta=None, **kw):
        self.calls += 1
        body = self.first_body if self.calls == 1 else self.retry_body
        return {"choices": [{"message": {"content": body},
                             "finish_reason": "stop"}]}


def test_content_chars_counts_content_not_scaffolding():
    assert _content_chars(SLOP) < 300 < _content_chars(RICH)


def test_coerced_slop_fails_on_image_pages(tmp_path):
    """Dense-page signature: huge first-pass grind, near-empty rescue."""
    p = tmp_path / "page.png"; p.write_bytes(b"png")
    res = extract_page(str(p), TwoPassClient(json.dumps(SLOP), first_body=BROKEN_LONG))
    assert not res["ok"]
    assert res["status"] == "coerced_empty"
    assert res["schema_coerced"] is True   # visible why it was attempted
    assert "slop" in res["error"]


def test_coerced_sparse_page_passes_when_first_pass_was_short(tmp_path):
    """Honest sparsity: a handwriting photo fails first-pass on FORM (short
    malformed emission), then rescues to proportionate small content. The
    2026-08-15 VHR audit found 204/206 absolute-floor kills were this case."""
    p = tmp_path / "page.png"; p.write_bytes(b"png")
    res = extract_page(str(p), TwoPassClient(json.dumps(SLOP), first_body=BROKEN))
    assert res["ok"] and res["status"] == "ok_coerced"


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
    # measured slop: 206 rescued chars against a 13,948-char first-pass grind
    # must stay under the ratio; a 150-char rescue after a ~300-char first
    # pass (GNHK handwriting) must stay over it.
    assert 206 < COERCED_MIN_RATIO * 13948
    assert 150 > COERCED_MIN_RATIO * 300
