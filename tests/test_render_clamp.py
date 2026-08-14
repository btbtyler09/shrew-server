"""Rasterizer render clamp + streaming wall-clock cap (VHR-run hardening).

Both failure modes were found live on the benchmark corpus: poster-sized pages
whose 200-DPI render trips PIL's decompression-bomb guard before the bucket
transform can run, and manuscript pages grinding to the 20k-token cap for hours
because `timeout` only bounds byte gaps on a streaming response.
"""
import json
import time

from app.rasterizer import RENDER_MAX_LONG_EDGE, _clamped_scale
from app.structured_page import (
    STREAM_WALL_CLOCK_S,
    WALL_CLOCK_ABORT,
    _extract,
    build_text_messages,
)


class FakePage:
    def __init__(self, w_pts, h_pts):
        self._size = (w_pts, h_pts)

    def get_size(self):
        return self._size


def test_normal_page_renders_at_requested_dpi():
    # US letter: 612x792 pts -> 1700x2200 at 200 DPI, nowhere near the clamp.
    page = FakePage(612, 792)
    assert _clamped_scale(page, 200) == 200 / 72


def test_poster_page_clamps_to_max_long_edge():
    # The live failure: ~40x53 inch page -> 297 MP at 200 DPI.
    page = FakePage(40 * 72, 53 * 72)
    s = _clamped_scale(page, 200)
    long_edge = 53 * 72 * s
    assert abs(long_edge - RENDER_MAX_LONG_EDGE) < 1
    # Never below the B3 grid — the routing rule needs room to pick 2304x3072.
    assert long_edge >= 3072


def test_clamp_preserves_bucket_choice():
    """The routing rule is scale-invariant: effective glyph height uses
    min(bw/w, bh/h), so scaling the render scales glyph_px and the fit ratio
    together and the chosen bucket cannot change."""
    from app.preprocess import pick_bucket

    native = (28080, 21600)     # 297 MP render that PIL would refuse
    glyph_native = 40.0
    clamped_scale = RENDER_MAX_LONG_EDGE / max(native)
    clamped = (round(native[0] * clamped_scale), round(native[1] * clamped_scale))
    glyph_clamped = glyph_native * clamped_scale

    assert pick_bucket(native, glyph_native) == pick_bucket(clamped, glyph_clamped)


class SlowStreamClient:
    """Trickles deltas forever (well past any test horizon)."""

    model = "shrew-ocr-preview"

    def chat_completion_stream(self, messages, max_tokens=None, temperature=None,
                               timeout=None, extra_params=None, on_delta=None,
                               wall_clock_s=None):
        start = time.time()
        acc = ""
        # Varied content: the zlib guard must NOT fire — that's the whole point.
        for i in range(10_000):
            if wall_clock_s is not None and time.time() - start > wall_clock_s:
                return {"choices": [{"message": {"content": acc},
                                     "finish_reason": WALL_CLOCK_ABORT}]}
            piece = f"w{i * 7919 % 104729}x "
            acc += piece
            if on_delta is not None and on_delta(piece, acc):
                return {"choices": [{"message": {"content": acc},
                                     "finish_reason": "repetition_abort"}]}
            time.sleep(0.001)
        return {"choices": [{"message": {"content": acc}, "finish_reason": "stop"}]}


def test_wall_clock_abort_fails_page_without_retry(monkeypatch):
    import app.structured_page as sp
    monkeypatch.setattr(sp, "STREAM_WALL_CLOCK_S", 0.05)

    client = SlowStreamClient()
    t0 = time.time()
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    elapsed = time.time() - t0

    assert res["status"] == "wall_clock_abort"
    assert not res["ok"]
    assert res["attempts"] == 1, "greedy would grind identically — no retry"
    # The abort must bound the call: well under one full trickle cycle.
    assert elapsed < 2


def test_wall_clock_default_allows_slowest_legitimate_page():
    # 20k tokens at ~10 tok/s is ~2000s; the default must clear it with margin.
    assert STREAM_WALL_CLOCK_S >= 3000
