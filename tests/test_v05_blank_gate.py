"""v0.5 near-blank exemption from the coercion-density gate.

Found on the v0.5 OHR-Bench run: on blank pages the first pass hallucinated
until the loop guard aborted it (~3,600 chars), the retry correctly returned
~50 chars, and the density gate (< 300 chars AND < 2% of the first pass)
failed the correct near-empty answer. The exemption accepts a short retry when
the NATIVE page render carries less ink than any real 300-char page
(BLANK_INK_MAX, derived on OHR-Bench + OmniDocBench). Pages with ink keep the
gate: a drawing page whose retry says almost nothing still fails.
"""
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import structured_page as sp
from app.contract import V04, V05
from app.structured_page import BLANK_INK_MAX, DecodePolicy, extract_page, page_ink
from app import structured_pipeline as pl
from app.structured_pipeline import _PAGE_FIELDS, _page_result

V05_POLICY = DecodePolicy.from_env(V05)   # explicit: independent of the shell's SHREW_CONTRACT

HALLUCINATION = "Davis, S. M. Reed, T. A. Smith, M. J. Brown, J. L. {{{ " * 70   # broken, long
BLANK_ANSWER = {
    "metadata": {"title": None, "authors": None, "organization": None, "year": None,
                 "doc_type": None},
    "summary": "Blank page.",
    "semantic_chunks": [], "figures": [], "tables": [],
}


class TwoPassClient:
    """First call: a long unparseable emission; second (retry): `retry_body`."""
    model = "shrew-ocr-preview"

    def __init__(self, retry_body, first_body=HALLUCINATION):
        self.retry_body, self.first_body, self.calls = retry_body, first_body, 0

    def chat_completion_stream(self, messages, on_delta=None, **kw):
        self.calls += 1
        body = self.first_body if self.calls == 1 else self.retry_body
        return {"choices": [{"message": {"content": body}, "finish_reason": "stop"}]}


def _aged_blank(size=(1700, 2200)):
    """Yellowed paper: tint, a vertical shading gradient, dark scan edges, a
    fold line and sparse specks — all background or near it, no text."""
    w, h = size
    rng = np.random.default_rng(0)
    base = np.array([236, 214, 160], np.float32)
    shade = np.linspace(0.92, 1.0, h, dtype=np.float32)[:, None, None]
    a = np.clip(base * shade + rng.normal(0, 2.0, (h, w, 3)), 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 12, h], fill=(150, 120, 80))              # scan edge
    d.line([(0, h // 2), (w, h // 2 + 6)], fill=(205, 182, 130), width=2)   # fold
    for x, y in rng.integers(40, min(w, h) - 40, (25, 2)):        # specks
        d.ellipse([x, y, x + 3, y + 3], fill=(120, 100, 70))
    return im


def _text_page(lines, size=(1700, 2200)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    for i in range(lines):
        d.text((150, 150 + 34 * i), "Revenue grew 4.2% in 2025; demand is expected to remain "
               "stable through the next two quarters.", fill="black")
    return im


def _drawing_page(size=(1700, 2200)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    for i in range(0, 1400, 70):
        d.line([(150 + i, 300), (150 + i, 1900)], fill="black", width=3)
        d.line([(150, 300 + i), (1550, 300 + i)], fill="black", width=3)
    return im


def _save(im, tmp_path, name):
    p = tmp_path / name
    im.save(p)
    return str(p)


def test_page_ink_reads_paper_as_background_and_marks_as_ink():
    assert page_ink(Image.new("RGB", (1700, 2200), "white")) == 0.0
    assert page_ink(_aged_blank()) < BLANK_INK_MAX
    assert page_ink(_drawing_page()) > 3 * BLANK_INK_MAX
    assert page_ink(_text_page(40)) > BLANK_INK_MAX


def test_page_ink_accepts_a_path(tmp_path):
    im = _text_page(10)
    assert page_ink(_save(im, tmp_path, "p.png")) == pytest.approx(page_ink(im))


def test_blank_page_short_retry_is_accepted(tmp_path):
    model = _save(_aged_blank(), tmp_path, "model.png")
    native = _save(_aged_blank(), tmp_path, "hires.png")
    res = extract_page(model, TwoPassClient(json.dumps(BLANK_ANSWER)), ink_source=native, decode=V05_POLICY)
    assert res["ok"] and res["status"] == "ok_coerced"
    nb = res["near_blank"]
    assert nb["ink"] < nb["max"] == BLANK_INK_MAX
    assert nb["content_chars"] == len("Blank page.")


def test_inked_page_short_retry_still_fails(tmp_path):
    """The gate keeps its job: a drawing page whose retry says almost nothing
    after a long first pass is the slop signature."""
    native = _save(_drawing_page(), tmp_path, "hires.png")
    res = extract_page(native, TwoPassClient(json.dumps(BLANK_ANSWER)), ink_source=native, decode=V05_POLICY)
    assert not res["ok"] and res["status"] == "coerced_empty"
    assert "near-blank bound" in res["error"] and "near_blank" not in res


def test_no_ink_source_keeps_the_gate(tmp_path):
    p = _save(_aged_blank(), tmp_path, "model.png")
    res = extract_page(p, TwoPassClient(json.dumps(BLANK_ANSWER)), decode=V05_POLICY)
    assert res["status"] == "coerced_empty" and "page ink" not in res["error"]


def test_unreadable_ink_source_keeps_the_gate(tmp_path):
    p = _save(_aged_blank(), tmp_path, "model.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    calls = []
    orig = sp.page_ink
    sp.page_ink = lambda src: calls.append(src) or orig(src)
    try:
        res = extract_page(p, TwoPassClient(json.dumps(BLANK_ANSWER)), ink_source=str(bad),
                           decode=V05_POLICY)
    finally:
        sp.page_ink = orig
    assert calls == [str(bad)]                       # the measurement was attempted
    assert res["status"] == "coerced_empty" and "page ink" not in res["error"]


def test_v04_contract_is_unchanged(tmp_path):
    """v0.4 keeps 0.4.1 behaviour exactly: no exemption, no measurement."""
    native = _save(_aged_blank(), tmp_path, "hires.png")
    called = []
    orig = sp.page_ink
    sp.page_ink = lambda *a, **k: called.append(1) or orig(*a, **k)
    try:
        res = extract_page(native, TwoPassClient(json.dumps(BLANK_ANSWER)),
                           decode=DecodePolicy.from_env(V04), ink_source=native)
    finally:
        sp.page_ink = orig
    assert res["status"] == "coerced_empty" and not called


def test_ink_is_measured_only_when_the_gate_would_fire(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sp, "page_ink", lambda src: calls.append(src) or 0.0)
    native = _save(_aged_blank(), tmp_path, "hires.png")
    rich = dict(BLANK_ANSWER, summary="A real page. " * 40)
    res = extract_page(native, TwoPassClient(json.dumps(rich)), ink_source=native, decode=V05_POLICY)
    assert res["status"] == "ok_coerced" and "near_blank" not in res and not calls


def test_page_record_carries_near_blank(tmp_path):
    native = _save(_aged_blank(), tmp_path, "hires.png")
    res = extract_page(native, TwoPassClient(json.dumps(BLANK_ANSWER)),
                       decode=DecodePolicy.from_env(V05), ink_source=native)
    rec = _page_result(3, res, "B2")
    assert "near_blank" in _PAGE_FIELDS and rec["near_blank"] == res["near_blank"]
    plain = _page_result(4, {"ok": True, "data": {}, "status": "ok"}, "B1")
    assert plain["near_blank"] is None


def test_bound_sits_between_blank_paper_and_real_text():
    """Derivation record (OHR-Bench 8,561 pages + OmniDocBench 1,651 pages,
    native renders): lowest ink of any page with >= 300 ground-truth chars was
    0.00484; visually blank pages reached 0.0044; the three blank pages the
    v0.5 OHR run failed measured 0.0016, 0.0013 and 0.0004; the drawing page it
    failed measured 0.082."""
    assert 0.0044 < BLANK_INK_MAX <= 0.00484
    assert max(0.0016, 0.0013, 0.0004) < BLANK_INK_MAX < 0.082


def test_ink_at_the_bound_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "page_ink", lambda src: BLANK_INK_MAX)
    native = _save(_aged_blank(), tmp_path, "hires.png")
    res = extract_page(native, TwoPassClient(json.dumps(BLANK_ANSWER)), ink_source=native,
                       decode=V05_POLICY)
    assert res["status"] == "coerced_empty"


def test_real_sequence_loop_abort_then_short_retry(tmp_path, monkeypatch):
    """The OHR sequence: first pass aborted by the loop guard (~3,600 chars of
    invented names), retry stops with ~50 chars."""
    seq = iter([(HALLUCINATION[:3600], sp.REPETITION_ABORT, None),
                (json.dumps(BLANK_ANSWER), "stop", None)])
    monkeypatch.setattr(sp, "_call", lambda *a, **k: next(seq))
    native = _save(_aged_blank(), tmp_path, "hires.png")
    res = extract_page(native, TwoPassClient("unused"), ink_source=native, decode=V05_POLICY)
    assert res["ok"] and res["status"] == "ok_coerced" and res["repetition_abort"]
    assert res["degenerate"] and res["near_blank"]["content_chars"] == len("Blank page.")


def test_page_calls_measure_the_native_render():
    """The bound was derived on native renders; the enhanced model image reads
    paper grain as ink. Pin the call sites."""
    src = inspect.getsource(pl._process_one_page)
    assert "ink_source=hires_path" in src and "ink_source=model_png" not in src
    assert 'ink_source=m["path"]' in inspect.getsource(pl._extract_and_caption_media)


@pytest.mark.parametrize("bad", ["nan", "-0.1", "1.5", "inf"])
def test_bad_bound_stops_import(bad):
    env = dict(os.environ, SHREW_BLANK_INK_MAX=bad)
    r = subprocess.run([sys.executable, "-c", "import app.structured_page"], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "SHREW_BLANK_INK_MAX" in r.stderr
