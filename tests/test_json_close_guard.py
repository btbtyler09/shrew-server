"""Stop-at-JSON-close guard (serving path only).

Belt and braces against "close the top-level object, then keep generating until
the cap". The case is theoretical — unobserved in the paired eval — so the
guard's value is that it is cheap, visible when it fires (its own finish
reason, its own counter) and inert on every other page. Loops happen INSIDE
the JSON before it ever closes; those stay the repetition guard's job and
these tests pin that the two never trade places.
"""
import json
import random

from app.structured_page import (
    JSON_CLOSE,
    REPETITION_ABORT,
    JsonCloseGuard,
    RepetitionGuard,
    StreamGuards,
    _call,
    _gate,
)


def _feed(guard, text, chunk=7):
    """Drive a guard the way chat_completion_stream does: returns (reason, acc)
    for the chunk that fired, or (None, full text)."""
    acc = ""
    for i in range(0, len(text), chunk):
        piece = text[i:i + chunk]
        acc += piece
        reason = guard(piece, acc)
        if reason:
            return reason, acc
    return None, acc


DOC = json.dumps({"metadata": {"title": "a } b", "authors": ["x"], "organization": "o",
                               "year": "2020", "doc_type": "report"},
                  "summary": "x",
                  "semantic_chunks": [{"chunk_id": "c1", "title": "t", "section_type": "abstract",
                                       "content": 'text with "quotes" and {braces}'}],
                  "tables": [], "figures": []})


# ── 1. closed doc + trailing junk → stops at the brace, object intact ───────

def test_fires_at_top_level_close_and_trims_trailing_junk():
    for chunk in (1, 3, 7, 64, 10_000):   # boundary-independent
        guard = JsonCloseGuard()
        reason, acc = _feed(guard, DOC + " trailing junk }}}{{{ " * 20, chunk=chunk)
        assert reason == JSON_CLOSE
        assert guard.fired and guard.closed and guard.close_index == len(DOC)
        assert json.loads(acc[:guard.close_index]) == json.loads(DOC)
        assert guard.trailing == len(acc) - len(DOC) > 0


def test_clean_close_with_eot_does_not_fire_and_keeps_stop():
    # The model closes and stops on its own: the close is RECORDED, nothing
    # fires, the natural finish reason survives. Whitespace after the brace is
    # not "kept generating".
    for tail in ("", "\n", "  \n\n"):
        guard = JsonCloseGuard()
        reason, acc = _feed(guard, DOC + tail, chunk=7)
        assert reason is None and not guard.fired
        assert guard.closed and guard.close_index == len(DOC)
    assert guard.stats() == {"fired": False, "closed": True, "close_chars": len(DOC),
                             "trailing_chars_dropped": 0}


def test_gate_accepts_a_json_close_completion_as_ok():
    parsed, verdict, err = _gate(DOC, JSON_CLOSE, None)
    assert verdict == "ok" and parsed["summary"] == "x", err


# ── 2. a `}` inside a string value does NOT fire ─────────────────────────────

def test_brace_inside_string_does_not_fire():
    guard = JsonCloseGuard()
    text = '{"a": "close } here", "b": "and \\" escaped quote } too", "c": {"d": "}"}}'
    reason, _ = _feed(guard, text[:-1], chunk=5)     # everything but the real close
    assert reason is None and not guard.fired
    assert guard.depth == 1 and not guard.closed
    assert guard("}", text) is None                     # the real close: recorded, not fired
    assert guard.closed and guard.close_index == len(text)
    assert guard("x", text + "x") == JSON_CLOSE         # ...and only content after it fires


def test_escaped_backslash_before_quote_ends_the_string():
    # "...\\" is an escaped backslash followed by the closing quote: the string
    # ENDS there, so the following } is structural.
    guard = JsonCloseGuard()
    text = '{"a": "path\\\\"}'
    reason, _ = _feed(guard, text + " junk", chunk=2)
    assert reason == JSON_CLOSE and guard.close_index == len(text)


def test_prefix_before_first_brace_is_ignored():
    guard = JsonCloseGuard()
    text = "```json\n" + DOC + "\n```"
    reason, acc = _feed(guard, text, chunk=9)
    assert reason == JSON_CLOSE                         # the closing fence is content after the close
    assert acc[:guard.close_index].endswith(DOC)


def test_never_fires_without_a_top_level_object():
    guard = JsonCloseGuard()
    reason, _ = _feed(guard, '["a", "b", {"c": 1}] and more', chunk=4)
    # A top-level array is not the contract. The first opener is `[`, so the
    # guard stays inert and the parse gate owns the page — it must NOT stop at
    # the inner object's close and hand back a fragment as if it were the doc.
    assert reason is None and not guard.fired and not guard.closed


# ── 3. a never-closing loop → the repetition guard catches it, unchanged ─────

def test_loop_is_still_the_repetition_guards_call():
    loop = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 4000
    guards = StreamGuards(RepetitionGuard(), JsonCloseGuard())
    reason, acc = _feed(guards, loop, chunk=64)
    assert reason == REPETITION_ABORT
    assert guards.repetition.fired and not guards.json_close.fired
    assert len(acc) < len(loop) / 4
    # And the gate verdict is exactly what it was before the new guard existed.
    parsed, verdict, err = _gate(acc, reason, guards.repetition)
    assert parsed is None and verdict == "degenerate" and "repetition_abort" in err
    stats = guards.stats()
    assert stats["aborted"] and stats["json_close"]["fired"] is False


def test_clean_long_page_fires_neither_guard_until_the_close():
    rng = random.Random(1)
    chunks = [{"chunk_id": f"c{i}", "content": " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 11)))
        for _ in range(60))} for i in range(120)]
    doc = json.dumps({"metadata": {}, "summary": "s", "semantic_chunks": chunks,
                      "tables": [], "figures": []})
    guards = StreamGuards(RepetitionGuard(), JsonCloseGuard())
    reason, acc = _feed(guards, doc, chunk=64)
    assert reason is None and len(acc) == len(doc)
    assert guards.json_close.closed and not guards.json_close.fired
    assert not guards.repetition.fired


# ── the serving path: _call trims at the close and keeps the reason ──────────

class _FakeStreamClient:
    """Streams `text` in chunks through on_delta exactly like VLMClient."""
    model = "shrew-ocr-preview"

    def __init__(self, text, chunk=11):
        self.text, self.chunk = text, chunk

    def chat_completion_stream(self, messages, *, max_tokens, temperature,
                               extra_params, timeout, on_delta, wall_clock_s):
        acc, reason = "", None
        for i in range(0, len(self.text), self.chunk):
            piece = self.text[i:i + self.chunk]
            acc += piece
            reason = on_delta(piece, acc)
            if reason:
                break
        return {"choices": [{"message": {"role": "assistant", "content": acc},
                             "finish_reason": reason or "stop"}]}


def test_call_returns_trimmed_text_and_json_close_reason(monkeypatch):
    import app.structured_page as sp
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", True)
    monkeypatch.setattr(sp, "JSON_CLOSE_GUARD_ENABLED", True)
    client = _FakeStreamClient(DOC + "\n\n{ garbage after the close" * 50)
    text, reason, guard = _call([], client, max_tokens=100, temperature=0,
                                extra_params=None, timeout=5, page_no=3, attempt="t")
    assert reason == JSON_CLOSE
    assert text == DOC                      # trailing junk gone
    assert guard.stats()["json_close"] == {"fired": True, "closed": True, "close_chars": len(DOC),
                                           "trailing_chars_dropped": guard.json_close.trailing}
    assert guard.json_close.trailing > 0


def test_call_on_a_clean_stop_keeps_stop_and_the_whole_text(monkeypatch):
    import app.structured_page as sp
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", True)
    monkeypatch.setattr(sp, "JSON_CLOSE_GUARD_ENABLED", True)
    client = _FakeStreamClient(DOC + "\n")
    text, reason, guard = _call([], client, max_tokens=100, temperature=0,
                                extra_params=None, timeout=5)
    assert reason == "stop" and text == DOC + "\n"
    assert guard.stats()["json_close"]["fired"] is False and guard.stats()["json_close"]["closed"] is True


def test_call_with_guard_disabled_is_the_old_behaviour(monkeypatch):
    import app.structured_page as sp
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", True)
    monkeypatch.setattr(sp, "JSON_CLOSE_GUARD_ENABLED", False)
    junk = DOC + " tail"
    client = _FakeStreamClient(junk)
    text, reason, guard = _call([], client, max_tokens=100, temperature=0,
                                extra_params=None, timeout=5)
    assert reason == "stop" and text == junk
    # (by name: another test module reloads structured_page, which would make
    # an isinstance check order-dependent)
    assert type(guard).__name__ == "RepetitionGuard" and not hasattr(guard, "json_close")


# ── the per-document counter exists even when it is zero ────────────────────

def test_gate_metrics_counts_json_close_stops():
    from app.structured_pipeline import _page_result, gate_metrics
    def res(fired):
        return {"ok": True, "data": {}, "status": "ok",
                "loop_guard": {"aborted": False, "json_close": {"fired": fired}}}
    fired = _page_result(1, res(True)); quiet = _page_result(2, res(False))
    legacy = _page_result(3, {"ok": True, "data": {}, "status": "ok", "loop_guard": None})
    assert fired["json_close"] is True and quiet["json_close"] is False and legacy["json_close"] is False
    assert gate_metrics([fired, quiet, legacy])["json_close_stops"] == 1
    assert gate_metrics([quiet, legacy])["json_close_stops"] == 0
