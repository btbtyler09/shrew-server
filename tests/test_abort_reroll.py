"""Re-roll on repetition-abort (the newspaper-loop fix).

The retry ladder assumed greedy decode is deterministic, so it only ever
retried with ENFORCEMENT (a changed request). But on the batched TP=4 gfx908
stack greedy is NOT deterministic across batch compositions: a dense page that
repetition-collapses on one batch draw decodes clean on another, and the
enforced pp-0.6 retry never recovers these (measured 2/74 -> 0/72). So on a
repetition_abort we re-run the SAME first-pass config first — a re-roll — which
is the only thing that recovers this class.
"""
import json

from app.structured_page import _extract, build_text_messages

GOOD = json.dumps({
    "metadata": {"title": "T", "authors": [], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "s",
    "semantic_chunks": [{"chunk_id": "1", "title": "A", "content": "body",
                         "keywords": [], "section_type": "introduction"}],
    "figures": [], "tables": [],
})
LOOP = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 6000


class _StreamClient:
    """Streams a per-call body; the guard's on_delta can abort mid-stream."""
    model = "shrew-ocr-preview"

    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.calls = 0

    def chat_completion_stream(self, messages, max_tokens=None, temperature=None,
                               timeout=None, extra_params=None, on_delta=None,
                               wall_clock_s=None):
        body = self.bodies[min(self.calls, len(self.bodies) - 1)]
        self.calls += 1
        acc, stop = "", None
        for i in range(0, len(body), 100):
            acc += body[i:i + 100]
            if on_delta is not None:
                stop = on_delta(body[i:i + 100], acc)
                if stop:
                    break
        return {"choices": [{"message": {"content": acc},
                             "finish_reason": stop or "stop"}]}


def test_reroll_recovers_a_nondeterministic_loop(monkeypatch):
    monkeypatch.setenv("SHREW_ABORT_REROLL", "1")
    # first pass loops (abort); the re-roll (same config) comes back clean.
    client = _StreamClient([LOOP, GOOD])
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    assert res["ok"] and res["status"] == "ok"
    assert client.calls == 2, "must re-roll the first-pass config, not jump to enforcement"


def test_reroll_count_is_configurable(monkeypatch):
    monkeypatch.setenv("SHREW_ABORT_REROLL", "2")
    # loop, loop, then clean on the 2nd re-roll
    client = _StreamClient([LOOP, LOOP, GOOD])
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    assert res["ok"]
    assert client.calls == 3


def test_persistent_loop_still_fails_after_rerolls(monkeypatch):
    monkeypatch.setenv("SHREW_ABORT_REROLL", "1")
    # always loops: first pass + 1 re-roll + the enforced retry = 3 calls, then fail
    client = _StreamClient([LOOP])
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    assert not res["ok"] and res["degenerate"] is True
    assert client.calls == 3


def test_no_reroll_when_disabled(monkeypatch):
    monkeypatch.setenv("SHREW_ABORT_REROLL", "0")
    client = _StreamClient([LOOP])  # loops, no re-roll -> straight to enforced retry
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    assert not res["ok"]
    assert client.calls == 2  # first pass + enforced retry only


def test_schema_failure_does_not_reroll_goes_to_enforcement(monkeypatch):
    monkeypatch.setenv("SHREW_ABORT_REROLL", "3")
    # a non-abort failure (invalid JSON, not a loop) must NOT re-roll — the
    # re-roll tier is only for repetition_abort; enforcement is for form errors.
    client = _StreamClient(["not json at all", GOOD])
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    # first pass fails to parse -> enforced retry returns GOOD -> ok, 2 calls,
    # no re-rolls consumed.
    assert res["ok"]
    assert client.calls == 2
