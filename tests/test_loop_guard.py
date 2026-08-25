"""§5.2 streaming repetition guard + §3.1 output budget.

The guard's whole job is to separate three outcomes that all look like
"truncated output" downstream: a loop, a legitimately long page, and a
false-positive abort. These tests pin that separation.
"""
import json
import random

import pytest

from app.structured_page import (
    BUCKET_IMAGE_TOKENS,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    REPETITION_ABORT,
    RepetitionGuard,
    _gate,
    check_output_budget,
    output_room,
)


def _varied(n_chars, seed=5):
    """Text with a real page's entropy. Synthetic filler built from one repeated
    sentence compresses harder than genuine prose and makes a poor control."""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    out = []
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(3, 11))))
    return " ".join(out)[:n_chars]


def _feed(guard, text, chunk=64):
    """Drive a guard the way chat_completion_stream does."""
    acc = ""
    for i in range(0, len(text), chunk):
        piece = text[i:i + chunk]
        acc += piece
        if guard(piece, acc):
            return acc
    return None


# ── the guard fires on loops ────────────────────────────────────────────────


def test_aborts_a_loop_and_reports_where():
    guard = RepetitionGuard()
    loop = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 4000
    stopped_at = _feed(guard, loop)
    assert stopped_at is not None, "a hard loop must abort"
    assert guard.fired and guard.ratio > guard.threshold
    # Aborting early is the point: it must not have paid for the whole output.
    assert len(stopped_at) < len(loop) / 4
    stats = guard.stats()
    assert stats["aborted"] and stats["position_chars"] == len(stopped_at)


def test_abort_needs_consecutive_violations():
    """Two spiking windows separated by a clean one must NOT abort — that is the
    false-positive class the K-consecutive rule exists to remove."""
    # Non-overlapping windows: repetitive, clean, repetitive.
    text = "ab" * 500 + _varied(1000) + "cd" * 1000

    single = RepetitionGuard(window=1000, check_every=1000, consecutive=1)
    double = RepetitionGuard(window=1000, check_every=1000, consecutive=2)

    assert _feed(single, text) is not None, "K=1 aborts on the first spike"
    assert _feed(double, text) is None, "K=2 must not, the spikes are not consecutive"
    # Both repetitive windows did violate; the clean one between them reset the run.
    assert double.violations == 1 and double.max_ratio > double.threshold


# ── the guard does NOT fire on legitimate content ───────────────────────────


def test_dense_numeric_table_does_not_abort():
    """The named false-positive risk: a page that is legitimately repetitive."""
    rows = "".join(
        f"<tr><td>{r}</td><td>{r * 37 % 991}.{r % 100:02d}</td>"
        f"<td>{r * 7 % 89}</td><td>Item {r}</td></tr>"
        for r in range(1, 900)
    )
    guard = RepetitionGuard()
    assert _feed(guard, json.dumps({"tables": [{"html": f"<table>{rows}</table>"}]})) is None
    assert not guard.fired
    # Calibration says clean pages peak around 10.1 with the threshold at 15.
    assert guard.max_ratio < guard.threshold


def test_long_prose_page_does_not_abort():
    guard = RepetitionGuard()
    assert _feed(guard, _varied(60000)) is None
    assert not guard.fired


def test_short_output_never_checked():
    """Checks start only once a FULL window exists — a partial window of JSON
    scaffolding compresses much harder than a real one."""
    guard = RepetitionGuard()
    assert _feed(guard, "aaaa" * 100) is None  # 400 chars, under the 2000 window
    assert guard.max_ratio == 0.0


# ── the abort is what marks the page looped ─────────────────────────────────


def test_abort_marks_degenerate_even_when_the_text_looks_clean():
    """§5.2: 'recorded as looped BY THE ABORT, not re-derived from the truncated
    text'. Stopping early leaves a short string whose whole-document ratio can
    land UNDER the §5.1 gate — re-deriving would count a real loop as clean."""
    # A loop whose repeating unit is long emits a good-looking prefix before the
    # abort lands, so the truncated string's own ratio is unremarkable.
    truncated = '{"summary": "' + _varied(1500) + '"'
    guard = RepetitionGuard()
    guard.fired, guard.ratio, guard.position = True, 41.2, 1500

    from app.structured_page import zlib_ratio, ZLIB_GATE_RATIO
    assert zlib_ratio(truncated) < ZLIB_GATE_RATIO, "precondition: looks clean"

    _, verdict, error = _gate(truncated, REPETITION_ABORT, guard)
    assert verdict == "degenerate"
    assert "41.2" in error and "1500" in error


def test_length_and_abort_are_distinguishable():
    """A page that runs long with a healthy ratio is over budget, not looping —
    different verdict, different remedy."""
    text = '{"metadata": {}, "semantic_chunks": [{"content": "real work"'
    _, over_budget, _ = _gate(text, "length", None)
    _, looped, _ = _gate(text, REPETITION_ABORT, RepetitionGuard())
    assert over_budget == "length" and looped == "degenerate"


# ── §3.1 output budget ──────────────────────────────────────────────────────


def test_configured_budget_is_reachable_on_every_bucket():
    report = check_output_budget()
    assert report["ok"], report["constrained_buckets"]
    assert report["output_room"]["B3"] >= MAX_TOKENS


def test_the_16384_trap_is_detected():
    """The exact misconfiguration that was misdiagnosed as model truncation
    twice: B3 can emit only 7,872 tokens at --max-model-len 16384."""
    report = check_output_budget(16384, 12000)
    assert not report["ok"]
    assert report["constrained_buckets"]["B3"] == 7872


def test_output_room_matches_the_spec_table():
    # §3.1 table, at the required --max-model-len 32768.
    assert output_room("B1", 32768) == 29488
    assert output_room("B2", 32768) == 27736
    assert output_room("B3", 32768) == 24256


def test_defaults_are_the_e4_contract():
    assert (MAX_TOKENS, MAX_MODEL_LEN) == (20000, 32768)
    assert BUCKET_IMAGE_TOKENS["B3"] == 7152


# ── page/attempt context on the abort warning ──────────────────────────────
# Pages stream concurrently, so an unprefixed warning burst can't be tied to
# page results; the prefix is the correlation key. Observability only — the
# abort behavior itself is pinned by the tests above.


def test_page_and_attempt_prefix_the_abort_warning(caplog):
    import logging
    guard = RepetitionGuard(page_no=37, attempt="attempt 1/2, first pass")
    loop = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 4000
    with caplog.at_level(logging.WARNING, logger="shrew.structured_page"):
        stopped_at = _feed(guard, loop)
    assert stopped_at is not None and guard.fired
    msgs = [r.getMessage() for r in caplog.records
            if "repetition_abort" in r.getMessage()]
    assert msgs and msgs[0].startswith(
        "Page 37 (attempt 1/2, first pass): repetition_abort")


def test_warning_stays_unprefixed_without_page_context(caplog):
    import logging
    guard = RepetitionGuard()
    loop = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 4000
    with caplog.at_level(logging.WARNING, logger="shrew.structured_page"):
        assert _feed(guard, loop) is not None
    msgs = [r.getMessage() for r in caplog.records
            if "repetition_abort" in r.getMessage()]
    assert msgs and msgs[0].startswith("repetition_abort")
