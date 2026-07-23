"""shrew-ocr-preview serving-contract tests (SHREW_OCR_PREVIEW.md §2/§3/§5).

These pin the parts of the contract that silently void the eval numbers if
they drift: sampling params, message shape, the zlib gate, and the
single-flagged-enforcement-retry ladder.
"""

import zlib
from pathlib import Path

import pytest

from app.structured_page import (
    ENFORCEMENT_SCHEMA,
    MAX_TOKENS,
    SENTINEL,
    TEXT_PAGE_MAX_CHARS,
    ZLIB_GATE_RATIO,
    extract_page,
    extract_text_page,
    is_degenerate,
    paginate_text,
    parse_json_lenient,
    validate_schema,
    zlib_ratio,
)

# extract_page opens image_path to base64-encode it; the fake client never
# touches disk itself, so a tiny placeholder file at the hardcoded path used
# by every test below is enough (contents are irrelevant, only bytes matter).
Path("/tmp/fake.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

GOOD = ('{"metadata":{"title":null,"authors":[],"organization":null,"year":null,'
        '"doc_type":null},"summary":"s","semantic_chunks":[],"figures":[],"tables":[]}')

# Parses and passes the schema, but bad section_type -> schema gate failure.
BAD_SCHEMA = ('{"metadata":{"title":null,"authors":[],"organization":null,'
              '"year":null,"doc_type":null},"summary":"s","semantic_chunks":'
              '[{"chunk_id":"1","title":"t","content":"c","section_type":"bogus"}],'
              '"figures":[],"tables":[]}')


class FakeClient:
    def __init__(self, replies):  # list of (text, finish_reason)
        self.replies = list(replies)
        self.calls = []
        self.model = "shrew-ocr-preview"

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                         timeout=None, extra_params=None):
        self.calls.append({
            "system": messages[0]["content"],
            "user_content": messages[1]["content"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_params": extra_params,
        })
        text, finish_reason = self.replies.pop(0)
        return {"choices": [{"finish_reason": finish_reason,
                              "message": {"content": text}}]}


# ── §2 message shape ────────────────────────────────────────────────────────


def test_image_modality_sentinel_and_image_only_user():
    c = FakeClient([(GOOD, "stop")])
    extract_page("/tmp/fake.png", c)
    call = c.calls[0]
    assert call["system"] == SENTINEL
    user_content = call["user_content"]
    assert isinstance(user_content, list) and len(user_content) == 1
    assert user_content[0]["type"] == "image_url"


def test_text_modality_sends_raw_string_not_content_parts():
    """§2: the text arm was trained on a plain-string user message. Wrapping it
    in a [{"type": "text", ...}] part list is a different shape."""
    c = FakeClient([(GOOD, "stop")])
    extract_text_page("some messy html", c)
    call = c.calls[0]
    assert call["system"] == SENTINEL
    assert call["user_content"] == "some messy html"
    assert isinstance(call["user_content"], str)


# ── §3 sampling contract ────────────────────────────────────────────────────


def test_first_pass_sampling_is_greedy_and_bare():
    """§3: exactly temperature 0 and max_tokens 12000 — nothing else. Any decode
    knob here voids Gate F and smoke_test_preview reports DRIFT."""
    c = FakeClient([(GOOD, "stop")])
    extract_page("/tmp/fake.png", c)
    call = c.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == MAX_TOKENS == 12000
    assert not call["extra_params"], f"first pass must send no extras, got {call['extra_params']}"


def test_no_sampling_knobs_leak_in_for_qwen_named_models():
    """The generic Qwen flag-merge must not touch this stage — the served model
    could be named anything, and the contract admits no extra params."""
    c = FakeClient([(GOOD, "stop")])
    c.model = "qwen3.5-35b"
    extract_page("/tmp/fake.png", c)
    assert not c.calls[0]["extra_params"]


# ── §3 retry tier ───────────────────────────────────────────────────────────


def test_happy_path_is_a_single_call():
    c = FakeClient([(GOOD, "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert r["ok"] and r["status"] == "ok" and r["attempts"] == 1
    assert r["schema_coerced"] is False and r["degenerate"] is False
    assert r["data"]["summary"] == "s"


def test_parse_failure_retries_once_with_enforcement():
    """§3.2: the retry is the ONLY place enforcement belongs, and it must use
    structured_outputs — guided_json is silently ignored by the fork."""
    c = FakeClient([("not json", "stop"), (GOOD, "stop")])
    r = extract_page("/tmp/fake.png", c)

    assert r["ok"] and r["attempts"] == 2
    assert r["status"] == "ok_coerced" and r["schema_coerced"] is True

    retry = c.calls[1]
    assert retry["extra_params"]["structured_outputs"]["json"] == ENFORCEMENT_SCHEMA
    assert "guided_json" not in retry["extra_params"]
    # Never retry at temperature > 0 — that output is a lottery ticket.
    assert retry["temperature"] == 0


def test_schema_failure_takes_the_same_enforcement_retry():
    c = FakeClient([(BAD_SCHEMA, "stop"), (GOOD, "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert r["ok"] and r["status"] == "ok_coerced" and r["schema_coerced"] is True
    assert "structured_outputs" in c.calls[1]["extra_params"]


def test_retry_is_never_a_blind_repeat():
    """Greedy is deterministic, so an identical second request reproduces the
    identical failure. The retry must differ from the first call."""
    c = FakeClient([("not json", "stop"), (GOOD, "stop")])
    extract_page("/tmp/fake.png", c)
    first, retry = c.calls[0], c.calls[1]
    assert first["extra_params"] != retry["extra_params"]


def test_exactly_one_retry_then_page_level_failure():
    c = FakeClient([("not json", "stop"), ("still not json", "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert not r["ok"] and r["status"] == "failed" and r["attempts"] == 2
    assert len(c.calls) == 2, "must not retry more than once"


def test_length_finish_does_not_escalate_max_tokens():
    """max_tokens is fixed at the eval setting; escalating it would both deviate
    from §3 and blow past --max-model-len 16384."""
    c = FakeClient([('{"partial', "length"), (GOOD, "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert r["ok"] and r["status"] == "ok_coerced"
    assert c.calls[1]["max_tokens"] == c.calls[0]["max_tokens"] == MAX_TOKENS


def test_length_twice_reports_overlong_failed():
    c = FakeClient([('{"partial', "length"), ('{"still partial', "length")])
    r = extract_page("/tmp/fake.png", c)
    assert not r["ok"] and r["status"] == "overlong_failed"


def test_empty_200_is_reported_distinctly():
    """§5.3: empty completions with HTTP 200 signal TP-rank desync, not a bad
    page — the server watchdog needs to count them separately."""
    c = FakeClient([("", "stop"), ("", "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert not r["ok"] and r["status"] == "empty_completion"


# ── §5.1 zlib degeneration gate ─────────────────────────────────────────────


def test_zlib_ratio_matches_the_spec_formula():
    raw = GOOD
    expected = len(raw.encode()) / len(zlib.compress(raw.encode()))
    assert zlib_ratio(raw) == pytest.approx(expected)


def test_clean_output_is_below_the_gate():
    assert not is_degenerate(GOOD)
    assert zlib_ratio(GOOD) < ZLIB_GATE_RATIO


def test_loop_output_trips_the_gate():
    loop = '{"summary":"' + "the same clause over and over. " * 900 + '"}'
    assert zlib_ratio(loop) > ZLIB_GATE_RATIO
    assert is_degenerate(loop)


def test_degenerate_page_is_never_blind_retried():
    """§3.3: greedy reproduces the loop, so the retry must carry enforcement —
    and if it still fails the page is flagged degenerate, not merely failed."""
    loop = '{"summary":"' + "repeat forever. " * 1200 + '"}'
    c = FakeClient([(loop, "stop"), (loop, "stop")])
    r = extract_page("/tmp/fake.png", c)

    assert not r["ok"] and r["status"] == "degenerate" and r["degenerate"] is True
    assert "structured_outputs" in c.calls[1]["extra_params"]
    assert c.calls[1]["temperature"] == 0


def test_degenerate_page_recovered_by_enforcement_is_flagged_both_ways():
    loop = '{"summary":"' + "repeat forever. " * 1200 + '"}'
    c = FakeClient([(loop, "stop"), (GOOD, "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert r["ok"] and r["schema_coerced"] is True and r["degenerate"] is True


def test_degeneration_is_checked_before_the_schema_gate():
    """A loop that repeats a well-formed chunk still parses and still validates;
    only the raw-string zlib check catches it."""
    chunk = ('{"chunk_id":"1","title":"t","content":"identical body text here",'
             '"section_type":"results"}')
    loop = ('{"metadata":{"title":null,"authors":[],"organization":null,"year":null,'
            '"doc_type":null},"summary":"s","semantic_chunks":['
            + ",".join([chunk] * 400) + '],"figures":[],"tables":[]}')
    parsed, err = parse_json_lenient(loop)
    assert parsed is not None and validate_schema(parsed)[0], "loop passes parse+schema"
    assert is_degenerate(loop), "but the zlib gate must catch it"

    c = FakeClient([(loop, "stop"), (loop, "stop")])
    assert extract_page("/tmp/fake.png", c)["status"] == "degenerate"


# ── §5.4 never truncate ─────────────────────────────────────────────────────


def test_oversize_text_page_is_filtered_not_truncated():
    c = FakeClient([])  # must never be called
    r = extract_text_page("x" * (TEXT_PAGE_MAX_CHARS + 1), c)
    assert not r["ok"] and r["status"] == "oversize" and r["attempts"] == 0
    assert c.calls == [], "oversize page must not be sent"


def test_page_at_the_bound_is_sent():
    c = FakeClient([(GOOD, "stop")])
    r = extract_text_page("x" * TEXT_PAGE_MAX_CHARS, c)
    assert r["ok"] and len(c.calls) == 1


# ── pagination ──────────────────────────────────────────────────────────────


def test_paginate_keeps_short_text_as_one_page():
    assert paginate_text("para one\n\npara two") == ["para one\n\npara two"]


def test_paginate_splits_on_paragraph_boundaries():
    para = "y" * 4000
    pages = paginate_text("\n\n".join([para] * 6), max_chars=9000)
    assert len(pages) > 1
    assert all(len(p) <= 9000 for p in pages)
    # Nothing is lost and nothing is duplicated.
    assert "".join(p.replace("\n", "") for p in pages) == "y" * 24000


def test_paginate_never_splits_mid_line():
    lines = [f"row {i}: " + "z" * 200 for i in range(200)]
    pages = paginate_text("\n".join(lines), max_chars=2000)
    for page in pages:
        for line in page.split("\n"):
            assert line == "" or line in lines, f"line was cut: {line[:40]!r}"


def test_paginate_emits_unsplittable_line_as_its_own_page():
    """A single line over the bound can't be split without truncating, so it
    becomes its own page and extract_text_page filters it (§5.4)."""
    monster = "q" * 5000
    pages = paginate_text(f"short intro\n\n{monster}\n\nshort outro", max_chars=1000)
    assert monster in pages


def test_paginate_ignores_blank_input():
    assert paginate_text("   \n\n  ") == []


# ── §5.3 empty-200 watchdog ─────────────────────────────────────────────────


def test_empty_200_streak_counts_consecutive_blanks():
    from app import structured_page as sp
    sp.reset_empty_200_streak()

    for _ in range(3):
        c = FakeClient([("", "stop"), ("", "stop")])
        extract_page("/tmp/fake.png", c)
    # Two completions per page (first pass + retry), both empty.
    assert sp.empty_200_streak() == 6
    sp.reset_empty_200_streak()


def test_a_good_completion_clears_the_streak():
    """Baseline XGMI noise self-clears; only an unbroken run means desync."""
    from app import structured_page as sp
    sp.reset_empty_200_streak()

    extract_page("/tmp/fake.png", FakeClient([("", "stop"), ("", "stop")]))
    assert sp.empty_200_streak() > 0

    extract_page("/tmp/fake.png", FakeClient([(GOOD, "stop")]))
    assert sp.empty_200_streak() == 0


def test_streak_alerts_past_the_threshold(caplog):
    from app import structured_page as sp
    sp.reset_empty_200_streak()

    with caplog.at_level("ERROR"):
        for _ in range(sp.EMPTY_200_ALERT_THRESHOLD):
            extract_page("/tmp/fake.png", FakeClient([("", "stop"), ("", "stop")]))

    assert any("EMPTY-200 ALERT" in r.message for r in caplog.records)
    sp.reset_empty_200_streak()
