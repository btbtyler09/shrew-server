"""Decode knobs: SHREW_TRY1_TEMP / SHREW_TRY2_TEMP / SHREW_SEED_POLICY and the
per-request overrides (temperature, retry_temperature, seed_policy).

The load-bearing tests are the first two:
  * at the v0.4.0 DEFAULTS every request the ladder sends is byte-identical to
    tests/fixtures/decode_payloads_v040.json (the recipe shipped with
    shrew-ocr-preview v0.4: two sampled attempts at 0.4 / pp 0.3, fixed seeds,
    un-enforced retry, no re-roll);
  * with the v0.3.x recipe set by ENV, every request is byte-identical to
    tests/fixtures/decode_payloads_v0315.json (recorded from the pre-knob
    v0.3.15-era tree) — the old ladder stays one env block away.
Both fixtures are produced by tests/decode_payload_capture.py.
"""
import json
import os
from pathlib import Path

import pytest

from app import structured_page as sp
from app.structured_page import DecodePolicy, parse_seed_policy, parse_temperature

HERE = Path(__file__).parent
KNOBS = ("SHREW_TRY1_TEMP", "SHREW_TRY2_TEMP", "SHREW_SEED_POLICY", "SHREW_TRY2_ENFORCE",
         "SHREW_ABORT_REROLL", "SHREW_RETRY_PP")

# The v0.3.x greedy ladder, as DecodePolicy fields and as the env block an
# operator sets to restore it (README "Decode recipe").
V0315 = dict(try1_temp=0.0, try2_temp=0.0, seed_policy="fixed", try2_enforce=True)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def v0315_env(monkeypatch):
    """Restore the v0.3.x recipe the way an operator would (SHREW_RETRY_PP is
    read at import, so the module value is patched alongside the env)."""
    for k, v in (("SHREW_TRY1_TEMP", "0"), ("SHREW_TRY2_TEMP", "0"),
                 ("SHREW_TRY2_ENFORCE", "1"), ("SHREW_ABORT_REROLL", "1"),
                 ("SHREW_RETRY_PP", "0.6")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sp, "RETRY_PP", 0.6)


def _capture(tmp_path):
    import sys
    sys.path.insert(0, str(HERE))
    import decode_payload_capture as cap
    return cap.capture(str(tmp_path))


def _bodies(tmp_path):
    return {k: [json.loads(b) for b in v] for k, v in _capture(tmp_path).items()}


# ── defaults: the exact payload shipped with v0.4.0 ─────────────────────────

def test_default_payloads_are_byte_identical_to_the_v040_recipe(tmp_path):
    golden = json.loads((HERE / "fixtures" / "decode_payloads_v040.json").read_text())
    now = _capture(tmp_path)
    assert now.keys() == golden.keys()
    for scenario in golden:
        assert now[scenario] == golden[scenario], scenario   # string compare = byte compare


def test_v0315_recipe_by_env_is_byte_identical_to_the_pre_knob_ladder(v0315_env, tmp_path):
    golden = json.loads((HERE / "fixtures" / "decode_payloads_v0315.json").read_text())
    now = _capture(tmp_path)
    assert now.keys() == golden.keys()
    for scenario in golden:
        assert now[scenario] == golden[scenario], scenario


def test_v040_defaults_are_two_sampled_unenforced_attempts(tmp_path):
    b = _bodies(tmp_path)
    first, retry = b["parse_then_retry"]
    for body in (first, retry):
        assert body["temperature"] == 0.4 and body["presence_penalty"] == 0.3
        assert isinstance(body["seed"], int)
        assert "structured_outputs" not in body and "json_schema" not in body
    assert first["seed"] != retry["seed"]                       # the retry is a fresh sample
    assert len(b["loop_reroll_retry"]) == 2                     # no re-roll: loop -> retry, done


def test_defaults_parse_to_the_v040_recipe():
    d = DecodePolicy.from_env()
    assert (d.try1_temp, d.try2_temp, d.seed_policy, d.try2_enforce) == (0.4, 0.4, "fixed", False)
    assert d == DecodePolicy()                                  # env defaults == dataclass defaults
    assert d.seed_for("k", 0, 0) is None                       # greedy never carries a seed


# ── env parsing ──────────────────────────────────────────────────────────────

def test_env_parsing(monkeypatch):
    monkeypatch.setenv("SHREW_TRY1_TEMP", "0.4")
    monkeypatch.setenv("SHREW_TRY2_TEMP", "0.7")
    monkeypatch.setenv("SHREW_SEED_POLICY", "RANDOM")
    d = DecodePolicy.from_env()
    assert (d.try1_temp, d.try2_temp, d.seed_policy) == (0.4, 0.7, "random")
    monkeypatch.setenv("SHREW_SEED_POLICY", "1234")
    assert DecodePolicy.from_env().seed_policy == 1234


@pytest.mark.parametrize("bad", ["-0.1", "2.5", "warm", "", "nan"])
def test_bad_temperature_is_rejected(monkeypatch, bad):
    monkeypatch.setenv("SHREW_TRY1_TEMP", bad)
    with pytest.raises(ValueError, match="SHREW_TRY1_TEMP"):
        DecodePolicy.from_env()


@pytest.mark.parametrize("bad", ["sometimes", "-1", str(2 ** 31), "1.5"])
def test_bad_seed_policy_is_rejected(bad):
    with pytest.raises(ValueError):
        parse_seed_policy(bad)


# ── request payloads carry temperature / seed as configured ─────────────────

def test_sampled_first_pass_payload(v0315_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY1_TEMP", "0.4")
    b = _bodies(tmp_path)
    first = b["clean"][0]
    assert first["temperature"] == 0.4 and isinstance(first["seed"], int)
    assert first["presence_penalty"] == 0.3                     # the stage's own knob kept
    # the retry stays greedy (TRY2 default 0) and so carries no seed
    retry = b["parse_then_retry"][1]
    assert retry["temperature"] == 0 and "seed" not in retry and "structured_outputs" in retry


def test_sampled_retry_payload(v0315_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY2_TEMP", "0.4")
    b = _bodies(tmp_path)
    first, retry = b["parse_then_retry"]
    assert first["temperature"] == 0 and "seed" not in first    # first pass untouched
    assert retry["temperature"] == 0.4 and isinstance(retry["seed"], int)
    assert retry["presence_penalty"] == 0.6 and "structured_outputs" in retry and "json_schema" in retry


def test_random_policy_sends_no_seed(monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY1_TEMP", "0.4")
    monkeypatch.setenv("SHREW_TRY2_TEMP", "0.4")
    monkeypatch.setenv("SHREW_SEED_POLICY", "random")
    for calls in _bodies(tmp_path).values():
        for body in calls:
            assert body["temperature"] == 0.4 and "seed" not in body


def test_integer_policy_pins_every_attempt(v0315_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY1_TEMP", "0.4")
    monkeypatch.setenv("SHREW_TRY2_TEMP", "0.4")
    monkeypatch.setenv("SHREW_SEED_POLICY", "77")
    assert [c["seed"] for c in _bodies(tmp_path)["loop_reroll_retry"]] == [77, 77, 77]


# ── fixed-by-hash: reproducible per page, distinct per attempt ──────────────

def test_fixed_seed_is_stable_across_calls_and_distinct_across_attempts(v0315_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY1_TEMP", "0.4")
    monkeypatch.setenv("SHREW_TRY2_TEMP", "0.4")
    a = _bodies(tmp_path)["loop_reroll_retry"]
    b = _bodies(tmp_path)["loop_reroll_retry"]
    seeds_a = [c["seed"] for c in a]
    assert seeds_a == [c["seed"] for c in b]                    # rerun reproduces
    assert len(set(seeds_a)) == 3                               # first pass, re-roll, retry all differ


def test_fixed_seed_depends_on_the_page_input():
    d = DecodePolicy(0.4, 0.4, "fixed")
    k1 = sp.page_input_key(sp.build_text_messages("page one"))
    k2 = sp.page_input_key(sp.build_text_messages("page two"))
    assert d.seed_for(k1, 0, 0.4) == d.seed_for(k1, 0, 0.4)
    assert d.seed_for(k1, 0, 0.4) != d.seed_for(k2, 0, 0.4)
    assert 0 <= d.seed_for(k1, 5, 0.4) < 2 ** 31


# ── per-request overrides ────────────────────────────────────────────────────

def test_overrides_layer_on_env(monkeypatch):
    monkeypatch.setenv("SHREW_TRY2_TEMP", "0.2")
    d = DecodePolicy.from_env().with_overrides(temperature="0.4", seed_policy="random")
    assert (d.try1_temp, d.try2_temp, d.seed_policy) == (0.4, 0.2, "random")
    assert DecodePolicy.from_env().with_overrides() == DecodePolicy.from_env()


def test_server_validates_overrides_as_400():
    from fastapi import HTTPException
    from app.server import _validate_decode_params
    assert _validate_decode_params(None, None, None) is None     # absent = env defaults, untouched
    d = _validate_decode_params("0.4", None, "123")
    assert (d.try1_temp, d.seed_policy) == (0.4, 123)
    for bad in (("hot", None, None), (None, "3", None), (None, None, "often")):
        with pytest.raises(HTTPException) as e:
            _validate_decode_params(*bad)
        assert e.value.status_code == 400


def test_pipeline_threads_the_request_policy_to_every_page(monkeypatch, tmp_path):
    import app.structured_pipeline as spl
    seen = []
    monkeypatch.setattr(spl, "extract_page",
                        lambda path, client, **kw: seen.append(kw.get("decode")) or
                        {"ok": False, "data": None, "status": "failed", "error": "x",
                         "attempts": 1, "raw_len": 0})
    from PIL import Image
    img = tmp_path / "p.png"
    Image.new("RGB", (400, 600), "white").save(img)
    pol = DecodePolicy(0.4, 0.0, "fixed")
    spl._process_one_page(1, str(img), type("C", (), {"low_dpi": 100, "high_dpi": 200})(),
                          str(tmp_path), client=None, decode=pol)
    assert seen == [pol]


# ── SHREW_TRY2_ENFORCE / retry_enforce ──────────────────────────

@pytest.mark.parametrize("val", ["1", "on", "true", "YES"])
def test_explicit_enforce_on_is_still_byte_identical(v0315_env, monkeypatch, tmp_path, val):
    monkeypatch.setenv("SHREW_TRY2_ENFORCE", val)
    golden = json.loads((HERE / "fixtures" / "decode_payloads_v0315.json").read_text())
    assert _capture(tmp_path) == golden


def test_enforce_off_retry_uses_the_first_pass_contract(v0315_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY2_ENFORCE", "off")
    b = _bodies(tmp_path)
    first, retry = b["parse_then_retry"]
    assert "structured_outputs" not in retry and "json_schema" not in retry
    assert retry["presence_penalty"] == 0.6                     # SHREW_RETRY_PP
    assert retry["temperature"] == 0 and "seed" not in retry     # SHREW_TRY2_TEMP default
    # apart from the penalty, the retry request is the first pass's
    strip = lambda d: {k: v for k, v in d.items() if k != "presence_penalty"}
    assert strip(retry) == strip(first)
    assert b["clean"][0] == json.loads(json.loads(
        (HERE / "fixtures" / "decode_payloads_v0315.json").read_text())["clean"][0])  # first pass untouched


def test_enforce_off_with_sampled_retry(v0315_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY2_ENFORCE", "0")
    monkeypatch.setenv("SHREW_TRY2_TEMP", "0.4")
    retry = _bodies(tmp_path)["loop_reroll_retry"][2]
    assert retry["temperature"] == 0.4 and isinstance(retry["seed"], int)
    assert retry["presence_penalty"] == 0.6 and "structured_outputs" not in retry


def test_bad_enforce_value_fails_fast(monkeypatch):
    monkeypatch.setenv("SHREW_TRY2_ENFORCE", "maybe")
    with pytest.raises(ValueError, match="SHREW_TRY2_ENFORCE"):
        DecodePolicy.from_env()


def test_retry_enforce_override():
    from fastapi import HTTPException
    from app.server import _validate_decode_params
    assert DecodePolicy.from_env().with_overrides(retry_enforce="off").try2_enforce is False
    d = _validate_decode_params(None, None, None, "false")
    assert d is not None and d.try2_enforce is False
    with pytest.raises(HTTPException) as e:
        _validate_decode_params(None, None, None, "sometimes")
    assert e.value.status_code == 400


def test_describe_records_the_whole_recipe(monkeypatch):
    d = DecodePolicy.from_env().describe()
    assert d == {"try1_temperature": 0.4, "try2_temperature": 0.4, "seed_policy": "fixed",
                 "try1_pp": 0.3, "retry_pp": 0.3, "try2_enforce": False}
    d2 = DecodePolicy(0.4, 0.4, 7, False).describe()
    assert (d2["try1_temperature"], d2["try2_temperature"], d2["seed_policy"], d2["try2_enforce"]) == (0.4, 0.4, 7, False)
    json.dumps(d2)                                              # log-serialisable


# ── rescued_by: which rung produced the page ──────────────────────────────

class _SeqClient:
    """Streams scripted completions in call order through on_delta."""
    model = "shrew-ocr-preview"

    def __init__(self, completions):
        self.q = list(completions)

    def chat_completion_stream(self, messages, *, max_tokens, temperature, extra_params,
                               timeout, on_delta, wall_clock_s):
        text, acc, reason = self.q.pop(0), "", None
        for i in range(0, len(text), 50):
            acc += text[i:i + 50]
            reason = on_delta(text[i:i + 50], acc)
            if reason:
                break
        return {"choices": [{"message": {"role": "assistant", "content": acc},
                             "finish_reason": reason or "stop"}]}


def _run(completions, **policy):
    import sys
    sys.path.insert(0, str(HERE))
    import decode_payload_capture as cap
    seq = [{"GOOD": cap.GOOD, "LOOP": cap.LOOP}.get(c, c) for c in completions]
    return sp.extract_text_page("A short text page.", _SeqClient(seq), page_no=1,
                                decode=DecodePolicy(**policy))


@pytest.mark.parametrize("script, policy, want_status, want_rescued, want_coerced", [
    (["GOOD"], V0315, "ok", "none", False),
    (["LOOP", "GOOD"], V0315, "ok", "reroll", False),
    (["not json", "GOOD"], V0315, "ok_coerced", "retry_enforced", True),
    (["not json", "GOOD"], {**V0315, "try2_enforce": False}, "ok_coerced", "retry_unenforced", True),
    (["LOOP", "LOOP", "GOOD"], {**V0315, "try2_enforce": False}, "ok_coerced", "retry_unenforced", True),
    (["not json", "still not json"], V0315, "failed", "none", False),
])
def test_rescued_by_names_the_rung(v0315_env, script, policy, want_status, want_rescued, want_coerced):
    r = _run(script, **policy)
    assert r["status"] == want_status
    assert r["rescued_by"] == want_rescued
    # schema_coerced keeps its historical meaning: "rescued by the retry tier",
    # enforced or not — the density floor and gate counts depend on it.
    assert r["schema_coerced"] is want_coerced


def test_rescued_by_flows_to_page_results_and_the_decode_log():
    from app.structured_pipeline import _page_result, rescued_by_counts
    prs = [_page_result(i, {"ok": True, "data": {}, "status": "ok", "rescued_by": rb})
           for i, rb in enumerate(["none", "none", "reroll", "retry_unenforced"], 1)]
    prs.append(_page_result(5, {"ok": False, "data": None, "status": "failed"}))  # legacy / no field
    assert [p["rescued_by"] for p in prs] == ["none", "none", "reroll", "retry_unenforced", "none"]
    assert rescued_by_counts(prs) == {"none": 3, "reroll": 1, "retry_enforced": 0,
                                      "retry_unenforced": 1, "fallback": 0}


# ── attempts / model_calls / doc id on guard lines ───────────────

@pytest.mark.parametrize("script, policy, attempts, calls", [
    (["GOOD"], V0315, 1, 1),
    (["LOOP", "GOOD"], V0315, 2, 2),                                   # re-roll rescue
    (["not json", "GOOD"], V0315, 2, 2),                               # retry, no re-roll
    (["LOOP", "LOOP", "GOOD"], {**V0315, "try2_enforce": False}, 2, 3),  # re-roll then retry
    (["LOOP", "LOOP", "LOOP"], V0315, 2, 3),                           # all rungs fail
])
def test_attempts_keeps_its_value_and_model_calls_is_exact(v0315_env, script, policy, attempts, calls):
    r = _run(script, **policy)
    assert r["attempts"] == attempts                    # historical value unchanged
    assert r["model_calls"] == calls


def test_page_record_and_decode_log_carry_attempts():
    from app.structured_pipeline import _page_result, attempt_counts
    prs = [_page_result(1, {"ok": True, "data": {}, "status": "ok", "attempts": 1, "model_calls": 1,
                            "rescued_by": "none"}),
           _page_result(2, {"ok": True, "data": {}, "status": "ok", "attempts": 2, "model_calls": 2,
                            "rescued_by": "reroll"}),
           _page_result(3, {"ok": True, "data": {}, "status": "ok_coerced", "attempts": 2,
                            "model_calls": 3, "rescued_by": "retry_unenforced",
                            "schema_coerced": True})]
    assert [(p["attempts"], p["model_calls"], p["rescued_by"]) for p in prs] == \
        [(1, 1, "none"), (2, 2, "reroll"), (2, 3, "retry_unenforced")]
    # a re-roll success is no longer indistinguishable from a first-pass success
    assert prs[1]["status"] == prs[0]["status"] == "ok" and prs[1]["rescued_by"] != prs[0]["rescued_by"]
    assert attempt_counts(prs) == {"attempts": {"1": 1, "2": 2}, "model_calls": 6}


def test_guard_lines_carry_the_doc_id(caplog):
    import logging
    import sys
    sys.path.insert(0, str(HERE))
    import decode_payload_capture as cap
    caplog.set_level(logging.WARNING, logger="shrew.structured_page")
    sp.extract_text_page("x", _SeqClient([cap.LOOP, cap.GOOD]), page_no=4,
                         decode=DecodePolicy(), doc_id="abc123def4567890")
    lines = [r.getMessage() for r in caplog.records if "repetition_abort" in r.getMessage()]
    assert lines and lines[0].startswith("Doc abc123def4567890 Page 4 (attempt 1/2, first pass): ")


def test_guard_line_format_is_unchanged_without_a_doc_id():
    g = sp.RepetitionGuard(page_no=3, attempt="attempt 1/2, first pass")
    assert g.log_context() == "Page 3 (attempt 1/2, first pass): "
    assert sp.RepetitionGuard().log_context() == ""
    assert sp.RepetitionGuard(doc_id="d1").log_context() == "Doc d1: "


# ── the two-attempt recipe reaches the fallback ──────────────────────────

def test_two_attempt_unenforced_policy_then_fallback(monkeypatch, tmp_path):
    """First pass temp 0.4 / pp 0.3 / fixed seed, SHREW_ABORT_REROLL=0, one
    un-enforced retry at the same setting with a fresh seed; both fail ->
    the fallback client runs (the same `not result["ok"]` gate as today)."""
    import app.fallback as fb
    import app.structured_pipeline as spl
    from PIL import Image
    monkeypatch.setenv("SHREW_ABORT_REROLL", "0")
    monkeypatch.setenv("SHREW_RETRY_PP", "0.3")
    import importlib
    importlib.reload(sp)                                        # RETRY_PP is read at import
    try:
        calls = []

        class Rec(_SeqClient):
            def chat_completion_stream(self, messages, **kw):
                calls.append(kw)
                return super().chat_completion_stream(messages, **kw)

        client = Rec(["not json", "still not json"])
        monkeypatch.setattr(spl, "extract_page", sp.extract_page)
        fb_calls = []
        monkeypatch.setattr(fb, "extract_page_fallback",
                            lambda *a, **k: fb_calls.append(a) or {"summary": "rescued"})
        img = tmp_path / "p.png"
        Image.new("RGB", (400, 600), "white").save(img)
        cfg = type("C", (), {"low_dpi": 100, "high_dpi": 200})()
        pol = sp.DecodePolicy(0.4, 0.4, "fixed", False)
        r = spl._process_one_page(1, str(img), cfg, str(tmp_path), client,
                                  fallback_client=object(), decode=pol, doc_id="d")
        assert len(calls) == 2                                  # two shrew attempts, no re-roll
        first, retry = calls
        assert first["temperature"] == retry["temperature"] == 0.4
        assert first["extra_params"]["presence_penalty"] == retry["extra_params"]["presence_penalty"] == 0.3
        assert "structured_outputs" not in retry["extra_params"] and "json_schema" not in retry["extra_params"]
        assert first["extra_params"]["seed"] != retry["extra_params"]["seed"]   # fresh seed
        assert len(fb_calls) == 1                               # fallback ran after try-2
        assert r["status"] == "fallback_ok" and r["rescued_by"] == "fallback"
        assert r["model_calls"] == 3
    finally:
        monkeypatch.delenv("SHREW_RETRY_PP")
        importlib.reload(sp)


def test_no_fallback_client_means_the_page_fails_after_try2(monkeypatch, tmp_path):
    import app.structured_pipeline as spl
    from PIL import Image
    monkeypatch.setenv("SHREW_ABORT_REROLL", "0")
    monkeypatch.setattr(spl, "extract_page", sp.extract_page)
    img = tmp_path / "p.png"
    Image.new("RGB", (400, 600), "white").save(img)
    r = spl._process_one_page(1, str(img), type("C", (), {"low_dpi": 100, "high_dpi": 200})(),
                              str(tmp_path), _SeqClient(["bad", "bad"]),
                              decode=sp.DecodePolicy(0.4, 0.4, "fixed", False))
    assert r["ok"] is False and r["status"] == "failed" and r["rescued_by"] == "none"
    assert r["model_calls"] == 2


# ── the /v1/convert response carries processing_log.decode ────────────────

def test_convert_response_carries_the_decode_block():
    from app.models import PipelineResult
    from app.server import _build_response
    from app.structured_pipeline import attempt_counts, rescued_by_counts
    decode = {**DecodePolicy.from_env().describe(), "rescued_by": rescued_by_counts([]),
              **attempt_counts([])}
    log = {"total_pages": 1, "total_figures": 0, "total_time_seconds": 1.0, "modality": "image",
           "gates": {"pages": 1}, "failed_pages": 0, "decode": decode,
           "table_stitch": {"attempted": 0, "model_refined": 0},
           "doc_summary": {"status": "ok", "chars": 10},
           "fidelity": {"flagged": [{"token": "x"}] * 3}}
    r = PipelineResult(clean_markdown="m", structured_json={}, processing_log=log)
    out = _build_response(r, False)["processing_log"]
    assert set(out["decode"]) == {"try1_temperature", "try2_temperature", "seed_policy", "try1_pp",
                                  "retry_pp", "try2_enforce", "rescued_by", "attempts", "model_calls"}
    assert out["decode"] == decode
    assert out["table_stitch"] == log["table_stitch"] and out["doc_summary"] == log["doc_summary"]
    assert "fidelity" not in out                                 # unbounded list, left out
    json.dumps(out)


# ── the v0.4.0 ladder: two attempts, then the fallback ───────────────────────

@pytest.mark.parametrize("script, want_status, want_rescued, calls", [
    (["GOOD"], "ok", "none", 1),
    (["LOOP", "GOOD"], "ok_coerced", "retry_unenforced", 2),      # no re-roll rung: straight to try 2
    (["not json", "GOOD"], "ok_coerced", "retry_unenforced", 2),
    (["LOOP", "LOOP"], "degenerate", "none", 2),                  # -> fallback client when configured
])
def test_v040_default_ladder(script, want_status, want_rescued, calls):
    r = _run(script)
    assert (r["status"], r["rescued_by"], r["model_calls"]) == (want_status, want_rescued, calls)
    assert r["attempts"] == min(calls, 2)
