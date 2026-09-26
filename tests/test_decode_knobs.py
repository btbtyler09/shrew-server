"""Decode knobs: SHREW_TRY1_TEMP / SHREW_TRY2_TEMP / SHREW_SEED_POLICY and the
per-request overrides (temperature, retry_temperature, seed_policy).

The load-bearing test is the first one: at the defaults every request the
ladder sends is byte-identical to the payloads recorded from the pre-knob
source (tests/fixtures/decode_payloads_v0315.json, captured with
tests/decode_payload_capture.py against the v0.3.15-era tree).
"""
import json
import os
from pathlib import Path

import pytest

from app import structured_page as sp
from app.structured_page import DecodePolicy, parse_seed_policy, parse_temperature

HERE = Path(__file__).parent
KNOBS = ("SHREW_TRY1_TEMP", "SHREW_TRY2_TEMP", "SHREW_SEED_POLICY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)


def _capture(tmp_path):
    import sys
    sys.path.insert(0, str(HERE))
    import decode_payload_capture as cap
    return cap.capture(str(tmp_path))


def _bodies(tmp_path):
    return {k: [json.loads(b) for b in v] for k, v in _capture(tmp_path).items()}


# ── defaults: the exact payload shipped today ───────────────────────────────

def test_default_payloads_are_byte_identical_to_the_pre_knob_ladder(tmp_path):
    golden = json.loads((HERE / "fixtures" / "decode_payloads_v0315.json").read_text())
    now = _capture(tmp_path)
    assert now.keys() == golden.keys()
    for scenario in golden:
        assert now[scenario] == golden[scenario], scenario   # string compare = byte compare


def test_defaults_parse_to_greedy_fixed():
    d = DecodePolicy.from_env()
    assert (d.try1_temp, d.try2_temp, d.seed_policy) == (0.0, 0.0, "fixed")
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

def test_sampled_first_pass_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY1_TEMP", "0.4")
    b = _bodies(tmp_path)
    first = b["clean"][0]
    assert first["temperature"] == 0.4 and isinstance(first["seed"], int)
    assert first["presence_penalty"] == 0.3                     # the stage's own knob kept
    # the retry stays greedy (TRY2 default 0) and so carries no seed
    retry = b["parse_then_retry"][1]
    assert retry["temperature"] == 0 and "seed" not in retry and "structured_outputs" in retry


def test_sampled_retry_payload(monkeypatch, tmp_path):
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


def test_integer_policy_pins_every_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv("SHREW_TRY1_TEMP", "0.4")
    monkeypatch.setenv("SHREW_TRY2_TEMP", "0.4")
    monkeypatch.setenv("SHREW_SEED_POLICY", "77")
    assert [c["seed"] for c in _bodies(tmp_path)["loop_reroll_retry"]] == [77, 77, 77]


# ── fixed-by-hash: reproducible per page, distinct per attempt ──────────────

def test_fixed_seed_is_stable_across_calls_and_distinct_across_attempts(monkeypatch, tmp_path):
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
