"""Decode ladder (§3 revised 2026-08-16, shrew-server-private#4):
try 1 = greedy + presence_penalty 0.3, retry = enforcement + presence_penalty
0.6. Enforcement on the first pass was measured and rejected (table-page
one-shot 0.225 vs 0.975)."""
import importlib

from app.generation import get_generation_params
from app.structured_page import enforcement_params


def test_first_pass_is_greedy_plus_mild_presence_penalty():
    p = get_generation_params("shrew-ocr-preview", "structured_page")
    assert p["temperature"] == 0
    assert p["extra_params"] == {"presence_penalty": 0.3}   # and NOTHING else


def test_retry_composes_enforcement_with_stronger_penalty():
    e = enforcement_params()
    assert "structured_outputs" in e          # schema enforcement retry-only
    assert e["presence_penalty"] == 0.6


def test_retry_penalty_overrides_first_pass_value():
    p = get_generation_params("shrew-ocr-preview", "structured_page")
    merged = {**(p["extra_params"] or {}), **enforcement_params()}
    assert merged["presence_penalty"] == 0.6


def test_env_zero_disables_penalties(monkeypatch):
    monkeypatch.setenv("SHREW_TRY1_PP", "0")
    monkeypatch.setenv("SHREW_RETRY_PP", "0")
    import app.generation as g, app.structured_page as sp
    importlib.reload(g)
    importlib.reload(sp)
    try:
        assert g.get_generation_params("m", "structured_page")["extra_params"] is None
        assert "presence_penalty" not in sp.enforcement_params()
    finally:
        monkeypatch.delenv("SHREW_TRY1_PP")
        monkeypatch.delenv("SHREW_RETRY_PP")
        importlib.reload(g)
        importlib.reload(sp)
