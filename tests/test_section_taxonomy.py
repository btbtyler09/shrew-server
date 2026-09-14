"""v0.3.14 (GitLab #27): SECTION_ENUM must be the model's TRAINED taxonomy.

Attribution finding (training, via the hub): 496 of 541 first-pass schema
errors were ``chunk:bad-section_type:body`` (+38 ``news_article``). The server
carried the 8-value v2 set while 100% of broadsheet rows and ~99% of dense rows
were trained on a larger taxonomy. So a CORRECT first pass failed schema, the
enforced retry's grammar (which derives from the same enum) forced a wrong
value, and the page landed "coerced" with mislabeled chunks after a second
full generation on the slowest class — 1,138 pages, 13% of the corpus.

Fix: grow the enum (never map down — the model is trained on these values);
normalize any value still outside it via canonical() and log it instead of
failing the page; keep the taxonomy a single source of truth — app/section_types.py
is vendored raw-identical from shrew_ocr/section_types.py, and a divergence test
fails if the copies drift.
"""
import json
import os

import pytest

from app import fallback, section_types
from app import structured_page as sp

GOOD_META = {"title": "T", "authors": [], "organization": None, "year": None,
             "doc_type": "report"}


def _page(section_type):
    return json.dumps({"metadata": GOOD_META, "summary": "s",
                       "semantic_chunks": [{"chunk_id": "1", "title": "A",
                                            "content": "body text", "keywords": [],
                                            "section_type": section_type}],
                       "figures": [], "tables": []})


# ── the whole trained taxonomy passes FIRST pass, untouched ──────────────────

@pytest.mark.parametrize("st", sorted(section_types.SECTION_SET))
def test_trained_section_types_pass_first_pass(st):
    parsed, verdict, err = sp._gate(_page(st), "stop")
    assert verdict == "ok", f"{st}: {err}"
    assert parsed["semantic_chunks"][0]["section_type"] == st  # untouched


def test_taxonomy_is_the_wide_one_not_the_v2_eight():
    # the two labels that caused 534/541 first-pass schema errors
    assert "body" in sp.SECTION_ENUM and "news_article" in sp.SECTION_ENUM
    assert len(sp.SECTION_ENUM) > 8
    assert sp.SECTION_FALLBACK in sp.SECTION_ENUM


# ── legacy / near-duplicate labels fold onto the taxonomy, never a failed page ─

@pytest.mark.parametrize("legacy,canon", [
    # labels E6 still emits from the pre-fold dense slice (rewritten in train v2.6.4)
    ("article", "news_article"), ("contents", "index"), ("entry", "body"),
    # form noise
    ("Stat-Box", "stat_box"), ("news article", "news_article"),
])
def test_legacy_section_type_folds_to_canonical(legacy, canon, caplog):
    caplog.set_level("INFO", logger="shrew.structured_page")
    parsed, verdict, err = sp._gate(_page(legacy), "stop")
    assert verdict == "ok", err
    assert parsed["semantic_chunks"][0]["section_type"] == canon
    assert any("section_type" in r.getMessage() and legacy in r.getMessage()
               for r in caplog.records)


def test_unknown_section_type_normalizes_to_fallback_not_failure(caplog):
    caplog.set_level("INFO", logger="shrew.structured_page")
    parsed, verdict, err = sp._gate(_page("zzqx_nonsense"), "stop")
    assert verdict == "ok", err
    assert parsed["semantic_chunks"][0]["section_type"] == sp.SECTION_FALLBACK == "other"
    # recorded in the gate log, so the drift is visible without failing the page
    assert any("section_type" in r.getMessage() and "zzqx_nonsense" in r.getMessage()
               for r in caplog.records)


def test_missing_section_type_normalizes_too():
    page = json.loads(_page("body"))
    del page["semantic_chunks"][0]["section_type"]
    parsed, verdict, _ = sp._gate(json.dumps(page), "stop")
    assert verdict == "ok"
    assert parsed["semantic_chunks"][0]["section_type"] == sp.SECTION_FALLBACK


def test_non_string_section_type_is_still_a_schema_error():
    # normalization only touches str-or-missing; a wrong TYPE keeps the old ladder
    page = json.loads(_page("body"))
    page["semantic_chunks"][0]["section_type"] = 7
    _, verdict, err = sp._gate(json.dumps(page), "stop")
    assert verdict == "schema" and "bad-section_type" in err


def test_normalize_returns_stray_values_and_ignores_malformed():
    parsed = {"semantic_chunks": [{"section_type": "article"}, "not-a-dict",
                                  {"section_type": "body"}, {}]}
    assert sp.normalize_section_types(parsed) == ["article", "<missing>"]
    assert sp.normalize_section_types("junk") == []
    assert sp.normalize_section_types({"semantic_chunks": None}) == []


# ── the enforcement grammar + fallback path offer the SAME taxonomy ──────────

def test_enforcement_schema_enum_matches_section_enum():
    enum = sp.ENFORCEMENT_SCHEMA["properties"]["semantic_chunks"]["items"] \
        ["properties"]["section_type"]["enum"]
    assert set(enum) == set(sp.SECTION_ENUM)
    assert "body" in enum and "news_article" in enum


def test_fallback_prompt_and_coercer_use_the_taxonomy():
    assert "__SECTION_TYPES__" not in fallback.FALLBACK_SYSTEM
    assert "news_article" in fallback.FALLBACK_SYSTEM and "body" in fallback.FALLBACK_SYSTEM
    out = fallback._coerce_five_key({"semantic_chunks": [
        {"content": "x", "section_type": "article"},
        {"content": "y", "section_type": "zzqx"},
        {"content": "z", "section_type": "news_article"}]})
    assert [c["section_type"] for c in out["semantic_chunks"]] == \
        ["news_article", "other", "news_article"]


# ── single source of truth: vendored copy must not diverge from shrew_ocr ────

SHREW_OCR_SECTION_TYPES = "/home/tyler/shrew/shrew_ocr/section_types.py"
METRICS_V2 = "/home/tyler/shrew/shrew_ocr/structured_eval/metrics_v2.py"


@pytest.mark.skipif(not os.path.exists(SHREW_OCR_SECTION_TYPES),
                    reason="shrew_ocr not present (e.g. public CI)")
def test_section_types_vendored_raw_identical():
    ours = open(section_types.__file__, "rb").read()
    theirs = open(SHREW_OCR_SECTION_TYPES, "rb").read()
    assert ours == theirs, "app/section_types.py diverged from shrew_ocr/section_types.py — cp it"


@pytest.mark.skipif(not os.path.exists(METRICS_V2),
                    reason="shrew_ocr eval not present (e.g. public CI)")
def test_eval_gate_derives_from_the_same_module():
    src = open(METRICS_V2).read()
    assert "from shrew_ocr.section_types import SECTION_SET as SECTION_ENUM" in src
