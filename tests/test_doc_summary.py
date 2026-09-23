"""Bounded document summary (platform ops 2026-09-23).

The structured path used to newline-join every page's summary into the
document ``summary``: 1.1 MB for a 3,501-page handbook, which overflowed a
downstream reranker and made a retrieval collection unsearchable. These tests
pin the replacement: bounded, never a concatenation on failure, single-page
documents untouched.
"""
import json
import logging

import pytest

import app.doc_summary as ds
from app.assembly import assemble_document, join_doc_summary
from app.structured_pipeline import (build_structured_json, render_raw_text,
                                       synthesize_markdown)


def _page(n, summary, section_types=("technical_content",), chunks=3):
    return {"page": n, "ok": True, "data": {
        "metadata": {"title": "Handbook", "authors": [], "organization": None,
                     "year": None, "doc_type": "manual"},
        "summary": summary,
        "semantic_chunks": [{"chunk_id": str(i), "title": f"S{n}.{i}",
                             "content": f"Page {n} chunk {i} body text.",
                             "section_type": section_types[i % len(section_types)]}
                            for i in range(chunks)],
        "figures": [], "tables": []}}


def _doc(page_results):
    return assemble_document("doc1", "/x.pdf", "pdf", page_results, stitch=False)


def _big_doc(n_pages=1000, chunks=3):
    """3,000 chunks: 1,000 pages x 3, ~315-char page summaries (the measured
    Machinery's Handbook ratio), a few overview pages scattered through."""
    prs = []
    for n in range(1, n_pages + 1):
        s = (f"Page {n} covers dimensional tolerances for machined part family {n}. "
             f"It lists fits, allowances and surface finish limits for {n} cases. "
             f"Worked examples follow the tables on this page for reference use in shop practice.")
        types = ("technical_content",)
        if n == 1:
            s = "Machinery's Handbook is a comprehensive reference for mechanical engineers and machinists."
            types = ("title",)
        elif n in (2, 3):
            types = ("introduction",)
        elif n == 500:
            types = ("abstract",)
        elif n == 999:
            types = ("conclusion",)
        prs.append(_page(n, s, types, chunks))
    return _doc(prs)


class _NoCallClient:
    model = "shrew-ocr-preview"

    def chat_completion(self, *a, **k):
        raise AssertionError("the deterministic path must not call the model")


class _ReplyClient:
    model = "shrew-ocr-preview"

    def __init__(self, text=None, finish="stop", exc=None):
        self.text, self.finish, self.exc, self.calls = text, finish, exc, []

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                        timeout=None, extra_params=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if self.exc:
            raise self.exc
        return {"choices": [{"finish_reason": self.finish,
                             "message": {"content": self.text}}]}


# ── a 3,000-chunk document stays under the budget ────────────────────────────

def test_3000_chunk_document_deterministic_is_bounded(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "deterministic")
    doc = _big_doc()
    assert len(doc["chunks"]) == 3000
    joined = join_doc_summary([p["page_summary"] for p in doc["pages"]])
    assert len(joined) > 200_000                 # what the old code returned

    report = ds.apply_document_summary(doc, synthesize_markdown(doc), _NoCallClient())
    s = doc["doc_summary"]
    assert report["status"] == "ok" and s and len(s) <= ds.DOC_SUMMARY_MAX_CHARS
    assert s.startswith("Machinery's Handbook is a comprehensive reference")
    # Selection rule: page 1, then abstract (500), then introduction (2, 3),
    # then conclusion (999) — never a walk over the whole document.
    assert report["pages_used"][:3] == [1, 500, 2]
    assert set(report["pages_used"]) <= {1, 2, 3, 500, 999}
    assert "Page 4 " not in s


def test_budget_cut_lands_on_a_sentence_boundary(monkeypatch):
    doc = _big_doc(n_pages=40)
    for p in doc["pages"][1:]:
        p["page_summary"] = "Alpha beta gamma delta epsilon. " * 20
    s, report = ds.compose_document_summary(doc, max_chars=300)
    assert s and len(s) <= 300 and s.endswith(".")


def test_3000_chunk_document_model_mode_is_bounded(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "model")
    doc = _big_doc()
    client = _ReplyClient("Machinery's Handbook is the standard reference for mechanical "
                          "engineering and machining practice, covering tolerances, fits, "
                          "materials, fasteners, gearing and shop mathematics.")
    report = ds.apply_document_summary(doc, synthesize_markdown(doc), client)
    assert report["status"] == "ok" and len(doc["doc_summary"]) <= ds.DOC_SUMMARY_MAX_CHARS
    assert len(client.calls) == 1                           # ONE call, not one per page
    user = client.calls[0]["messages"][1]["content"]
    # The legacy section rule, exactly: the first ~8k-approx-token section only.
    from app.structured import _section_document
    first = _section_document(synthesize_markdown(doc), max_tokens=8000)[0]["content"]
    assert ds._cap_tokens(first, 8000) in user and len(user) < len(synthesize_markdown(doc)) / 3
    assert client.calls[0]["messages"][0]["content"].startswith("You are a concise summarization")


# ── on failure: summary null + loud log, never a concatenation ──────────────

@pytest.mark.parametrize("client, why", [
    (_ReplyClient(exc=ConnectionError("backend down")), "ConnectionError"),
    (_ReplyClient(""), "empty completion"),
    (_ReplyClient("partial abstract that ran on", finish="length"), "token cap"),
    (_ReplyClient("the same words " * 400), "degenerate"),
    (_ReplyClient(" ".join(f"w{i * 7919 % 104729}" for i in range(900))), "fits the"),
    (_ReplyClient(json.dumps({"metadata": {}, "summary": "s"})), "structured JSON"),
])
def test_model_mode_failure_returns_null_and_logs(monkeypatch, caplog, client, why):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "model")
    doc = _big_doc(n_pages=20)
    with caplog.at_level(logging.ERROR, logger="app.doc_summary"):
        report = ds.apply_document_summary(doc, synthesize_markdown(doc), client)
    assert doc["doc_summary"] is None
    assert report["status"] == "failed" and why in report["error"]
    assert any("summary: null" in r.getMessage() for r in caplog.records)
    out = build_structured_json(doc, total_pages=20)
    assert out["summary"] is None


def test_deterministic_failure_returns_null_and_logs(monkeypatch, caplog):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "deterministic")
    doc = _big_doc(n_pages=5)
    for p in doc["pages"]:
        p["page_summary"] = ""
    with caplog.at_level(logging.ERROR, logger="app.doc_summary"):
        report = ds.apply_document_summary(doc, "", _NoCallClient())
    assert doc["doc_summary"] is None and report["status"] == "failed"
    assert any("summary: null" in r.getMessage() for r in caplog.records)
    # a first sentence longer than the whole budget is a failure too, not a cut word
    doc = _big_doc(n_pages=5)
    doc["pages"][0]["page_summary"] = "x" * 5000
    s, report = ds.compose_document_summary(doc, max_chars=4000)
    assert s is None and report["status"] == "failed"


def test_off_mode_is_null_without_a_call(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "off")
    doc = _big_doc(n_pages=5)
    report = ds.apply_document_summary(doc, "", _NoCallClient())
    assert doc["doc_summary"] is None and report["status"] == "disabled"


# ── single-page documents: byte-identical to the previous code ──────────────

@pytest.mark.parametrize("mode", ["deterministic", "summary_of_summaries", "model", "off"])
def test_single_page_output_is_byte_identical(monkeypatch, mode):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", mode)
    doc = _doc([_page(1, "A one-page memo about bearing preload limits.")])
    before = (json.dumps(build_structured_json(doc, 1), sort_keys=True),
              synthesize_markdown(doc), render_raw_text(doc))
    report = ds.apply_document_summary(doc, synthesize_markdown(doc), _NoCallClient())
    after = (json.dumps(build_structured_json(doc, 1), sort_keys=True),
             synthesize_markdown(doc), render_raw_text(doc))
    assert report["status"] == "single_page"
    assert after == before
    assert doc["doc_summary"] == "A one-page memo about bearing preload limits."


def test_page_summaries_are_untouched(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "deterministic")
    doc = _big_doc(n_pages=30)
    per_page = [p["page_summary"] for p in doc["pages"]]
    ds.apply_document_summary(doc, "", _NoCallClient())
    assert [p["page_summary"] for p in doc["pages"]] == per_page


# ── summary_of_summaries: one call over the page summaries ──────────────────

def test_summary_of_summaries_one_call_over_page_summaries(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "summary_of_summaries")
    doc = _big_doc()
    client = _ReplyClient("Machinery's Handbook is the standard shop reference for tolerances, "
                          "fits, materials and machining practice.")
    report = ds.apply_document_summary(doc, synthesize_markdown(doc), client)
    assert report["status"] == "ok" and report["mode"] == "summary_of_summaries"
    assert len(client.calls) == 1 and client.calls[0]["max_tokens"] == 4096
    system = client.calls[0]["messages"][0]["content"]
    user = client.calls[0]["messages"][1]["content"]
    assert "per-page summaries of ONE document" in system
    # page summaries in page order, not markdown; cut at ~8k tokens
    assert user.index("Machinery's Handbook is a comprehensive") < user.index("Page 2 covers")
    assert "<page " not in user and "chunk 0 body text" not in user
    assert ds._approx_tokens_safe(user) <= ds.DOC_SUMMARY_INPUT_TOKENS + 50
    assert "Page 1000 covers" not in user            # 315 KB of summaries do not all fit


def test_over_budget_answer_is_cut_on_a_sentence_boundary(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "summary_of_summaries")
    doc = _big_doc(n_pages=20)
    long = " ".join(f"Sentence {i} names a distinct topic {i * 7919 % 104729}." for i in range(300))
    report = ds.apply_document_summary(doc, "", _ReplyClient(long))
    s = doc["doc_summary"]
    assert report["status"] == "ok" and len(s) <= ds.DOC_SUMMARY_MAX_CHARS
    assert s.endswith(".") and report["cut_from_chars"] == len(long)


def test_cjk_input_window_does_not_overflow_context():
    zh = "\n".join("本页介绍了机械加工中的公差与配合标准以及表面粗糙度的要求。" * 3 for _ in range(2000))
    capped = ds._cap_tokens(zh, 8000)
    # chars/4 would have admitted ~32k CJK chars (~32k tokens); the safe cap keeps ~8k.
    assert len(capped) <= 8100 and ds._approx_tokens_safe(capped) <= 8000
