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
    # A 1,000-page document is LONG: page 1, then `abstract` pages only
    # (introduction/conclusion labels are unreliable on long manuals).
    assert report["overview_types"] == ["abstract"]
    assert report["pages_used"] == [1, 500]
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

@pytest.mark.parametrize("mode", ["deterministic", "contract_sos", "summary_of_summaries", "model", "off"])
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


# ── near-duplicate dedupe, on the real paper18 page summaries (card 3, 09-23) ──

PAPER18 = {1: "This research paper presents a massive 172-billion-token study using the RIKER methodology to quantify hallucination rates in large language models across various context lengths, temperatures, and hardware platforms. The findings reveal that even top-performing models fabricate information at significant rates, with hallucination increasing sharply as context length grows, while model selection and temperature settings are identified as primary drivers of fabrication resistance.",
           2: "This research paper evaluates the hallucination rates of large language models in enterprise document Q&A scenarios, analyzing the impact of temperature settings, hardware platforms, and context length. The study finds that fabrication rates increase with context length and that setting temperature to zero does not universally improve factuality, challenging conventional wisdom on model selection.",
           16: "This research paper evaluates the hallucination rates of 35 open-weight LLMs when answering questions about documents, finding that fabrication is substantial and increases significantly with context length. The study concludes that model selection and context length are the primary drivers of accuracy, while temperature settings require nuanced optimization to balance fabrication resistance with coherence."}


def test_paper18_restatements_are_deduplicated():
    prs = [_page(1, PAPER18[1], ("abstract",)), _page(2, PAPER18[2], ("introduction",))]
    prs += [_page(n, "", ("technical_content",)) for n in range(3, 16)]
    prs += [_page(16, PAPER18[16], ("conclusion",))] + [_page(n, "", ("technical_content",)) for n in (17, 18)]
    doc = _doc(prs)
    s, report = ds.compose_document_summary(doc)
    assert report["status"] == "ok" and 450 <= len(s) <= 900, len(s)
    assert s.count("This research paper") < 3
    assert s.startswith("This research paper presents a massive 172-billion-token study")
    # coverage kept: something from the introduction page AND the conclusion page
    assert report["pages_used"] == [1, 2, 16]
    assert report["sentences_dropped_as_duplicate"] == 2


def test_dedupe_works_on_cjk_text():
    a = "本手册介绍了机械加工中的公差与配合标准。"
    b = "本手册介绍了机械加工中的公差与配合的标准。"          # near-duplicate
    c = "第二部分讨论齿轮设计与材料热处理工艺。"              # new content
    doc = _doc([_page(1, a, ("title",)), _page(2, b, ("introduction",)),
                _page(3, c, ("conclusion",))])
    s, report = ds.compose_document_summary(doc)
    assert b not in s and a in s and c in s
    assert report["sentences_dropped_as_duplicate"] == 1


# ── contract_sos: page summaries through the TEXT-PAGE contract ─────────────

class _StreamContractClient:
    """Streams a 5-key contract answer through on_delta like VLMClient and
    records how much of it was actually generated before the abort."""
    model = "shrew-ocr-preview"

    def __init__(self, answer, chunk=17, exc=None):
        self.answer, self.chunk, self.exc, self.calls = answer, chunk, exc, []

    def chat_completion_stream(self, messages, *, max_tokens, temperature,
                               extra_params, on_delta, wall_clock_s, timeout=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "extra_params": extra_params})
        if self.exc:
            raise self.exc
        acc, reason = "", None
        for i in range(0, len(self.answer), self.chunk):
            acc += self.answer[i:i + self.chunk]
            reason = on_delta(self.answer[i:i + self.chunk], acc)
            if reason:
                break
        self.generated = len(acc)
        return {"choices": [{"message": {"role": "assistant", "content": acc},
                             "finish_reason": reason or "stop"}]}


def _contract_answer(summary):
    chunks = [{"chunk_id": f"c{i}", "title": "t", "section_type": "body",
               "content": "transcribed input " * 40} for i in range(200)]
    return json.dumps({"metadata": {"title": "Machinery's Handbook", "authors": [],
                                    "organization": None, "year": None, "doc_type": "manual"},
                       "summary": summary, "semantic_chunks": chunks,
                       "tables": [], "figures": []})


def test_contract_sos_request_shape_and_early_stop(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "contract_sos")
    doc = _big_doc()
    answer = _contract_answer('Machinery\'s Handbook is the "standard" shop reference.')
    client = _StreamContractClient(answer)
    report = ds.apply_document_summary(doc, "", client)
    assert report["status"] == "ok" and report["mode"] == "contract_sos"
    assert doc["doc_summary"] == 'Machinery\'s Handbook is the "standard" shop reference.'
    call = client.calls[0]
    # the text-page contract, exactly: sentinel system prompt, raw string user turn
    from app.structured_page import SENTINEL, TEXT_PAGE_MAX_CHARS, get_generation_params
    assert call["messages"][0] == {"role": "system", "content": SENTINEL}
    assert isinstance(call["messages"][1]["content"], str)
    assert call["messages"][1]["content"].startswith("Machinery's Handbook is a comprehensive")
    assert len(call["messages"][1]["content"]) <= TEXT_PAGE_MAX_CHARS
    p = get_generation_params("shrew-ocr-preview", "structured_page")
    assert call["temperature"] == p["temperature"] and call["extra_params"] == p["extra_params"]
    # stopped right after the summary key: the chunk transcription is never paid for
    assert client.generated < len(answer) / 20


@pytest.mark.parametrize("client, why", [
    (_StreamContractClient(_contract_answer(None)), "empty or null"),
    (_StreamContractClient(_contract_answer("")), "empty or null"),
    (_StreamContractClient('{"metadata": {"title": "x"}, "semantic_chunks": [' + '{"c": 1},' * 50),
     "no summary key"),
    (_StreamContractClient("", exc=ConnectionError("backend down")), "ConnectionError"),
])
def test_contract_sos_failure_falls_back_to_deterministic(monkeypatch, caplog, client, why):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "contract_sos")
    doc = _big_doc(n_pages=10)
    with caplog.at_level(logging.WARNING, logger="app.doc_summary"):
        report = ds.apply_document_summary(doc, "", client)
    # the tree failed loudly ...
    assert report["contract_sos"]["status"] == "failed" and why in report["contract_sos"]["error"]
    assert any("summary: null" in r.getMessage() for r in caplog.records)
    assert any("falling back to deterministic" in r.getMessage() for r in caplog.records)
    # ... and the document still gets a bounded deterministic abstract
    assert report["mode_used"] == "deterministic_fallback" and report["status"] == "ok"
    assert doc["doc_summary"].startswith("Machinery's Handbook is a comprehensive reference")


def test_contract_sos_and_fallback_both_failing_is_null(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "contract_sos")
    doc = _big_doc(n_pages=10)
    for p in doc["pages"]:
        p["page_summary"] = ""
    report = ds.apply_document_summary(doc, "", _StreamContractClient("", exc=ConnectionError("x")))
    assert doc["doc_summary"] is None and report["mode_used"] == "none"


def test_mode_used_is_reported(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "contract_sos")
    doc = _big_doc(n_pages=10)
    report = ds.apply_document_summary(doc, "", _StreamContractClient(_contract_answer("An abstract.")))
    assert report["mode_used"] == "contract_sos"
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "deterministic")
    doc = _big_doc(n_pages=10)
    assert ds.apply_document_summary(doc, "", _NoCallClient())["mode_used"] == "deterministic"


@pytest.mark.parametrize("n_pages, expect_intro", [(18, True), (30, True), (31, False), (50, False)])
def test_long_documents_use_abstract_pages_only(n_pages, expect_intro):
    prs = [_page(1, "A reference manual for machinists.", ("title",)),
           _page(2, "Page two introduces the conventions used for tolerances.", ("introduction",))]
    prs += [_page(n, f"Body page {n} lists gear data {n}.", ("technical_content",)) for n in range(3, n_pages + 1)]
    s, report = ds.compose_document_summary(_doc(prs))
    assert ("conventions used for tolerances" in s) is expect_intro
    assert report["overview_types"] == (["abstract", "introduction", "conclusion"] if expect_intro
                                        else ["abstract"])


def test_contract_sos_blocking_client_parses_the_summary_key(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "contract_sos")
    doc = _big_doc(n_pages=10)
    report = ds.apply_document_summary(doc, "", _ReplyClient(_contract_answer("A bounded abstract.")))
    assert report["status"] == "ok" and doc["doc_summary"] == "A bounded abstract."


# ── contract_sos is hierarchical: blocks -> level summaries -> one root ─────

class _TreeClient:
    """Thread-safe streaming mock. Every call answers with a 5-key contract
    whose summary is ~360 chars and whose chunk list is long, so an early stop
    is observable. fail_every=N makes every Nth call return a null summary."""
    model = "shrew-ocr-preview"

    def __init__(self, fail_every=0, fail_root=False):
        import threading
        self.lock = threading.Lock()
        self.n = 0
        self.inputs = []
        self.generated = []
        self.fail_every, self.fail_root = fail_every, fail_root

    def chat_completion_stream(self, messages, *, max_tokens, temperature,
                               extra_params, on_delta, wall_clock_s, timeout=None):
        text = messages[1]["content"]
        with self.lock:
            self.n += 1
            k = self.n
            self.inputs.append(text)
        is_root_like = text.count("\n") < 10 and "Block" in text   # the root sees a few block summaries
        summ = (None if (self.fail_every and k % self.fail_every == 0) or
                (self.fail_root and is_root_like) else
                f"Block {k} summary: " + " ".join(f"topic{k}x{i}" for i in range(45)) + ".")
        answer = _contract_answer(summ)
        acc, reason = "", None
        for i in range(0, len(answer), 40):
            acc += answer[i:i + 40]
            reason = on_delta(answer[i:i + 40], acc)
            if reason:
                break
        with self.lock:
            self.generated.append((len(acc), len(answer)))
        return {"choices": [{"message": {"role": "assistant", "content": acc},
                             "finish_reason": reason or "stop"}]}


def _pages_3000():
    return "\n".join(
        f"Page {n} summary: " + " ".join(f"w{n}x{i}" for i in range(32)) + "."   # ~315 chars, the measured ratio
        for n in range(1, 3001))


def test_contract_sos_tree_on_3000_page_summaries():
    text = _pages_3000()
    client = _TreeClient()
    s, report = ds.summarize_via_contract(text, client, doc_id="book")
    shape = [(l["level"], l["inputs"], l["calls"], l["root"]) for l in report["levels"]]
    from app.structured_page import TEXT_PAGE_MAX_CHARS
    assert all(len(i) <= TEXT_PAGE_MAX_CHARS for i in client.inputs)   # every call in distribution
    assert shape[0][1] == 3000 and 80 <= shape[0][2] <= 95         # ~86 level-0 block calls
    assert shape[-1][2] == 1 and shape[-1][3] is True               # exactly one root call
    assert 2 <= len(shape) <= 3
    assert report["calls"] == sum(l["calls"] for l in report["levels"]) == client.n
    assert report["status"] == "ok" and s.startswith("Block ") and len(s) <= ds.DOC_SUMMARY_MAX_CHARS
    # every call stopped right after the summary key
    assert all(g < total / 10 for g, total in client.generated)


def test_contract_sos_drops_a_failed_block_and_logs(caplog):
    client = _TreeClient(fail_every=10)
    with caplog.at_level(logging.WARNING, logger="app.doc_summary"):
        s, report = ds.summarize_via_contract(_pages_3000(), client, doc_id="book")
    assert report["status"] == "ok" and s
    assert report["levels"][0]["dropped"] >= 8
    assert any("block dropped" in r.getMessage() for r in caplog.records)


def test_contract_sos_root_failure_is_null(caplog):
    client = _TreeClient(fail_root=True)
    with caplog.at_level(logging.ERROR, logger="app.doc_summary"):
        s, report = ds.summarize_via_contract(_pages_3000(), client, doc_id="book")
    assert s is None and report["status"] == "failed" and "root call failed" in report["error"]
    assert any("summary: null" in r.getMessage() for r in caplog.records)


def test_contract_sos_single_block_is_one_call():
    client = _TreeClient()
    text = "\n".join(f"Page {n} summary about item {n}." for n in range(1, 19))
    s, report = ds.summarize_via_contract(text, client)
    assert client.n == 1 and report["levels"] == [
        {**report["levels"][0], "level": 0, "inputs": 18, "calls": 1, "root": True}]


# ── modes as env flags + per-request override on /v1/convert ────────────────

def test_validators_share_the_enum_and_limits():
    assert ds.validate_mode(" Contract_SOS ") == "contract_sos"
    assert ds.validate_max_chars("1500") == 1500
    for bad in ("bogus", "", "join"):
        with pytest.raises(ValueError):
            ds.validate_mode(bad)
    for bad in ("abc", 5, 199, 20001, None):
        with pytest.raises(ValueError):
            ds.validate_max_chars(bad)


def test_invalid_env_value_falls_back_to_default_and_logs(monkeypatch, caplog):
    monkeypatch.setenv("SHREW_DOC_SUMMARY_MODE", "concat")
    with caplog.at_level(logging.ERROR, logger="app.doc_summary"):
        assert ds._env("SHREW_DOC_SUMMARY_MODE", "deterministic", ds.validate_mode) == "deterministic"
    assert any("ignored" in r.getMessage() for r in caplog.records)


def test_block_chars_env(monkeypatch):
    from app.structured_page import TEXT_PAGE_MAX_CHARS, context_limit_chars
    monkeypatch.delenv("SHREW_DOC_SUMMARY_BLOCK_CHARS", raising=False)
    assert ds._block_chars() == TEXT_PAGE_MAX_CHARS == 11000
    monkeypatch.setenv("SHREW_DOC_SUMMARY_BLOCK_CHARS", "6000")
    assert ds._block_chars() == 6000
    monkeypatch.setenv("SHREW_DOC_SUMMARY_BLOCK_CHARS", "99")
    assert ds._block_chars() == 11000                         # invalid -> default
    monkeypatch.setenv("SHREW_DOC_SUMMARY_BLOCK_CHARS", "10000000")
    assert ds._block_chars() == context_limit_chars()          # clamped to capacity
    # and the tree actually uses it: 6,000-char blocks -> roughly twice the level-0 calls
    monkeypatch.setenv("SHREW_DOC_SUMMARY_BLOCK_CHARS", "6000")
    s, report = ds.summarize_via_contract(_pages_3000(), _TreeClient())
    assert report["block_chars"] == 6000 and report["levels"][0]["calls"] >= 140


def test_per_call_mode_overrides_env(monkeypatch):
    monkeypatch.setattr(ds, "DOC_SUMMARY_MODE", "deterministic")
    doc = _big_doc(n_pages=10)
    report = ds.apply_document_summary(doc, "", _NoCallClient(), mode="off", max_chars=1000)
    assert report["status"] == "disabled" and doc["doc_summary"] is None


@pytest.fixture
def convert_client(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    import app.server as server
    from app.models import PipelineResult
    monkeypatch.setenv("SHREW_CONCURRENCY_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("SHREW_TELEMETRY_DIR", str(tmp_path / "tel"))
    monkeypatch.setattr(server, "_config", server.ServerConfig(
        vlm_url="http://fake", vlm_model="shrew-ocr-preview", workers=1,
        pipeline_concurrency=1, vlm_concurrency=4))
    monkeypatch.setattr(server, "_vlm_pool", ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(server, "_pipeline_gate", threading.Semaphore(1))
    seen = {}

    def fake_pipeline(path, out, cfg, **kw):
        seen.update(kw)
        return PipelineResult("# doc", {"metadata": {}, "summary": "s", "semantic_chunks": [],
                                        "tables": [], "images": []}, {"gates": {}})
    monkeypatch.setattr(server, "run_structured_pipeline", fake_pipeline)
    # the public build gates /v1/convert on inference readiness; stub the probe
    from tests.test_server_structured import FakeVLMClient
    monkeypatch.setattr("app.server.VLMClient", FakeVLMClient)
    return TestClient(server.app), seen


@pytest.mark.parametrize("endpoint", ["/v1/convert", "/v1/convert/stream"])
@pytest.mark.parametrize("field, value", [
    ("summary_mode", "concatenate"),
    ("summary_max_chars", "abc"),
    ("summary_max_chars", "50"),
    ("summary_max_chars", "1000000"),
])
def test_invalid_summary_override_is_400(convert_client, endpoint, field, value):
    client, seen = convert_client
    r = client.post(endpoint, files={"file": ("a.txt", b"hello world\n", "text/plain")},
                    data={field: value})
    assert r.status_code == 400, r.text
    assert "summary" in r.json()["detail"]
    assert not seen                                   # rejected before any pipeline work


def test_summary_override_reaches_the_pipeline(convert_client):
    client, seen = convert_client
    r = client.post("/v1/convert", files={"file": ("a.txt", b"hello world\n", "text/plain")},
                    data={"summary_mode": "contract_sos", "summary_max_chars": "1500"})
    assert r.status_code == 200, r.text
    assert seen["summary_mode"] == "contract_sos" and seen["summary_max_chars"] == 1500


def test_no_override_passes_none_so_env_default_applies(convert_client):
    client, seen = convert_client
    r = client.post("/v1/convert", files={"file": ("a.txt", b"hello world\n", "text/plain")})
    assert r.status_code == 200, r.text
    assert seen["summary_mode"] is None and seen["summary_max_chars"] is None


def test_root_prefix_is_off_by_default_and_only_on_a_level_above_zero():
    c = _TreeClient()
    ds.summarize_via_contract(_pages_3000(), c)
    assert not any(i.startswith("Section summaries") for i in c.inputs)
    c = _TreeClient()
    s, report = ds.summarize_via_contract(_pages_3000(), c,
                                          root_prefix="Section summaries of one document, in order:")
    starts = [i for i in c.inputs if i.startswith("Section summaries of one document, in order:")]
    assert len(starts) == 1 and report["root_prefix"]          # the root call only
    c = _TreeClient()                                           # single block: no level > 0, no prefix
    ds.summarize_via_contract("Page 1 about gears.\nPage 2 about cams.", c, root_prefix="X:")
    assert c.inputs == ["Page 1 about gears.\nPage 2 about cams."]
