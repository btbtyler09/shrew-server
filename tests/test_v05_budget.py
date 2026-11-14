"""v0.5 token budget: exact prompt counts, per-bucket image maxima, the 30,000
ceiling on every call path, and the runaway cut in output tokens."""
import base64
import importlib.util
import io
import json
import os
import random

import pytest
from PIL import Image

from app import contract as ct
from app import v05_tokens as rt
from app import v05_tools
from app import structured_page as sp
from app.structured_page import DecodePolicy

BOX = {"B0": (1152, 1152), "B1": (1152, 1536), "B2": (1536, 2304), "B3": (2304, 3072)}   # (w, h)
CKPT = os.environ.get("SHREW_V05_PROCESSOR_DIR", "")
HAVE_CKPT = (os.path.exists(os.path.join(CKPT, "processor_config.json"))
             and importlib.util.find_spec("transformers") is not None)
needs_ckpt = pytest.mark.skipif(not HAVE_CKPT, reason="needs transformers and an v0.5 checkpoint dir")

TAG_TEXTS = ["see <tables_otsl> here", "<tables_json>", "a <chart2csv> b <tables_html>", "<chart2summary>\nx",
             "<image> then <chart2code>", "<tables_html> then <image>", "<tables_jsonx> <table_otsl> < tables_otsl>"]
TEXTS = TAG_TEXTS + ["", "Hello world.\nSecond line", "  lead space", "汉字" * 800, "1,234.56 " * 400, "🙂" * 60,
         "<image> in text <|end_of_text|>", "x\n\n", "\ttab", "## 1 Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |"]


# --------------------------------------------------------------------------- image maxima

def test_bucket_maxima_are_reached_at_the_exact_fill_and_never_passed_inside_the_box():
    for name, (bw, bh) in BOX.items():
        assert ct.v05_image_tokens(bw, bh) == ct.V05_BUCKET_IMAGE_TOKENS[name]
        rnd = random.Random(name)
        sizes = [(rnd.randint(64, bw), rnd.randint(64, bh)) for _ in range(4000)]
        sizes += [(bw, h) for h in range(64, bh + 1, 7)] + [(w, bh) for w in range(64, bw + 1, 7)]
        assert max(ct.v05_image_tokens(w, h) for w, h in sizes) <= ct.V05_BUCKET_IMAGE_TOKENS[name]
    assert ct.V05_BUCKET_IMAGE_TOKENS == {"B0": 1476, "B1": 1920, "B2": 3672, "B3": 7152}


def test_no_image_of_any_size_costs_more_than_the_global_maximum():
    rnd = random.Random(0)
    sizes = [(rnd.randint(200, 6000), rnd.randint(200, 6000)) for _ in range(20000)]
    sizes += [(w, h) for w in range(200, 6001, 211) for h in range(200, 6001, 199)]
    assert max(ct.v05_image_tokens(w, h) for w, h in sizes) <= ct.V05_IMAGE_TOKENS_MAX == 7152


def test_a_page_just_outside_its_box_costs_the_next_bucket():
    """fit_to_bucket leaves an image within 2% of the box alone, so a "B1" page
    1175x1560 reaches the B2 pinpoint. This is why an unknown size is charged
    the global maximum and not the bucket's."""
    assert ct.v05_image_tokens(1175, 1560) > ct.V05_BUCKET_IMAGE_TOKENS["B1"]
    assert ct.v05_image_tokens(1556, 2078) > ct.V05_BUCKET_IMAGE_TOKENS["B2"]
    assert ct.V05_IMAGE_TOKENS_FALLBACK == ct.V05_IMAGE_TOKENS_MAX


@needs_ckpt
def test_bucket_maxima_match_the_real_processor():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    img_id = proc.tokenizer.convert_tokens_to_ids("<image>")
    for w, h in list(BOX.values()) + [(640, 623), (1023, 1336), (1086, 1607), (2301, 3037),
                                      (1175, 1560), (1556, 2078), (4544, 5996)]:
        got = int((proc(images=[Image.new("RGB", (w, h), "white")], text="<image>",
                        return_tensors="pt")["input_ids"][0] == img_id).sum())
        assert got == ct.v05_image_tokens(w, h), (w, h)


# --------------------------------------------------------------------------- tokenizer

def test_vendored_tokenizer_is_the_recorded_one_and_exact():
    rt.reset()
    st = rt.status()
    assert st["exact"] and st["sha256"] == rt.VENDORED_SHA256 and st["matches_served_checkpoint"]
    assert rt.image_prompt_tokens(ct.CMD_EXTRACT, 1) == 19           # 18 template tokens + the image
    assert rt.image_prompt_tokens(ct.CMD_EXTRACT, 7152) == 7170


def test_chars_per_token_is_not_3_5_for_cjk():
    text = "汉字" * 5000                                              # 10,000 chars
    assert rt.text_prompt_tokens(ct.CMD_EXTRACT, text) > 3 * len(text) / 3.5


@needs_ckpt
def test_prompt_counts_match_the_checkpoint_chat_template():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    rnd = random.Random(1)
    texts = TEXTS + ["".join(rnd.choice("abc 汉字\n1.<>|_é\t'") for _ in range(rnd.randint(0, 400)))
                     for _ in range(150)]
    for cmd in (ct.CMD_EXTRACT, ct.CMD_FIX_HEADINGS, ct.CMD_CLASSIFY_FIGURE):
        for t in texts:
            msgs = [{"role": "system", "content": cmd}, {"role": "user", "content": [{"type": "text", "text": t}]}]
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            assert s == rt.render(cmd, t)
            n_list = len(tok(s, add_special_tokens=False)["input_ids"])
            assert n_list == rt.count(rt.render(cmd, t))
            # vLLM's "string" content format: the template gets a plain string and indents it
            s2 = tok.apply_chat_template([msgs[0], {"role": "user", "content": t}], tokenize=False,
                                         add_generation_prompt=True)
            assert s2 == rt.render(cmd, t, string_format=True)
            n_str = len(tok(s2, add_special_tokens=False)["input_ids"])
            assert rt.text_prompt_tokens(cmd, t) == max(n_list, n_str) <= min(n_list, n_str) + 2
        msgs = [{"role": "system", "content": cmd}, {"role": "user", "content": [{"type": "image"}]}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert s == rt.render(cmd, rt.IMAGE_PLACEHOLDER)
        assert len(tok(s, add_special_tokens=False)["input_ids"]) == rt.image_prompt_tokens(cmd, 1)


def test_without_a_tokenizer_the_count_is_an_upper_bound(monkeypatch, tmp_path):
    exact = [rt.text_prompt_tokens(ct.CMD_EXTRACT, t) for t in TEXTS]
    exact_img = rt.image_prompt_tokens(ct.CMD_EXTRACT, 7152)
    monkeypatch.setenv("SHREW_V05_TOKENIZER", str(tmp_path / "missing.json"))
    rt.reset()
    try:
        st = rt.status()
        assert not st["exact"] and st["error"]
        for t, n in zip(TEXTS, exact):
            assert rt.text_prompt_tokens(ct.CMD_EXTRACT, t) >= n
        assert rt.image_prompt_tokens(ct.CMD_EXTRACT, 7152) >= exact_img
        plan = ct.v05_plan(ct.CMD_EXTRACT, text="汉字" * 2000)
        assert plan["total"] <= ct.V05_TOKEN_BUDGET and not plan["exact"]
        assert rt.status()["fallback_counts"] > 0
    finally:
        monkeypatch.delenv("SHREW_V05_TOKENIZER")
        rt.reset()


# --------------------------------------------------------------------------- the 30,000 ceiling

def test_plan_never_passes_the_budget():
    rnd = random.Random(2)
    plans = [ct.v05_plan(cmd, size=(rnd.randint(32, 6000), rnd.randint(32, 6000)), cap=cap)
             for cmd in (ct.CMD_EXTRACT, ct.CMD_CLASSIFY_FIGURE) for cap in (None, 8, 4096, 10**6)
             for _ in range(300)]
    plans += [ct.v05_plan(ct.CMD_EXTRACT, size=wh) for wh in BOX.values()]
    plans += [ct.v05_plan(ct.CMD_EXTRACT), ct.v05_plan(ct.CMD_CLASSIFY_FIGURE, cap=8)]     # size unknown
    for n in (0, 1, 500, 11000, 40000, 80000):
        for unit in ("word ", "汉", "9", "🙂", "é\n"):
            for cmd in (ct.CMD_EXTRACT, ct.CMD_FIX_HEADINGS):
                try:
                    plans.append(ct.v05_plan(cmd, text=(unit * n)[:n], cap=rnd.choice([None, 4096])))
                except ct.V05BudgetError:
                    assert rt.text_prompt_tokens(cmd, (unit * n)[:n]) >= ct.V05_TOKEN_BUDGET
    assert len(plans) > 2400
    for p in plans:
        assert p["prompt_tokens"] + p["max_tokens"] == p["total"] <= ct.V05_TOKEN_BUDGET == 30000
        assert p["max_tokens"] >= 1
    assert ct.v05_plan(ct.CMD_EXTRACT, size=(2304, 3072)) == {
        "prompt_tokens": 7170, "image_tokens": 7152, "max_tokens": 22830, "total": 30000, "exact": True}
    unknown = ct.v05_plan(ct.CMD_EXTRACT)
    assert unknown["image_tokens"] == 7152 and unknown["max_tokens"] == 22830 and not unknown["exact"]
    assert ct.v05_plan(ct.CMD_EXTRACT, size=(1152, 1152), cap=8000)["max_tokens"] == 8000


def test_a_prompt_that_fills_the_budget_is_refused_not_sent():
    text = "汉" * 40000
    with pytest.raises(ct.V05BudgetError):
        ct.v05_plan(ct.CMD_EXTRACT, text=text)
    client = _Recorder([])
    res = sp.extract_text_page(text, client, max_chars=10**9, decode=DecodePolicy.from_env(ct.V05))
    assert not res["ok"] and res["status"] == "oversize" and client.calls == []


class _Recorder:
    """A client that records every request and answers from a script."""
    model = "shrew-ocr-preview"

    def __init__(self, answers, usage_per_char=None):
        self.answers, self.calls, self.usage_per_char = list(answers), [], usage_per_char

    def _next(self, messages, max_tokens):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.answers.pop(0) if self.answers else json.dumps(_label())

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2, extra_params=None, timeout=None):
        return {"choices": [{"message": {"content": self._next(messages, max_tokens)}, "finish_reason": "stop"}]}

    def chat_completion_stream(self, messages, *, max_tokens, temperature=0.2, extra_params=None,
                               timeout=None, on_delta=None, wall_clock_s=None, on_usage=None):
        text, acc, reason = self._next(messages, max_tokens), "", None
        self.calls[-1]["asked_usage"] = on_usage is not None
        for i in range(0, len(text), 50):
            acc += text[i:i + 50]
            reason = on_delta(text[i:i + 50], acc)
            if not reason and on_usage is not None and self.usage_per_char:
                reason = on_usage({"prompt_tokens": 19, "completion_tokens": int(len(acc) * self.usage_per_char)}, acc)
            if reason:
                break
        return {"choices": [{"message": {"role": "assistant", "content": acc}, "finish_reason": reason or "stop"}]}


def _label():
    return {"metadata": {"title": "T", "authors": None, "organization": None, "year": 2024, "doc_type": "Report"},
            "summary": "A short page about sales by year, with one chart and one table.",
            "semantic_chunks": [{"chunk_id": "1", "title": "Intro", "content": "## 1 Introduction\n\nSales grew.",
                                 "keywords": ["sales"], "section_type": "introduction"}],
            "figures": [], "tables": []}


def _request_total(call) -> int:
    """prompt + max_tokens of a recorded request, recounted from the messages alone."""
    cmd, part = call["messages"][0]["content"], call["messages"][1]["content"][0]
    if part["type"] == "text":
        prompt = rt.text_prompt_tokens(cmd, part["text"])
    else:
        raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        prompt = rt.image_prompt_tokens(cmd, ct.v05_image_tokens(*Image.open(io.BytesIO(raw)).size))
    return prompt + call["max_tokens"]


def test_every_v05_call_path_stays_within_30000(tmp_path, monkeypatch):
    from app import doc_summary
    decode = DecodePolicy.from_env(ct.V05)
    client = _Recorder([])
    for i, (w, h) in enumerate([(1152, 1152), (1152, 1536), (1536, 2304), (2304, 3072), (1175, 1560),
                                (300, 2500), (3000, 400)]):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (w, h), "white").save(p)
        sp.extract_page(str(p), client, decode=decode, bucket="B1")                     # page / composite
        sp.extract_page(str(p), client, decode=decode, bucket=None, max_tokens=10**6)   # a caller's own number
    for text in ("Some text.", "word " * 2200, "汉字\n" * 700, "9 " * 1500):
        sp.extract_text_page(text, client, decode=decode)                               # text page
        doc_summary._contract_call(text, client, ct.V05)                                # document summary
    crop = tmp_path / "fig.png"
    Image.new("RGB", (900, 500), "white").save(crop)
    client.answers = ["figure"]
    v05_tools._classify_one({"crop_path": str(crop), "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]], "figure_id": "f1"},
                            client, decode)                                             # classify_figure
    doc = {"chunks": [{"content": f"## {i} 标题 number {i}\n\nBody text."} for i in range(1, 40)]}
    assert len(v05_tools.heading_lines(doc)) == 39
    v05_tools.heading_fix(doc, client, decode=decode)                                   # fix_headings
    assert len(client.calls) == 14 + 8 + 1 + 2
    cmds = {c["messages"][0]["content"] for c in client.calls}
    assert cmds == {ct.CMD_EXTRACT, ct.CMD_CLASSIFY_FIGURE, ct.CMD_FIX_HEADINGS}
    for c in client.calls:
        assert _request_total(c) <= 30000, (c["messages"][0]["content"], c["max_tokens"])
    assert max(_request_total(c) for c in client.calls) == 30000        # the budget is used, not undershot


def test_unreadable_image_is_charged_the_maximum(tmp_path):
    before = ct.v05_budget_status()["calls_with_unknown_image_size"]
    plan = ct.v05_plan(ct.CMD_EXTRACT, size=None)
    assert plan["image_tokens"] == 7152
    assert ct.v05_budget_status()["calls_with_unknown_image_size"] == before + 1


# --------------------------------------------------------------------------- text-page bound in tokens

def test_text_pages_are_split_at_the_token_bound():
    bound = ct.V05_TEXT_PAGE_MAX_TOKENS
    assert bound == 3500
    cjk = "\n\n".join("第%d段。" % i + "这是一个关于水利工程建设的报道，内容涉及多个省份的项目进展。" * 12 for i in range(40))
    assert len(cjk) < 2 * sp.TEXT_PAGE_MAX_CHARS and rt.text_tokens(cjk) > 3 * bound
    old = sp.paginate_text(cjk)
    new = sp.paginate_text(cjk, max_tokens=bound)
    assert len(old) == 2 and len(new) > len(old)
    assert all(rt.text_tokens(p) <= bound and len(p) <= sp.TEXT_PAGE_MAX_CHARS for p in new)
    assert "\n\n".join(new) == cjk                                   # nothing lost, nothing cut mid-line
    para = ("The committee reviewed the annual report and approved the budget for the next fiscal year. "
            "Members asked for a detailed schedule of the remaining construction work. ")
    english = "\n\n".join(f"Section {i}. " + para * 4 for i in range(40))   # ~5 chars per token: chars bind first
    assert len(english) > 2 * sp.TEXT_PAGE_MAX_CHARS
    assert sp.paginate_text(english, max_tokens=bound) == sp.paginate_text(english)
    numbers = "\n".join(",".join(str((i * 7919 + j * 104729) % 100000) for j in range(12)) for i in range(300))
    pages = sp.paginate_text(numbers, max_tokens=bound)
    assert len(pages) > len(sp.paginate_text(numbers)) and all(rt.text_tokens(p) <= bound for p in pages)
    assert "\n".join(pages) == numbers
    assert sp.paginate_text(cjk, max_tokens=None) == old               # v0.4: unchanged


def test_an_over_long_line_is_split_not_refused():
    """Text extracted from a PDF often arrives as one long line."""
    bound = ct.V05_TEXT_PAGE_MAX_TOKENS
    sentence = "The contractor shall deliver the equipment within thirty days of the award; late delivery is penalised. "
    cases = {
        "english sentences": sentence * 400,                              # 41,600 chars, one line
        "cjk sentences": "本报讯，水利工程建设项目进展顺利。各省份将继续加大投入！" * 600,
        "no sentence ends": "alpha beta gamma delta epsilon zeta eta theta " * 900,
        "no whitespace at all": "x9Zq" * 9000,
        "cjk without marks": "汉字报道" * 3000,
    }
    for name, line in cases.items():
        assert "\n" not in line
        stats = {}
        pages = sp.paginate_text(line, max_tokens=bound, stats=stats)
        assert len(pages) > 1 and stats == {"split_oversize_line": 1}, name
        assert "".join(pages) == line.strip(), name                      # nothing dropped, nothing rewritten
        assert all(rt.text_tokens(p) <= bound and len(p) <= sp.TEXT_PAGE_MAX_CHARS for p in pages), name
    pages = sp.paginate_text(cases["english sentences"], max_tokens=bound)
    assert all(p.rstrip().endswith((".", ";")) for p in pages)             # cut at sentence ends
    pages = sp.paginate_text(cases["no sentence ends"], max_tokens=bound)
    assert all(p.endswith(" ") or p is pages[-1] for p in pages)           # cut at whitespace, never inside a word
    # every page of such a document is sent; none is refused
    client = _Recorder([])
    for i, page in enumerate(sp.paginate_text(cases["cjk sentences"], max_tokens=bound)):
        assert sp.extract_text_page(page, client, decode=DecodePolicy.from_env(ct.V05))["status"] != "oversize"
    assert len(client.calls) == i + 1
    # v0.4 keeps its old behaviour: the line is one page and is filtered downstream
    assert sp.paginate_text(cases["english sentences"]) == [cases["english sentences"].strip()]


def test_split_by_tokens_is_exact_and_bounded():
    for text in ("x9Zq" * 5000, "汉字报道" * 2500, "🙂" * 3000, "a", ""):
        pieces = rt.split_by_tokens(text, 500)
        assert "".join(pieces) == text and all(0 < rt.text_tokens(p) <= 500 for p in pieces)


def test_embedded_data_uri_payload_is_not_paginated_into_the_model():
    import base64 as b64
    blob = b64.b64encode(bytes(range(256)) * 300).decode()                 # ~100 KB of base64
    md = f"## 2.2 Results\n\nThe projector decomposes the signal.\n\n![Image](data:image/png;base64,{blob})\n\nFigure 3 shows the branches."
    stats = {}
    pages = sp.paginate_text(md, max_tokens=ct.V05_TEXT_PAGE_MAX_TOKENS, stats=stats)
    assert stats == {"data_uri_payloads_removed": 1} and len(pages) == 1
    assert "![Image](data:image/png;base64,[payload removed])" in pages[0] and "Figure 3 shows" in pages[0]
    assert len(sp.paginate_text(md)) == 3 and blob in "".join(sp.paginate_text(md))      # v0.4: untouched
    assert sp.strip_data_uris("see data:text/plain;base64,QUJD and more") == ("see data:text/plain;base64,QUJD and more", 0)


def test_a_direct_call_over_the_token_bound_is_still_reported_not_truncated():
    line = "汉字报道" * 2000                                            # not paginated by the caller
    client = _Recorder([])
    res = sp.extract_text_page(line, client, decode=DecodePolicy.from_env(ct.V05))
    assert not res["ok"] and res["status"] == "oversize" and "token" in res["error"] and client.calls == []
    res = sp.extract_text_page(line, _Recorder([]), decode=DecodePolicy.from_env("v0.4"))
    assert res["status"] != "oversize"                                 # v0.4 keeps its character bound only


def test_text_page_at_the_bound_fits_the_budget_with_its_cut():
    assert ct.V05_TEXT_PAGE_MAX_TOKENS + 18 + ct.V05_OUTPUT_TOKEN_CUT["text"] <= ct.V05_TOKEN_BUDGET


def test_summary_blocks_respect_the_token_bound():
    from app import doc_summary
    items = ["第%d页：" % i + "水利工程建设进展报道。" * 20 for i in range(60)]
    blocks = doc_summary._pack_blocks(items, 11000, ct.V05_TEXT_PAGE_MAX_TOKENS)
    assert len(blocks) > len(doc_summary._pack_blocks(items, 11000))
    assert all(rt.text_tokens(b) <= ct.V05_TEXT_PAGE_MAX_TOKENS for b in blocks)
    assert "\n".join(blocks) == "\n".join(items)
    assert doc_summary._pack_blocks(items, 11000, None) == doc_summary._pack_blocks(items, 11000)


# --------------------------------------------------------------------------- runaway cut in tokens

CUT = {"B0": 5000, "B1": 10000, "B2": 27000, "B3": 24000, "text": 5000}


def test_token_cut_lookup(monkeypatch):
    monkeypatch.setattr(ct, "V05_OUTPUT_TOKEN_CUT", dict(CUT))
    assert ct.v05_output_token_cut("text") == 5000
    assert ct.v05_output_token_cut("image", "B0", 1328) == 5000
    assert ct.v05_output_token_cut("image", "B1", 1900) == 10000
    assert ct.v05_output_token_cut("image", "B1", 3672) == 27000      # named B1, tokenized as B2: never cut a real page
    assert ct.v05_output_token_cut("image", "dpi", 1188) == 27000     # unknown bucket: the largest, whatever the cost
    assert ct.v05_output_token_cut("image", None, 732) == 27000       # a wide B2 strip costs less than a B0 page
    assert ct.v05_output_token_cut("image", None, None) == 27000      # nothing known: the largest
    monkeypatch.setattr(ct, "V05_OUTPUT_TOKEN_CUT", {k: 0 for k in CUT})
    assert ct.v05_output_token_cut("text") is None and ct.v05_output_token_cut("image", "B1", 10) is None


def test_guard_cuts_on_reported_tokens_and_not_on_characters():
    g = sp.RepetitionGuard(**{**sp.v05_guard_params(), "token_cap": 5000})
    assert g.usage({"prompt_tokens": 19, "completion_tokens": 4999}, "x" * 100) is None
    assert g.usage({"completion_tokens": 5000}, "x" * 150) is None      # a page may end exactly on the cut
    assert g.usage({"completion_tokens": 5001}, "x" * 200) == sp.REPETITION_ABORT
    st = g.stats()
    assert st["abort_guard"] == "token_cap" and st["completion_tokens"] == 5001 and st["prompt_tokens"] == 19
    ev = st["abort_evidence"]
    assert ev["guard"] == "token_cap" and ev["score"] == 5001 and ev["threshold"] == 5000 and ev["at_chars"] == 200
    _, verdict, err = sp._gate("x", sp.REPETITION_ABORT, g, ct.V05)
    assert verdict == "degenerate" and err.startswith("repetition_abort: token_cap guard 5001 ")


def _prose(n, seed):
    rnd = random.Random(seed)
    words = ["".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(3, 9))) for _ in range(3000)]
    out, size = [], 0
    while size < n:
        w = rnd.choice(words)
        out.append(w)
        size += len(w) + 1
    return " ".join(out)[:n]


def _stream(text, usage_per_char, **over):
    g = sp.RepetitionGuard(**{**sp.v05_guard_params(), **over})
    for i in range(0, len(text), 50):
        acc = text[:i + 50]
        if g(text[i:i + 50], acc) or (usage_per_char and g.usage({"completion_tokens": int(len(acc) * usage_per_char)}, acc)):
            return g, len(acc)
    return g, None


def test_character_cap_is_only_the_fallback():
    text = _prose(70000, 3)                                   # never repeats, never stops
    g, at = _stream(text, 0.25, token_cap=25000)              # usage reported, 17,500 tokens: under the cut
    assert at is None and g.usage_seen                        # the 60,000-char cap did not fire
    g, at = _stream(text, 0.25, token_cap=12000)
    assert g.abort_guard == "token_cap" and 48000 <= at <= 48100
    g, at = _stream(text, None, token_cap=12000)              # the server sends no usage
    assert g.abort_guard == "length_cap" and not g.usage_seen and 60000 <= at < 61000
    g, at = _stream(_prose(36000, 6), 1.5, token_cap=25000)   # 36,000 chars = 54,000 tokens: chars never saw it
    assert g.abort_guard == "token_cap" and at < 17000


def test_call_asks_for_usage_counts_fallbacks_and_never_cuts_above_max_tokens(monkeypatch):
    monkeypatch.setattr(ct, "V05_OUTPUT_TOKEN_CUT", dict(CUT))
    monkeypatch.setattr(sp, "_V05_CUT", {k: 0 for k in sp._V05_CUT})
    decode = DecodePolicy.from_env(ct.V05)
    runaway = json.dumps({"content": _prose(40000, 4)})
    client = _Recorder([runaway, json.dumps(_label())], usage_per_char=0.25)       # 10,000 tokens > text cut 5,000
    res = sp.extract_text_page("Some text.", client, decode=decode)
    first = res["trail"][0]
    assert client.calls[0]["asked_usage"]
    assert first["abort_guard"] == "token_cap" and first["token_cap"] == 5000
    assert 5000 <= first["completion_tokens"] < 5100 and first["prompt_tokens"] == 19
    assert res["ok"] and res["trail"][1]["abort_guard"] is None
    st = sp.v05_runaway_cut_status()
    assert st["configured"] == "tokens" and st["observed"] == "tokens"
    assert st["streams_with_usage"] == 2 and st["token_cut_aborts"] == 1
    # the cut is never above the call's max_tokens
    client = _Recorder([json.dumps(_label())], usage_per_char=0.25)
    sp.extract_text_page("Some text.", client, max_tokens=1200, decode=decode)
    assert client.calls[0]["max_tokens"] == 1200
    # a server without usage: character cap, counted
    silent = _Recorder([json.dumps({"content": _prose(70000, 5)}), json.dumps(_label())])
    res = sp.extract_text_page("Some text.", silent, decode=decode)
    assert res["trail"][0]["abort_guard"] == "length_cap" and res["trail"][0]["completion_tokens"] is None
    st = sp.v05_runaway_cut_status()
    assert st["observed"] == "mixed" and st["streams_without_usage"] == 2
    assert st["char_cap_aborts_without_usage"] == 1


def test_stream_usage_switch_turns_the_token_cut_off(monkeypatch):
    monkeypatch.setattr(ct, "V05_OUTPUT_TOKEN_CUT", dict(CUT))
    monkeypatch.setattr(sp, "V05_STREAM_USAGE", False)
    client = _Recorder([json.dumps(_label())], usage_per_char=0.25)
    sp.extract_text_page("Some text.", client, decode=DecodePolicy.from_env(ct.V05))
    assert not client.calls[0]["asked_usage"]
    assert sp.v05_runaway_cut_status()["configured"] == "chars"


def test_v04_is_untouched_by_the_budget_and_the_cut(monkeypatch):
    monkeypatch.setattr(ct, "V05_OUTPUT_TOKEN_CUT", dict(CUT))
    client = _Recorder([json.dumps({k: v for k, v in _label().items()})], usage_per_char=0.25)
    sp.extract_text_page("Some text.", client, decode=DecodePolicy.from_env("v0.4"))
    assert client.calls[0]["max_tokens"] == sp.MAX_TOKENS and not client.calls[0]["asked_usage"]
    assert set(sp.RepetitionGuard().stats()) == {"aborted", "ratio", "max_window_ratio", "position_chars",
                                                 "threshold", "window_chars"}


# --------------------------------------------------------------------------- edge cases

def test_template_task_tags_are_counted_as_the_template_serves_them():
    """The chat template replaces a user message holding a task tag by a canned
    prompt; the count has to be of that prompt, or prompt + max_tokens passes 30,000."""
    plain = rt.text_prompt_tokens(ct.CMD_EXTRACT, "see tables_otsl here")
    served = rt.text_prompt_tokens(ct.CMD_EXTRACT, "see <tables_otsl> here")
    assert served > plain + 300
    assert rt.template_tag("a <chart2csv> b <tables_html>") == "chart2csv"          # the template's order
    assert rt.template_tag("<tables_jsonx> < tables_otsl>") is None
    assert rt.expand_tags("<image> then <chart2code>").startswith("<image>\n")
    assert rt.expand_tags("<tables_html> then <image>").endswith("<image>\n")
    for t in TAG_TEXTS:
        p = ct.v05_plan(ct.CMD_EXTRACT, text=t)
        assert p["prompt_tokens"] == rt.text_prompt_tokens(ct.CMD_EXTRACT, t) and p["total"] <= ct.V05_TOKEN_BUDGET
        assert p["prompt_tokens"] >= rt.count(rt.render(ct.CMD_EXTRACT, t, string_format=True))


def test_text_prompt_is_charged_the_string_format_rendering():
    """Live vLLM (2026-11-14) served a text page counted 777 as 778: it passes text
    to the template as a plain string, which the template indents by eight spaces."""
    t = "# Report\n\nRevenue grew 4.2% in 2025.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n" * 20
    assert rt.count(rt.render(ct.CMD_EXTRACT, t)) == 777
    assert rt.text_prompt_tokens(ct.CMD_EXTRACT, t) == 778 == ct.v05_plan(ct.CMD_EXTRACT, text=t)["prompt_tokens"]


def test_template_tag_prompts_are_the_checkpoint_templates():
    path = os.path.join(CKPT, "chat_template.jinja")
    if not os.path.exists(path):
        pytest.skip("needs an v0.5 checkpoint dir")
    import hashlib
    import re
    src = open(path, encoding="utf-8").read()
    tags = json.loads(rt.TEMPLATE_TAGS.read_text(encoding="utf-8"))
    assert hashlib.sha256(src.encode("utf-8")).hexdigest() == tags["chat_template_sha256"]
    assert re.findall(r'(?:if|elif) "<(\w+)>" in text -%}\n  \{\{-', src) == tags["order"]
    for k in tags["order"]:
        assert json.loads(re.search(r"set %s_prompt = (\".*?\") -%%}\n" % k, src, re.S).group(1)) == tags["prompts"][k]


def test_data_uri_stripping_takes_the_payload_and_nothing_else():
    b64 = base64.b64encode(os.urandom(300)).decode()
    after = "\n\nRevenue grew in all regions including North America\n\nNext section."
    out, n = sp.strip_data_uris("Logo: data:image/png;base64," + b64 + after)
    assert n == 1 and out == "Logo: data:image/png;base64," + sp.DATA_URI_MARKER + after
    prose = ("The header is data:text/plain;base64, and after the comma the encoded bytes follow " * 8) + "markup"
    assert sp.strip_data_uris(prose) == (prose, 0)
    spaced = "data:image/png;base64," + " ".join(["word"] * 100)
    assert sp.strip_data_uris(spaced) == (spaced, 0)
    md = "a ![Image](data:image/png;base64," + b64 + ") b"
    assert sp.strip_data_uris(md) == ("a ![Image](data:image/png;base64," + sp.DATA_URI_MARKER + ") b", 1)


class _SSE:
    """A fake requests response streaming vLLM-style chunks."""
    def __init__(self, events):
        self.events, self.status_code = events, 200
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def raise_for_status(self):
        pass
    def iter_lines(self, **kw):
        for e in self.events:
            yield "data: " + json.dumps(e)
        yield "data: [DONE]"


def _sse(text, total, finish, final_only=False):
    n = max(1, len(text) // 40)
    chunks = [text[i:i + n] for i in range(0, len(text), n)]
    ev = []
    for k, c in enumerate(chunks):
        e = {"choices": [{"delta": {"content": c}, "finish_reason": None}]}
        if not final_only:
            e["usage"] = {"prompt_tokens": 19, "completion_tokens": max(1, total * (k + 1) // len(chunks))}
        ev.append(e)
    e = {"choices": [{"delta": {}, "finish_reason": finish}]}
    if not final_only:
        e["usage"] = {"prompt_tokens": 19, "completion_tokens": total}
    ev.append(e)
    ev.append({"choices": [], "usage": {"prompt_tokens": 19, "completion_tokens": total}})
    return ev


def _real_client(monkeypatch, streams):
    import requests
    from app.vlm_client import VLMClient
    it = iter(streams)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _SSE(next(it)))
    return VLMClient("http://unused.invalid/v1", "m")


def test_a_page_that_stops_exactly_at_max_tokens_is_kept(monkeypatch):
    monkeypatch.setattr(sp, "_V05_CUT", {k: 0 for k in sp._V05_CUT})
    decode = DecodePolicy.from_env(ct.V05)
    client = _real_client(monkeypatch, [_sse(json.dumps(_label()), 900, "stop")])
    res = sp.extract_text_page("Some text.", client, max_tokens=900, decode=decode)
    assert res["ok"] and res["trail"][0]["abort_guard"] is None and len(res["trail"]) == 1
    # a page the server ended at max_tokens is "length", not a token-cut abort
    client = _real_client(monkeypatch, [_sse('{"content": "' + _prose(3000, 8), 900, "length")] * 4)
    res = sp.extract_text_page("Some text.", client, max_tokens=900, decode=decode)
    assert all(t["abort_guard"] is None for t in res["trail"])
    assert sp.v05_runaway_cut_status()["token_cut_aborts"] == 0


def test_usage_only_at_the_end_is_not_reported_as_the_token_cut(monkeypatch):
    monkeypatch.setattr(sp, "_V05_CUT", {k: 0 for k in sp._V05_CUT})
    client = _real_client(monkeypatch, [_sse(json.dumps(_label()), 300, "stop", final_only=True)])
    res = sp.extract_text_page("Some text.", client, decode=DecodePolicy.from_env(ct.V05))
    st = sp.v05_runaway_cut_status()
    assert res["ok"] and st["observed"] == "chars" and st["streams_without_usage"] == 1


def test_embedded_media_cut_follows_the_bucket_not_the_image_cost():
    assert ct.v05_output_token_cut("image", "B2", 732) == ct.V05_OUTPUT_TOKEN_CUT["B2"]
    import inspect
    from app import structured_pipeline as pl
    assert "extract_page(prepped, client, decode=decode, doc_id=doc_id, bucket=_bucket)" in inspect.getsource(pl)


def test_split_pieces_are_checked_by_their_exact_count():
    line = "ab…'s！" * 3000
    pieces = sp._split_long_line(line, 11000, 3500)
    assert "".join(pieces) == line
    assert all(len(p) <= 11000 and rt.text_tokens(p) <= 3500 for p in pieces)
    rnd = random.Random(11)
    for bound in (100, 300):
        for _ in range(40):
            s = "".join(rnd.choice(["…", "'s", "！", "ab", " ", "。", "x.", "汉", "'", "’t"]) for _ in range(rnd.randint(200, 3000)))
            ps = sp._split_long_line(s, 11000, bound)
            assert "".join(ps) == s and all(rt.text_tokens(p) <= bound for p in ps)


def test_a_malformed_cut_env_does_not_stop_the_server():
    import subprocess
    import sys
    code = ("from app import contract as ct; import json; "
            "print(json.dumps([ct.V05_OUTPUT_TOKEN_CUT['B3'], ct.v05_budget_status()['output_token_cut_env_error']]))")

    def run(value):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={**os.environ, "SHREW_V05_OUTPUT_TOKEN_CUT": value})
        return json.loads(out.stdout.strip().splitlines()[-1])

    for bad in ("B2=20000", '{"b3": 1000}', '{"B3": "x"}'):
        b3, err = run(bad)
        assert b3 == 26500 and err
    assert run('{"B3": 20000}') == [20000, None]
