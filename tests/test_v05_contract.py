"""v0.5 contract: commands as prompts, quad boxes, greedy budget decode,
figure_check / heading_fix — and v0.4 untouched.

"""
import base64
import importlib.util
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app import contract as ct
from app import structured_page as sp
from app.structured_page import DecodePolicy

HERE = Path(__file__).parent
PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC")

QUAD_T = [[100, 520], [900, 520], [900, 880], [100, 880]]
QUAD_F1 = [[100, 100], [480, 100], [480, 450], [100, 450]]
QUAD_F2 = [[520, 100], [900, 100], [900, 450], [520, 450]]
CHART = {"header": ["Year", "Sales"], "rows": [["2023", "4.1"], ["2024", "5.0"]]}


def v05_label(**over):
    lab = {
        "metadata": {"title": "T", "authors": None, "organization": None, "year": 2024,
                     "doc_type": "Report"},
        "summary": "A short page about sales by year, with one chart and one table.",
        "semantic_chunks": [
            {"chunk_id": "1", "title": "Intro", "content": "## 1 Introduction\n\nSales grew.",
             "keywords": ["sales"], "section_type": "introduction"},
            {"chunk_id": "2", "title": "Scope", "content": "## 1.1 Scope\n\nOnly 2023-2024.",
             "keywords": ["scope"], "section_type": "technical_content"},
        ],
        "figures": [
            {"bbox": QUAD_F1, "caption": None, "description": "A company logo."},
            {"bbox": QUAD_F2, "caption": "Figure 1. Sales.", "description": "Bar chart of sales.",
             "chart_data": CHART},
        ],
        "tables": [{"bbox": QUAD_T, "caption": "Table 1.", "html": "<table><tr><td>a</td></tr></table>",
                    "description": "One-cell table."}],
    }
    lab.update(over)
    return lab


# --------------------------------------------------------------------------- image tokens

@pytest.mark.parametrize("size, tokens", [
    # Verified against the real Granite4VisionProcessor (a v0.5 checkpoint):
    # 396 fitted sizes, 0 mismatches (2026-11-09). The first six were in that set
    # explicitly; 1328 (B0) and 3528 (B2) are training's own measured values.
    ((1152, 1152), 1476), ((1152, 1536), 1920), ((1536, 2304), 3672),
    ((2304, 3072), 7152), ((850, 1100), 1188), ((1700, 2200), 7006),
    ((1100, 1000), 1328), ((1490, 2304), 3528),
])
def test_image_tokens_pinned(size, tokens):
    assert ct.v05_image_tokens(*size) == tokens


def _v05_processor_dir():
    d = os.environ.get("SHREW_V05_PROCESSOR_DIR", "")
    return d if os.path.exists(os.path.join(d, "processor_config.json")) else None


@pytest.mark.skipif(_v05_processor_dir() is None or importlib.util.find_spec("transformers") is None,
                    reason="needs transformers and an v0.5 checkpoint dir")
def test_image_tokens_match_real_processor():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(_v05_processor_dir(), trust_remote_code=True)
    img_id = proc.tokenizer.convert_tokens_to_ids("<image>")
    for w, h in [(1152, 1152), (1152, 1536), (1100, 1000), (1490, 2304), (2304, 2980), (600, 2500)]:
        got = int((proc(images=[Image.new("RGB", (w, h), "white")], text="<image>",
                        return_tensors="pt")["input_ids"][0] == img_id).sum())
        assert got == ct.v05_image_tokens(w, h), (w, h)


def test_max_tokens_is_budget_minus_prompt():
    # 18 template tokens + the image tokens; the full matrix is tests/test_v05_budget.py
    assert ct.v05_max_tokens(size=(2304, 3072)) == 30000 - 7152 - 18
    assert ct.v05_max_tokens() == 30000 - 7152 - 18                 # size unknown -> the global maximum
    assert ct.v05_max_tokens(size=(2304, 3072), cap=8000) == 8000


# --------------------------------------------------------------------------- messages

def test_v05_messages_are_bare_commands(tmp_path):
    img = tmp_path / "p.png"
    img.write_bytes(PNG_1x1)
    page = sp.build_image_messages(str(img), ct.V05)
    assert page[0] == {"role": "system", "content": "structured_extraction"}
    assert page[1]["role"] == "user" and len(page[1]["content"]) == 1
    part = page[1]["content"][0]
    assert part["type"] == "image_url" and set(part["image_url"]) == {"url"}      # no "detail"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert sp.build_text_messages("hello", ct.V05) == [
        {"role": "system", "content": "structured_extraction"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    fig = sp.build_command_messages(ct.CMD_CLASSIFY_FIGURE, image_path=str(img))
    assert fig[0]["content"] == "classify_figure" and fig[1]["content"][0]["type"] == "image_url"
    hd = sp.build_command_messages(ct.CMD_FIX_HEADINGS, text="## a\n## b")
    assert hd == [{"role": "system", "content": "fix_headings"},
                  {"role": "user", "content": [{"type": "text", "text": "## a\n## b"}]}]


def test_v04_messages_unchanged():
    assert sp.build_text_messages("hello") == [
        {"role": "system", "content": "structured_extraction"},
        {"role": "user", "content": "hello"}]


# --------------------------------------------------------------------------- schema

def test_v04_gate_rejects_quads_which_is_why_v040_failed_v05():
    _, verdict, err = sp._gate(json.dumps(v05_label()), "stop")
    assert verdict == "schema" and "bad-bbox" in err


def test_v05_gate_accepts_quads_and_keeps_new_fields():
    parsed, verdict, err = sp._gate(json.dumps(v05_label()), "stop", contract=ct.V05)
    assert verdict == "ok", err
    assert parsed["figures"][1]["chart_data"] == CHART
    assert parsed["tables"][0]["bbox"] == QUAD_T


def test_v05_gate_repairs_flat_box_and_bad_chart_data():
    lab = v05_label()
    lab["figures"][0]["bbox"] = [100, 100, 480, 450]
    lab["figures"][1]["chart_data"] = {"header": ["a"], "rows": [["1", "2"]]}    # ragged
    parsed, verdict, _ = sp._gate(json.dumps(lab), "stop", contract=ct.V05)
    assert verdict == "ok"
    assert parsed["figures"][0]["bbox"] == QUAD_F1
    assert "chart_data" not in parsed["figures"][1]


def test_v05_gate_null_table_box_ok_bad_quad_rejected():
    lab = v05_label()
    lab["tables"][0]["bbox"] = None
    assert sp._gate(json.dumps(lab), "stop", contract=ct.V05)[1] == "ok"
    lab["tables"][0]["bbox"] = [[1, 2], [3, 4], [5, 6]]
    assert sp._gate(json.dumps(lab), "stop", contract=ct.V05)[1] == "schema"


@pytest.mark.parametrize("cd, ok", [
    (CHART, True),
    ({**CHART, "values": "estimated_from_graphic", "precision": "0.5"}, True),
    ({**CHART, "values": "estimated_from_graphic"}, False),
    ({**CHART, "precision": 1}, False),
    ({"header": [], "rows": []}, False),
    (None, False),
])
def test_chart_data_rule(cd, ok):
    assert ct.chart_data_ok(cd) is ok


def test_envelope():
    assert ct.envelope(QUAD_F1) == [100, 100, 480, 450]
    assert ct.envelope([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert ct.envelope(None) is None
    rot = [[500, 100], [600, 200], [500, 300], [400, 200]]
    assert ct.envelope(rot) == [400, 100, 600, 300]


def test_crop_by_quad():
    from app.assembly import crop_bbox, pad_bbox, should_stitch_tables
    im = Image.new("RGB", (1000, 2000), "white")
    c = crop_bbox(im, QUAD_F1)
    assert c.size == (380, 700)
    assert pad_bbox(QUAD_F1, 10) == [90, 90, 490, 460]
    prev = {"bbox": [[0, 600], [1000, 600], [1000, 990], [0, 990]], "html": "<table><tr><td>a</td><td>b</td></tr></table>"}
    nxt = {"bbox": [[0, 10], [1000, 10], [1000, 400], [0, 400]], "html": "<table><tr><td>c</td><td>d</td></tr></table>", "caption": None}
    assert should_stitch_tables(prev, nxt)


# --------------------------------------------------------------------------- decode

class _Rec:
    """Blocking client: records every call, answers from a script."""
    model = "shrew-ocr-preview"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat_completion(self, messages, max_tokens=None, temperature=None, timeout=None,
                        extra_params=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "extra_params": extra_params})
        text, fin = self.replies.pop(0)
        return {"choices": [{"message": {"content": text}, "finish_reason": fin}]}


def _sampling_view(call):
    """The sampling params of one recorded call: everything but messages/max_tokens,
    with the seed reduced to its presence (its VALUE is derived from the input,
    and v0.5 vs v0.4 inputs differ by construction)."""
    x = dict(call["extra_params"] or {})
    has_seed = isinstance(x.pop("seed", None), int)
    return {"temperature": call["temperature"], "extra": x, "seeded": has_seed}


def test_v05_and_v04_send_identical_sampling_params(tmp_path, monkeypatch):
    """Greedy decoding is never a production default. v0.5 decodes
    with v0.4.0's shipped recipe: first pass 0.4 / pp 0.3 / fixed seed; retry
    un-enforced 0.4 / pp 0.3 / fixed seed. Identical to v0.4, call for call."""
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", False)
    for k in ("SHREW_TRY1_TEMP", "SHREW_TRY2_TEMP", "SHREW_SEED_POLICY", "SHREW_TRY2_ENFORCE"):
        monkeypatch.delenv(k, raising=False)
    img = tmp_path / "p.png"
    Image.new("RGB", (1152, 1536), "white").save(img)
    views = {}
    for contract, good in (("v0.4", json.dumps(v05_label(figures=[], tables=[]))),
                           ("v0.5", json.dumps(v05_label()))):
        c = _Rec([("not json", "stop"), (good, "stop")])
        res = sp.extract_page(str(img), c, decode=DecodePolicy.from_env(contract))
        assert res["ok"] and res["rescued_by"] == "retry_unenforced"
        views[contract] = [_sampling_view(call) for call in c.calls]
        if contract == "v0.5":
            assert c.calls[0]["max_tokens"] == 30000 - 1920 - 18
    assert views["v0.5"] == views["v0.4"]
    first, retry = views["v0.5"]
    assert first == {"temperature": 0.4, "extra": {"presence_penalty": 0.3}, "seeded": True}
    assert retry == {"temperature": 0.4, "extra": {"presence_penalty": 0.3}, "seeded": True}


def test_greedy_is_opt_in_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", False)
    img = tmp_path / "p.png"
    Image.new("RGB", (1152, 1536), "white").save(img)
    c = _Rec([(json.dumps(v05_label()), "stop")])
    dec = DecodePolicy.from_env(ct.V05).with_overrides(temperature="0")
    assert sp.extract_page(str(img), c, decode=dec)["ok"]
    assert c.calls[0]["temperature"] == 0 and c.calls[0]["extra_params"] == {"presence_penalty": 0.3}


def test_v05_opt_in_enforced_retry_uses_quad_schema(monkeypatch):
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", False)
    c = _Rec([("not json", "stop"), (json.dumps(v05_label()), "stop")])
    dec = DecodePolicy.from_env(ct.V05).with_overrides(retry_enforce="1")
    res = sp.extract_text_page("Some text.", c, decode=dec)
    assert res["ok"] and res["rescued_by"] == "retry_enforced"
    x2 = c.calls[1]["extra_params"]
    assert x2["structured_outputs"]["json"] is sp.ENFORCEMENT_SCHEMA_V05
    assert x2["json_schema"] is sp.ENFORCEMENT_SCHEMA_V05
    bbox = sp.ENFORCEMENT_SCHEMA_V05["properties"]["tables"]["items"]["properties"]["bbox"]
    assert bbox["items"]["maxItems"] == 2 and bbox["maxItems"] == 4
    assert c.calls[0]["messages"][1]["content"] == [{"type": "text", "text": "Some text."}]


def test_policy_contract_switch_and_guards(monkeypatch):
    monkeypatch.delenv("SHREW_CONTRACT", raising=False)
    monkeypatch.delenv("SHREW_CONTRACT_ALIASES", raising=False)
    assert ct.default_contract() == "v0.5" == DecodePolicy.from_env().contract      # the current release
    monkeypatch.setenv("SHREW_CONTRACT", "v0.4")
    v4 = DecodePolicy.from_env()
    assert v4.contract == "v0.4" and v4.try1_temp == 0.4 and not v4.figure_check
    v5 = v4.with_overrides(contract="v0.5")
    assert (v5.contract, v5.try1_temp, v5.try2_temp, v5.try2_enforce, v5.seed_policy,
            v5.figure_check, v5.heading_fix) == ("v0.5", 0.4, 0.4, False, "fixed", True, True)
    assert v4.with_overrides(contract="v0.5", figure_check="0").figure_check is False
    with pytest.raises(ValueError):
        v4.with_overrides(figure_check="1")
    with pytest.raises(ValueError):
        v4.with_overrides(contract="v0.9")
    assert v5.describe()["contract"] == "v0.5"
    assert "contract" not in v4.describe()


def test_contract_aliases_are_opt_in(monkeypatch):
    """A deployment can accept extra names for a contract; none exist by default."""
    monkeypatch.delenv("SHREW_CONTRACT_ALIASES", raising=False)
    for bad in ("legacy", "v05", "v04", "0.5", "", "v0.6.1", "v0.5.x"):
        with pytest.raises(ValueError):
            ct.validate_contract(bad)
    assert ct.validate_contract("v0.4.1") == "v0.4" and ct.validate_contract("V0.5.0") == "v0.5"   # release line
    monkeypatch.setenv("SHREW_CONTRACT", "")                     # an unset ${SHREW_CONTRACT} in compose
    assert ct.default_contract() == "v0.5"
    monkeypatch.setenv("SHREW_CONTRACT", "v0.4.1")
    assert ct.default_contract() == "v0.4"
    assert ct.validate_contract(" V0.5 ") == "v0.5" and ct.validate_contract("v0.4") == "v0.4"
    monkeypatch.setenv("SHREW_CONTRACT_ALIASES", "legacy=v0.4, New = v0.5 ,broken=v9,=v0.5,alone")
    assert ct.validate_contract("legacy") == "v0.4" and ct.validate_contract("NEW") == "v0.5"
    for bad in ("broken", "alone", "v9"):
        with pytest.raises(ValueError):
            ct.validate_contract(bad)
    monkeypatch.setenv("SHREW_CONTRACT", "legacy")
    assert ct.default_contract() == "v0.4"


def test_payload_fixture_v05():
    """Pinned v0.5 request bodies (the request contract)."""
    import sys
    sys.path.insert(0, str(HERE))
    import v05_payload_capture as cap
    want = json.loads((HERE / "fixtures" / "v05_payloads.json").read_text())
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert cap.capture(d) == want


# --------------------------------------------------------------------------- tool passes

def _doc_with_figs(tmp_path):
    figs = []
    for i in (1, 2):
        p = tmp_path / f"f{i}.png"
        Image.new("RGB", (300, 300), "white").save(p)
        figs.append({"figure_id": f"d_p1_f{i}", "page": 1, "bbox": QUAD_F1, "caption": f"c{i}",
                     "crop_path": str(p)})
    figs.append({"figure_id": "d_p2_pagefail", "page": 2, "bbox": None, "caption": "x",
                 "crop_path": str(tmp_path / "f1.png")})
    return {"figures": figs, "chunks": []}


def test_figure_check_drops_artifacts(tmp_path):
    from app import v05_tools
    doc = _doc_with_figs(tmp_path)
    c = _Rec([("artifact", "stop"), ("figure", "stop")])
    rep = v05_tools.figure_check(doc, c, concurrency=1)
    assert [f["figure_id"] for f in doc["figures"]] == ["d_p1_f2", "d_p2_pagefail"]
    assert rep["checked"] == 2 and rep["artifacts"] == 1 and rep["figures"] == 1
    assert rep["dropped"][0]["figure_id"] == "d_p1_f1"
    for call in c.calls:
        assert call["messages"][0]["content"] == "classify_figure"
        assert call["temperature"] == 0.4 and call["extra_params"]["presence_penalty"] == 0.3
        assert isinstance(call["extra_params"]["seed"], int)          # same recipe as pages


def test_figure_check_fails_open(tmp_path):
    from app import v05_tools
    doc = _doc_with_figs(tmp_path)
    c = _Rec([("maybe?", "stop"), ("figure", "stop")])
    rep = v05_tools.figure_check(doc, c, concurrency=1)
    assert len(doc["figures"]) == 3 and rep["errors"] == 1


def test_heading_fix_relevels_without_touching_text():
    from app import v05_tools
    doc = {"chunks": [
        {"content": "## 1 Intro\n\nbody\n```\n# not a heading\n```"},
        {"content": "## 1.1 Scope\ntext"},
        {"content": "no heading"},
        {"content": "## 2 Results"},
    ]}
    c = _Rec([("## 1 Intro\n### 1.1 Scope\n## 2 Results", "stop")])
    rep = v05_tools.heading_fix(doc, c)
    assert c.calls[0]["messages"] == [
        {"role": "system", "content": "fix_headings"},
        {"role": "user", "content": [{"type": "text", "text": "## 1 Intro\n## 1.1 Scope\n## 2 Results"}]}]
    assert doc["chunks"][1]["content"] == "### 1.1 Scope\ntext"
    assert doc["chunks"][0]["content"].endswith("# not a heading\n```")
    assert rep == {"headings": 3, "calls": 1, "changed": 1, "rejected": 0, "errors": 0}


def test_heading_fix_rejects_rewrites():
    from app import v05_tools
    doc = {"chunks": [{"content": "## A\n## B"}]}
    c = _Rec([("## A\n### Bee", "stop")])
    rep = v05_tools.heading_fix(doc, c)
    assert doc["chunks"][0]["content"] == "## A\n## B" and rep["rejected"] == 1


def test_heading_windows_stay_in_training_envelope(monkeypatch):
    from app import v05_tools
    lines = [f"## Heading number {i}" for i in range(60)]
    spans = v05_tools._windows(lines)
    assert spans[0] == (0, 24) and spans[-1][1] == 60
    for a, b in spans:
        assert b - a <= 24 and len("\n".join(lines[a:b])) <= 1700


# --------------------------------------------------------------------------- end to end

class _Router:
    """Answers by command, like an v0.5 checkpoint would."""
    model = "shrew-ocr-preview"

    def __init__(self):
        self.calls = []

    def chat_completion(self, messages, max_tokens=None, temperature=None, timeout=None,
                        extra_params=None):
        cmd = messages[0]["content"]
        self.calls.append((cmd, messages, max_tokens, temperature, extra_params))
        n = sum(1 for c in self.calls if c[0] == cmd)
        if cmd == "structured_extraction":
            out = json.dumps(v05_label())
        elif cmd == "classify_figure":
            # By what the crop shows, not by arrival order: the crops are
            # classified on parallel threads. The logo region is a solid block.
            raw = base64.b64decode(messages[1]["content"][0]["image_url"]["url"].split(",", 1)[1])
            dark = Image.open(io.BytesIO(raw)).convert("L").resize((16, 16)).tobytes()
            out = "artifact" if sum(dark) / 256 < 100 else "figure"
        elif cmd == "fix_headings":
            out = "## 1 Introduction\n### 1.1 Scope"
        else:
            raise AssertionError(f"unexpected system turn {cmd!r}")
        return {"choices": [{"message": {"content": out}, "finish_reason": "stop"}]}


def _page_png(path):
    im = Image.new("RGB", (1700, 2200), "white")
    d = ImageDraw.Draw(im)
    for y in range(100, 2100, 30):
        for x in range(100, 1600, 14):
            d.rectangle([x, y, x + 8, y + 12], fill="black")
    d.rectangle([190, 240, 800, 970], fill="black")      # the first figure's box (the "logo")
    im.save(path)


def test_end_to_end_v05(tmp_path, monkeypatch):
    from app import fallback as fb
    from app.models import PipelineConfig
    from app.structured_pipeline import run_structured_pipeline
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", False)
    monkeypatch.setattr(fb, "make_fallback_client", lambda: None)
    page = tmp_path / "page.png"
    _page_png(page)
    client = _Router()
    res = run_structured_pipeline(str(page), str(tmp_path / "out"),
                                  PipelineConfig(vlm_url="http://fake", vlm_model="shrew-ocr-preview"),
                                  client=client, decode=DecodePolicy.from_env(ct.V05))
    sj, log = res.structured_json, res.processing_log
    # every system turn is a bare command; every call uses the one decode recipe
    assert {c[0] for c in client.calls} == {"structured_extraction", "classify_figure", "fix_headings"}
    assert all(c[3] == 0.4 and c[4]["presence_penalty"] == 0.3 for c in client.calls)
    # boxes are the quad, in place of the 2-point box (breaking vs v0.4)
    assert sj["tables"][0]["bbox"] == QUAD_T
    assert sj["tables"][0]["description"] == "One-cell table."
    assert [i["bbox"] for i in sj["images"]] == [QUAD_F2]               # the logo was an artifact
    assert sj["images"][0]["chart_data"] == CHART and sj["images"][0]["data"]
    assert log["figure_check"]["artifacts"] == 1
    assert log["heading_fix"]["changed"] == 1
    assert "### 1.1 Scope" in res.clean_markdown
    assert log["decode"]["contract"] == "v0.5" and log["failed_pages"] == 0


def test_end_to_end_v05_switches_off(tmp_path, monkeypatch):
    from app import fallback as fb
    from app.models import PipelineConfig
    from app.structured_pipeline import run_structured_pipeline
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", False)
    monkeypatch.setattr(fb, "make_fallback_client", lambda: None)
    page = tmp_path / "page.png"
    _page_png(page)
    client = _Router()
    dec = DecodePolicy.from_env(ct.V05).with_overrides(figure_check="0", heading_fix="0")
    res = run_structured_pipeline(str(page), str(tmp_path / "out"),
                                  PipelineConfig(vlm_url="http://fake", vlm_model="shrew-ocr-preview"),
                                  client=client, decode=dec)
    assert {c[0] for c in client.calls} == {"structured_extraction"}
    assert len(res.structured_json["images"]) == 2
    assert "figure_check" not in res.processing_log and "heading_fix" not in res.processing_log


def _latex_router(monkeypatch):
    """_Router whose page carries one bare-LaTeX chunk."""
    label = v05_label()
    label["semantic_chunks"].append({"chunk_id": "3", "title": "Math", "content": "\\frac{a}{b} = c",
                                     "keywords": ["math"], "section_type": "technical_content"})
    monkeypatch.setitem(globals(), "v05_label", lambda: label)
    return _Router()


def test_end_to_end_v05_wraps_bare_latex_and_switch(tmp_path, monkeypatch):
    from app import fallback as fb
    from app.models import PipelineConfig
    from app.structured_pipeline import run_structured_pipeline
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", False)
    monkeypatch.setattr(fb, "make_fallback_client", lambda: None)
    page = tmp_path / "page.png"
    _page_png(page)
    cfg = PipelineConfig(vlm_url="http://fake", vlm_model="shrew-ocr-preview")
    res = run_structured_pipeline(str(page), str(tmp_path / "o1"), cfg, client=_latex_router(monkeypatch),
                                  decode=DecodePolicy.from_env(ct.V05))
    assert "$$\\frac{a}{b} = c$$" in res.clean_markdown
    assert res.processing_log["latex_wrap"] == {"chunks_changed": 1, "mixed": "inline"}
    monkeypatch.setenv("SHREW_LATEX_WRAP", "0")
    res = run_structured_pipeline(str(page), str(tmp_path / "o2"), cfg, client=_latex_router(monkeypatch),
                                  decode=DecodePolicy.from_env(ct.V05))
    assert "$$" not in res.clean_markdown and "latex_wrap" not in res.processing_log


# --------------------------------------------------------------------------- compose_markdown

def test_compose_markdown_matches_the_server_page(tmp_path, monkeypatch):
    """compose_markdown == the page block /v1/convert ships (figure_check/heading_fix off)."""
    from app import fallback as fb
    from app.compose import compose_markdown
    from app.models import PipelineConfig
    from app.structured_pipeline import run_structured_pipeline
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", False)
    monkeypatch.setattr(fb, "make_fallback_client", lambda: None)
    page = tmp_path / "page.png"
    _page_png(page)
    dec = DecodePolicy.from_env(ct.V05).with_overrides(figure_check="0", heading_fix="0")
    res = run_structured_pipeline(str(page), str(tmp_path / "out"),
                                  PipelineConfig(vlm_url="http://fake", vlm_model="shrew-ocr-preview"),
                                  client=_Router(), decode=dec)
    assert compose_markdown(v05_label(), page_tags=True) == res.clean_markdown


def test_compose_markdown_rules():
    from app.compose import compose_markdown
    lab = v05_label()
    lab["semantic_chunks"].insert(0, {"chunk_id": "0", "title": "Running head", "content": "ACME 2024",
                                      "keywords": [], "section_type": "header"})
    lab["semantic_chunks"].append({"chunk_id": "9", "title": "Caption", "content": "Figure 1. Sales.",
                                   "keywords": [], "section_type": "photo_caption"})
    before = json.dumps(lab, sort_keys=True)
    md = compose_markdown(lab)
    assert json.dumps(lab, sort_keys=True) == before                   # input untouched
    assert md.startswith("## Running head\n\nACME 2024")                 # headers kept
    assert "![Figure 1. Sales.](img:2)" in md and "## Caption" not in md  # placed at its caption chunk
    assert md.index("![Figure 1. Sales.]") < md.index("<table>") < md.index("![](img:1)")
    txt = compose_markdown(lab, image_page=False)
    assert "[Figure: untitled]" in txt and "## Caption\n\nFigure 1. Sales." in txt
    assert compose_markdown({"semantic_chunks": None}) == ""


def test_every_pipeline_phase_is_a_known_telemetry_phase():
    """A trace.phase() name missing from telemetry.PHASES is a 500 in the real
    server (the unit tests run with NullTrace, which accepts anything)."""
    import re
    from app import telemetry
    root = Path(sp.__file__).parent
    names = set()
    for p in root.glob("*.py"):
        names |= set(re.findall(r'trace\.phase\("([a-z_0-9]+)"\)', p.read_text()))
    assert names and names <= telemetry.PHASES, names - telemetry.PHASES


def test_response_carries_v05_pass_reports():
    from app.models import PipelineResult
    from app.server import _build_response
    log = {"total_pages": 1, "decode": {"contract": "v0.5"},
           "figure_check": {"checked": 2, "artifacts": 1}, "heading_fix": {"changed": 1}}
    r = _build_response(PipelineResult("md", {"images": [], "tables": []}, log), False)
    assert r["processing_log"]["figure_check"] == {"checked": 2, "artifacts": 1}
    assert r["processing_log"]["heading_fix"] == {"changed": 1}
    v4 = _build_response(PipelineResult("md", {"images": []}, {"total_pages": 1}), False)
    assert "figure_check" not in v4["processing_log"]


# --------------------------------------------------------------------------- per-page telemetry

class _Streamer:
    """Streams scripted completions through on_delta, like VLMClient."""
    model = "shrew-ocr-preview"

    def __init__(self, completions):
        self.q = list(completions)

    def chat_completion_stream(self, messages, *, max_tokens, temperature, extra_params,
                               timeout, on_delta, wall_clock_s, on_usage=None):
        text, acc, reason = self.q.pop(0), "", None
        for i in range(0, len(text), 50):
            acc += text[i:i + 50]
            reason = on_delta(text[i:i + 50], acc)
            if reason:
                break
        return {"choices": [{"message": {"role": "assistant", "content": acc},
                             "finish_reason": reason or "stop"}]}


def test_trail_records_loop_abort_then_retry():
    loop = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 4000
    res = sp.extract_text_page("Some text.", _Streamer([loop, json.dumps(v05_label())]),
                               decode=DecodePolicy.from_env(ct.V05))
    assert res["ok"] and res["rescued_by"] == "retry_unenforced"
    first, retry = res["trail"]
    assert first["rung"] == "first" and first["finish_reason"] == "repetition_abort"
    assert first["verdict"] == "degenerate" and 0 < first["abort_at_chars"] < len(loop)
    assert first["max_window_ratio"] > 15 and first["temperature"] == 0.4
    assert retry["rung"] == "retry_unenforced" and retry["verdict"] == "ok"
    assert retry["abort_at_chars"] is None


def test_processing_log_pages_v05_only(tmp_path, monkeypatch):
    from app import fallback as fb
    from app.models import PipelineConfig
    from app.server import _build_response
    from app.structured_pipeline import run_structured_pipeline
    monkeypatch.setattr(sp, "STREAM_GUARD_ENABLED", False)
    monkeypatch.setattr(fb, "make_fallback_client", lambda: None)
    page = tmp_path / "page.png"
    _page_png(page)
    cfg = PipelineConfig(vlm_url="http://fake", vlm_model="shrew-ocr-preview")
    res = run_structured_pipeline(str(page), str(tmp_path / "out"), cfg, client=_Router(),
                                  decode=DecodePolicy.from_env(ct.V05))
    pages = _build_response(res, False)["processing_log"]["pages"]
    assert len(pages) == 1
    p = pages[0]
    assert (p["page"], p["ok"], p["status"], p["rescued_by"], p["model_calls"]) == (1, True, "ok", "none", 1)
    assert p["bucket"] in ("B0", "B1", "B2", "B3") and p["seconds"] >= 0 and p["fallback"] is False
    assert [t["rung"] for t in p["trail"]] == ["first"] and p["trail"][0]["finish_reason"] == "stop"
    json.dumps(pages)                                                    # serializable
    v4 = run_structured_pipeline(str(page), str(tmp_path / "out4"), cfg, client=_V04Client(),
                                 decode=DecodePolicy.from_env("v0.4"))
    assert "pages" not in _build_response(v4, False)["processing_log"]


class _V04Client(_Router):
    def chat_completion(self, messages, **kw):
        lab = v05_label(figures=[], tables=[])
        return {"choices": [{"message": {"content": json.dumps(lab)}, "finish_reason": "stop"}]}


# --------------------------------------------------------------------------- v0.5 loop guard extensions

def _prose(n_chars, seed):
    import random
    rnd = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "theta", "kappa", "sigma", "omega", "pump", "valve",
             "flow", "head", "table", "figure", "report", "station", "design", "basis", "upgrade"]
    out = []
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(rnd.choice(words) + str(rnd.randint(0, 99999)))
    return " ".join(out)[:n_chars]


def _words(n_chars, seed):
    """Low-entropy prose (no numbers): a loop body like the model's."""
    import random
    rnd = random.Random(seed)
    w = ["alpha", "beta", "gamma", "delta", "theta", "kappa", "sigma", "omega", "pump", "valve",
         "flow", "head", "table", "figure", "report", "station", "design", "basis", "upgrade",
         "the", "of", "and", "with"]
    out = []
    while sum(len(x) + 1 for x in out) < n_chars:
        out.append(rnd.choice(w))
    return " ".join(out)[:n_chars]


def _stream_through(text, contract, **over):
    g = sp.RepetitionGuard(**({**sp.v05_guard_params(), **over} if contract == "v0.5" else {}))
    for i in range(0, len(text), 4):
        if g(text[i:i + 4], text[:i + 4]):
            return g, i + 4
    return g, None


OLD_V05 = {"long_window": 8000, "long_threshold": 12.0, "periodic_chars": None,
           "repeat_chars": None, "numeric_repeats": None}      # the first v0.5 development guard


def test_long_period_loop_caught_by_v05_periodic_guard():
    # Period 2,200 chars > the 2,000-char window: the v0.4 guard sees ratio ~3.8.
    block = _words(2200, 1)
    text = "{" + block * 15
    g4, at4 = _stream_through(text, "v0.4")
    assert at4 is None                            # v0.4 guard: unchanged, blind to it
    g9, at9 = _stream_through(text, "v0.5")
    assert at9 is not None and at9 < 20000 and g9.abort_guard == "periodic"
    ev = g9.stats()["abort_evidence"]
    assert ev["guard"] == "periodic" and ev["score"] >= ev["threshold"] == sp.V05_LOOP_PERIODIC_CHARS
    assert ev["period_chars"] == 2200 and ev["at_chars"] == at9
    assert ev["tail"] == text[:at9][-sp.V05_LOOP_TAIL_CHARS:]


def test_runaway_caught_by_v05_length_cap_only():
    text = _prose(70000, 2)                       # never repeats, just never stops
    assert _stream_through(text, "v0.4")[1] is None
    g9, at9 = _stream_through(text, "v0.5")
    assert g9.abort_guard == "length_cap" and 60000 <= at9 < 61000
    assert g9.stats()["abort_evidence"]["guard"] == "length_cap"


def _table(rows):
    body = "".join("<tr>\\n" + "".join(f"<td>{c}</td>\\n" for c in r) + "</tr>\\n" for r in rows)
    return ('{"metadata": {"title": null}, "summary": "A table.", "semantic_chunks": [], "figures": [], '
            '"tables": [{"bbox": [[1, 2], [3, 2], [3, 4], [1, 4]], "caption": null, "html": "<table>\\n<tbody>\\n'
            + body + '</tbody>\\n</table>"}]}')


def test_dense_table_with_distinct_rows_is_not_a_loop():
    """OmniDocBench page-213e6a57 (2026-11-14): a correct 68-row table whose rows
    share most cell values. The 8k zlib window aborted it; nothing repeats."""
    import random
    rnd = random.Random(7)
    names = ["".join(rnd.choice("abcdefghilmnoprstuv") for _ in range(rnd.randint(6, 13))).capitalize()
             for _ in range(110)]
    rows = [(f"{n[0]}.{n[1:]}", "U–A", rnd.choice("789"), "G–C", "9 (U20A)", "U–A", "7", "G–C",
             rnd.choice(["9 (CU20AB)", "8 (U20A)"]), rnd.choice("+−"), rnd.choice("+−")) for n in names]
    text = _table(rows)
    assert len(text) > 15000
    g_old, at_old = _stream_through(text, "v0.5", **OLD_V05)
    assert at_old is not None and g_old.abort_guard == "long_window"      # the false abort
    g, at = _stream_through(text, "v0.5")
    assert at is None and g.stats()["max_periodic_chars"] == 0 and g.stats()["max_repeat_chars"] == 0


def _dense_rows(n=110):
    import random
    rnd = random.Random(7)
    names = ["".join(rnd.choice("abcdefghilmnoprstuv") for _ in range(rnd.randint(6, 13))).capitalize()
             for _ in range(n)]
    return [(f"{x[0]}.{x[1:]}", "U–A", rnd.choice("789"), "G–C", "9 (U20A)", "U–A", "7", "G–C",
             rnd.choice(["9 (CU20AB)", "8 (U20A)"]), rnd.choice("+−"), rnd.choice("+−")) for x in names]


def test_dense_table_passes_the_v05_document_gate():
    """The same page finished normally and was then failed by the whole-document
    zlib gate (7.8-8.0 raw > 7.0): HTML markup, not content, is what compresses."""
    html = ("<table>\n<tbody>\n" + "".join(
        "<tr>\n" + "".join(f"<td>{c}</td>\n" for c in r) + "</tr>\n" for r in _dense_rows()) + "</tbody>\n</table>")
    lab = v05_label(tables=[{"bbox": QUAD_T, "caption": "Table 2.", "html": html, "description": "Species table."}])
    text = json.dumps(lab, ensure_ascii=False)
    assert sp.zlib_ratio(text) > sp.ZLIB_GATE_RATIO > sp.content_zlib_ratio(text)   # the false failure
    parsed, verdict, err = sp._gate(text, "stop", contract=ct.V05)
    assert verdict == "ok" and parsed is not None, err
    assert sp._gate(text, "stop")[1] == "degenerate"                 # v0.4 gate: unchanged
    # A loop that stopped by itself is still degenerate under the v0.5 gate.
    loop = json.dumps(v05_label(semantic_chunks=[
        {"chunk_id": "1", "title": "Notice", "content": "The meeting will be held at the county office on Monday. " * 120,
         "keywords": ["notice"], "section_type": "technical_content"}]))
    assert sp._gate(loop, "stop", contract=ct.V05)[1] == "degenerate"


def test_abort_error_names_the_guard_that_fired():
    g, at = _stream_through("{" + _words(2200, 1) * 15, "v0.5")
    _, verdict, err = sp._gate("x", sp.REPETITION_ABORT, g, ct.V05)
    assert verdict == "degenerate" and err.startswith("repetition_abort: periodic guard ") and f"at {at} chars" in err


def test_linear_numeric_table_is_not_a_loop():
    # A conversion table: every column steps by a constant. Real, and nothing is re-emitted.
    rows = [(str(i), f"{i * 25.4:.1f}", str(i + 60), f"{(i + 60) * 25.4:.1f}") for i in range(1, 61)]
    assert _stream_through(_table(rows), "v0.5")[1] is None
    rows = [(f"R3116x{311 + 10 * i}A/C", f"{3.075 + .099 * i:.3f}", f"{3.125 + .101 * i:.3f}") for i in range(90)]
    assert _stream_through(_table(rows), "v0.5")[1] is None


def test_numbered_rows_of_a_real_page_are_not_a_loop():
    """Pages whose rows are numbered and share their words."""
    grid = _table([[str(r * 10 + c + 1) for c in range(10)] for r in range(30)])              # a "300 chart"
    blanks = json.dumps({"content": "".join(f"{i}. ________\n" for i in range(1, 201))})
    schedule = _table([(str(i), f"{(i - 1) % 12 + 1:02d}/01/{2025 + (i - 1) // 12}", "Scheduled monthly payment",
                        "$1,250.00") for i in range(1, 121)])
    wide = _table([(str(i), f"{(i - 1) % 12 + 1:02d}/01/{2025 + (i - 1) // 12}",
                    "Scheduled monthly principal and interest payment", "$1,250.00") for i in range(1, 61)])
    answers = json.dumps({"content": "".join(f"Question {i}: A B C D E\n" for i in range(1, 301))})
    for text in (grid, blanks, schedule, wide, answers):
        for lead in ("", " ", "  ", "   "):                 # every alignment of the 4-char deltas
            g, at = _stream_through(lead + text, "v0.5")
            assert at is None, (g.abort_guard, at, text[:80])


def test_numbered_rows_that_never_end_are_a_loop():
    text = json.dumps({"content": "".join(f"第{i}條：\n" for i in range(1, 3000))}, ensure_ascii=False)
    g, at = _stream_through(text, "v0.5", threshold=1e9)
    assert g.abort_guard == "numeric" and at < 12000
    ev = g.stats()["abort_evidence"]
    assert ev["numbered_rows"] and "條" in ev["unit"] and ev["period_chars"] == len(ev["unit"])


def test_partial_tag_at_the_stream_edge_does_not_hide_a_loop():
    from app import loop_content as lc
    rows = [("产品提供商", "透明医疗", "1,002", "125", "51%", "15%", "1,440", "15%")] * 400
    text = _table(rows)
    scores = [lc.periodic(text[:n])["score"] for n in range(20000, 20200)]
    assert min(scores) > 0


def test_abort_guard_is_named_without_the_length_cap():
    g, at = _stream_through("{" + _words(2200, 1) * 15, "v0.5", length_cap_chars=None)
    assert at and g.stats()["abort_guard"] == "periodic"


def test_numbered_exhibit_rows_are_not_a_loop():
    """A legal exhibit lists units that share ~110
    letters of boilerplate. Numbered, so held to the numbered-row line."""
    rows = "".join(f"Unit {100 + i}, Building A, of Sunset Bay Condominium, according to the Declaration of "
                   f"Condominium recorded in the Public Records of Pinellas County, Florida.\n" for i in range(1, 36))
    text = json.dumps({"content": _prose(9000, 5) + "\n" + rows})
    for lead in ("", " ", "  ", "   "):
        g, at = _stream_through(lead + text, "v0.5")
        assert at is None, (g.abort_guard, at)
    assert g.counter_max > sp.V05_LOOP_PERIODIC_CHARS          # the old single line would have cut it


def test_numbered_list_sharing_a_phrase_is_not_re_emitted_text():
    # Rows that differ only in their number are new rows to the repeated guard.
    from app import loop_content as lc
    text = json.dumps({"content": "".join(f"Valve HV-{1000 + i} (ball valve) on line 3-DW-403-A1A to the header; "
                                          for i in range(80))})
    assert lc.repeated(text)["score"] == 0 and _stream_through(text, "v0.5")[1] is None


def test_rows_cycling_with_different_numbers_are_a_loop():
    # Three rows repeat as a group: the numbers are regular only at three periods.
    rows = [("Northern Coast Co.", "139", "2.50"), ("Northern Coast Co.", "78", "2.59"),
            ("Northern Coast Co.", "4532", "31")] * 300
    g, at = _stream_through(_table(rows), "v0.5", threshold=1e9)
    assert g.abort_guard == "periodic" and at < 30000


def test_abort_evidence_unit_is_the_text_that_repeats():
    text = json.dumps({"content": "".join(f"Valve HV-{1000 + i} (ball valve) on line 3-DW-403-A1A to the header; "
                                          for i in range(400))})
    g, at = _stream_through(text, "v0.5", threshold=1e9)
    ev = g.stats()["abort_evidence"]
    row = "(ball valve) on line 3-DW-403-A1A to the header; Valve HV-"
    assert ev["period_chars"] == len("Valve HV-1000 (ball valve) on line 3-DW-403-A1A to the header; ")
    assert row in ev["unit"] + ev["unit"] and ev["unit"] in text


def test_abort_error_without_evidence():
    g, at = _stream_through("{" + _words(2200, 1) * 15, "v0.5", tail_chars=0)
    assert g.evidence is None
    assert sp._gate("x", sp.REPETITION_ABORT, g, ct.V05)[2] == f"repetition_abort: periodic guard at {at} chars"


def test_table_rows_repeating_is_a_loop():
    rows = [("2018年 1-9月", "美国", "原材料及产品提供商", "1,002", "125", "51%", "15%", "1,440"),
            ("产品提供商", "透明医疗", "1,002", "125", "51%", "15%", "1,440", "15%")] * 400
    g, at = _stream_through(_table(rows)[:40000], "v0.5", threshold=1e9)   # judge the content guard alone
    assert g.abort_guard == "periodic" and at < 30000


def test_counter_loop_is_a_loop():
    # One counter steps, every word is re-emitted. Its 2k window stays under 15.
    text = '{"summary": "' + "".join(f'HV-{2269 + i} (ball valve) on line 3"-DW-403-A1A to the header; '
                                    for i in range(700))
    g4, at4 = _stream_through(text, "v0.4")
    assert at4 is None
    g, at = _stream_through(text, "v0.5")
    assert g.abort_guard == "periodic" and at < 12000


def test_number_only_counter_is_a_loop_by_length():
    text = '{"summary": "' + ", ".join(str(4012 + i) for i in range(4000))
    g, at = _stream_through(text, "v0.5", threshold=1e9)          # judge the numeric guard alone
    assert g.abort_guard == "numeric" and at < 6000
    assert g.stats()["abort_evidence"]["score"] >= sp.V05_LOOP_NUMERIC_REPEATS


def _notice(i, rnd):
    street = "".join(rnd.choice("abcdefghiklmnoprstuw") for _ in range(9)).capitalize()
    return (f"SUBSTITUTE TRUSTEES' SALE OF IMPROVED REAL PROPERTY\\n\\n{1000 + 37 * i} {street} ROAD\\n\\n"
            f"Under a power of sale contained in a Deed of Trust from {street} Holdings, dated the "
            f"{i + 1}th, the trustees will sell at public auction lot {500 + i} as recorded.\\n\\n"
            "The property will be sold in an as is condition and subject to conditions, restrictions and "
            "agreements of record affecting the same. Terms of Sale: A deposit by cash or certified check. "
            "Balance of the purchase price to be paid in cash within ten days of final ratification of the "
            "sale by the Circuit Court. TIME IS OF THE ESSENCE FOR THE PURCHASER. Purchaser is responsible "
            "for any recapture of homestead tax credit. All transfer taxes shall be paid by Purchaser.\\n\\n")


def test_boilerplate_around_new_items_is_not_a_loop_but_cycling_is():
    import random
    rnd = random.Random(3)
    notices = [_notice(i, rnd) for i in range(40)]
    page = '{"semantic_chunks": [{"content": "' + "".join(notices)       # 40 different notices
    g, at = _stream_through(page, "v0.5")
    assert at is None and g.stats()["max_repeat_chars"] == 0
    rnd = random.Random(4)                                              # the same 5, in drifting order
    five = notices[:5]
    cycling = '{"semantic_chunks": [{"content": "' + "".join(
        "".join(rnd.sample(five, 5)) for _ in range(12))
    g, at = _stream_through(cycling, "v0.5", periodic_chars=None)
    assert g.abort_guard == "repeated"
    ev = g.stats()["abort_evidence"]
    assert ev["score"] >= sp.V05_LOOP_REPEAT_CHARS and ev["novel_share"] <= 0.10 and ev["top_units"]


def test_content_statistics_ignore_markup_and_keys():
    from app import loop_content as lc
    assert lc.content('<tr>\\n<td>a</td>\\n</tr>') == " a "
    assert lc.periodic("<tr><td></td><td></td></tr>" * 400)["score"] == 0          # empty rows
    assert lc.periodic("____ ______ " * 400)["score"] == 0                          # blanks of a form
    assert lc.periodic("[illegible signature] " * 40)["score"] < sp.V05_LOOP_PERIODIC_CHARS
    assert lc._number("四十五")[0] == 45 and lc._number("XVIII")[0] == 18 and lc._number("二〇二五")[0] == 2025


def test_clean_prose_never_aborts_v05():
    for seed in range(5):
        g9, at9 = _stream_through(_prose(40000, 100 + seed), "v0.5")
        assert at9 is None, seed


def test_v04_guard_stats_shape_unchanged():
    g, _ = _stream_through("x", "v0.4")
    assert set(g.stats()) == {"aborted", "ratio", "max_window_ratio", "position_chars",
                              "threshold", "window_chars"}


def test_trail_names_the_abort_guard(monkeypatch):
    loop = "{" + _words(2200, 1) * 15
    res = sp.extract_text_page("Some text.", _Streamer([loop, json.dumps(v05_label())]),
                               decode=DecodePolicy.from_env(ct.V05))
    first = res["trail"][0]
    assert first["finish_reason"] == "repetition_abort" and first["abort_guard"] == "periodic"
    ev = first["abort_evidence"]
    assert ev["guard"] == "periodic" and ev["score"] >= ev["threshold"] and ev["window_chars"]
    assert len(ev["tail"]) == sp.V05_LOOP_TAIL_CHARS and loop[:ev["at_chars"]].endswith(ev["tail"])
    assert first["max_periodic_chars"] >= sp.V05_LOOP_PERIODIC_CHARS
    assert res["trail"][1]["abort_evidence"] is None                    # the clean retry


# --------------------------------------------------------------------------- raw dump

def test_raw_dump_v05_only_and_opt_in(tmp_path, monkeypatch):
    d = tmp_path / "dump"
    good = json.dumps(v05_label())
    good_v04 = json.dumps(v05_label(figures=[], tables=[]))
    monkeypatch.setattr(sp, "RAW_DUMP_DIR", None)
    sp.extract_text_page("x", _Streamer([good]), decode=DecodePolicy.from_env(ct.V05), doc_id="d1")
    assert not d.exists()                                              # off by default
    monkeypatch.setattr(sp, "RAW_DUMP_DIR", str(d))
    sp.extract_text_page("x", _Streamer([good_v04]), decode=DecodePolicy.from_env("v0.4"), doc_id="d0")
    assert not d.exists()                                              # never under v0.4
    sp.extract_text_page("x", _Streamer(["not json", good]), decode=DecodePolicy.from_env(ct.V05),
                         doc_id="d/2", page_no=3)
    rows = [json.loads(l) for l in (d / "d_2.jsonl").read_text().splitlines()]
    assert [(r["rung"], r["verdict"], r["page"]) for r in rows] == [("first", "parse", 3),
                                                                    ("retry_unenforced", "ok", 3)]
    assert rows[1]["text"] == good and rows[0]["chars"] == len("not json")
