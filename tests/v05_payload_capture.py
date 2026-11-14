"""Capture the exact JSON bodies the v0.5 contract POSTs to the VLM.

Pins tests/fixtures/v05_payloads.json (test_v05_contract.py::test_payload_fixture_v05).
The sampling params are v0.4.0's recipe (0.4 / pp 0.3 / fixed seed). The base64 image is elided to "<PNG>" so the fixture reads
as the contract, not as image bytes. Regenerate deliberately:

    python tests/v05_payload_capture.py > tests/fixtures/v05_payloads.json
"""
import json
import os
import sys
from unittest import mock

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from decode_payload_capture import _Resp  # noqa: E402

LABEL = json.dumps({"metadata": {"title": None, "authors": None, "organization": None, "year": None,
                                 "doc_type": None}, "summary": "s",
                    "semantic_chunks": [{"chunk_id": "1", "title": "t", "content": "hello world " * 40,
                                         "keywords": ["k"], "section_type": "body"}],
                    "figures": [], "tables": []})

SCENARIOS = {                     # (kind, completions in call order)
    "page_B1_clean": ("page", [LABEL]),
    "page_B1_parse_then_retry": ("page", ["not json", LABEL]),
    "text_clean": ("text", [LABEL]),
    "classify_figure": ("classify", ["figure"]),
    "fix_headings": ("headings", ["## A\n### B"]),
}


def _elide(body: dict) -> dict:
    for m in body.get("messages", []):
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "image_url":
                    p["image_url"]["url"] = p["image_url"]["url"].split(",")[0] + ",<PNG>"
    return body


def capture(workdir):
    from app import contract as ct
    from app import v05_tools
    from app import structured_page as sp
    from app.vlm_client import VLMClient
    img = os.path.join(workdir, "page.png")
    Image.new("RGB", (1152, 1536), "white").save(img)          # a B1-sized model input
    out = {}
    for name, (kind, completions) in SCENARIOS.items():
        bodies, queue = [], list(completions)

        def fake_post(url, headers=None, json=None, timeout=None, stream=False, _b=bodies, _q=queue):
            import copy
            import json as _json
            _b.append(_json.dumps(_elide(copy.deepcopy(json)), sort_keys=True))
            text = _q.pop(0)
            if stream:
                return _Resp(text)
            return _Plain(text)

        client = VLMClient("http://mock:8000", "shrew-ocr-preview")
        dec = sp.DecodePolicy.from_env(ct.V05)
        with mock.patch("app.vlm_client.requests.post", side_effect=fake_post):
            if kind == "page":
                sp.extract_page(img, client, page_no=1, decode=dec)
            elif kind == "text":
                sp.extract_text_page("A short text page.", client, page_no=1, decode=dec)
            elif kind == "classify":
                crop = os.path.join(workdir, "fig.png")
                Image.new("RGB", (400, 300), "white").save(crop)
                v05_tools.figure_check({"figures": [{"figure_id": "d_p1_f1", "page": 1,
                                                     "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]],
                                                     "crop_path": crop}], "chunks": []},
                                       client, concurrency=1)
            else:
                v05_tools.heading_fix({"chunks": [{"content": "## A\n## B"}]}, client)
        out[name] = bodies
    return out


class _Plain:
    def __init__(self, text):
        self.text = text
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": self.text},
                             "finish_reason": "stop"}]}


if __name__ == "__main__":
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(capture(d), indent=1, sort_keys=True))
