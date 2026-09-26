"""Capture the exact JSON bodies the structured-page ladder POSTs to the VLM.

Used two ways: once against the pre-knob source tree to record
tests/fixtures/decode_payloads_v0315.json (the payloads shipped before
SHREW_TRY1_TEMP / SHREW_TRY2_TEMP / SHREW_SEED_POLICY existed), and by
test_decode_knobs.py to replay the same scenarios against the current tree
and byte-compare. Scenarios cover every rung: clean first pass; parse failure
-> enforced retry; repetition abort -> re-roll -> enforced retry; and the
text modality.
"""
import base64
import json
import os
import sys
from unittest import mock

# 1x1 white PNG — the image bytes only need to be deterministic.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC")

GOOD = json.dumps({"metadata": {"title": "t", "authors": ["a"], "organization": "o", "year": "2020",
                                "doc_type": "report"}, "summary": "s",
                   "semantic_chunks": [{"chunk_id": "c1", "title": "t", "section_type": "abstract",
                                        "content": "hello world " * 40}],
                   "tables": [], "figures": []})
LOOP = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 4000

SCENARIOS = {                       # completions returned, in call order
    "clean": [GOOD],
    "parse_then_retry": ["not json at all", GOOD],
    "loop_reroll_retry": [LOOP, LOOP, GOOD],
    "text_clean": [GOOD],
}


class _Resp:
    def __init__(self, text):
        self._lines = [f"data: {json.dumps({'choices': [{'delta': {'content': text[i:i + 50]}}]})}"
                       for i in range(0, len(text), 50)]
        self._lines += ['data: {"choices":[{"delta":{},"finish_reason":"stop"}]}', "data: [DONE]"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


def capture(workdir):
    """Return {scenario: [json-body-string, ...]} using whatever `shrew` is importable."""
    from app import structured_page as sp
    from app.vlm_client import VLMClient
    img = os.path.join(workdir, "page.png")
    with open(img, "wb") as f:
        f.write(PNG)
    out = {}
    for name, completions in SCENARIOS.items():
        bodies, queue = [], list(completions)

        def fake_post(url, headers=None, json=None, timeout=None, stream=False, _b=bodies, _q=queue):
            import json as _json
            _b.append(_json.dumps(json, sort_keys=True))
            return _Resp(_q.pop(0))

        client = VLMClient("http://mock:8000", "shrew-ocr-preview")
        with mock.patch("app.vlm_client.requests.post", side_effect=fake_post):
            if name.startswith("text"):
                sp.extract_text_page("A short text page.", client, page_no=1)
            else:
                sp.extract_page(img, client, page_no=1)
        out[name] = bodies
    return out


if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])          # a src/ dir to capture from
    for k in [k for k in os.environ if k.startswith("SHREW_")]:
        del os.environ[k]
    print(json.dumps(capture(sys.argv[2]), indent=1))
