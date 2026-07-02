from pathlib import Path

from app.structured_page import extract_page, parse_json_lenient, validate_schema, SENTINEL

# extract_page opens image_path to base64-encode it; the fake client never
# touches disk itself, so a tiny placeholder file at the hardcoded path used
# by every test below is enough (contents are irrelevant, only bytes matter).
Path("/tmp/fake.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

GOOD = ('{"metadata":{"title":null,"authors":[],"organization":null,"year":null,'
        '"doc_type":null},"summary":"s","semantic_chunks":[],"figures":[],"tables":[]}')


class FakeClient:
    def __init__(self, replies):  # list of (text, finish_reason)
        self.replies = list(replies)
        self.calls = []
        self.model = "shrew-9b"

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                         timeout=None, extra_params=None):
        system = messages[0]["content"]
        user_content = messages[1]["content"]
        self.calls.append({
            "system": system,
            "user_content": user_content,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_params": extra_params,
        })
        text, finish_reason = self.replies.pop(0)
        return {"choices": [{"finish_reason": finish_reason,
                              "message": {"content": text}}]}


def test_happy_path_single_call():
    c = FakeClient([(GOOD, "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert r["ok"] and r["status"] == "ok" and r["attempts"] == 1
    assert r["data"]["summary"] == "s"


def test_parse_failure_resamples_once_then_fails():
    c = FakeClient([("not json", "stop"), ("still not json", "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert not r["ok"] and r["status"] == "failed" and r["attempts"] == 2


def test_length_finish_retries_with_higher_cap_never_truncates():
    c = FakeClient([("{\"partial", "length"), (GOOD, "stop")])
    r = extract_page("/tmp/fake.png", c)
    assert r["ok"] and r["attempts"] == 2
    assert c.calls[1]["max_tokens"] > c.calls[0]["max_tokens"]


def test_sentinel_and_image_only_user():
    c = FakeClient([(GOOD, "stop")])
    extract_page("/tmp/fake.png", c)
    call = c.calls[0]
    assert call["system"] == SENTINEL
    user_content = call["user_content"]
    assert isinstance(user_content, list) and len(user_content) == 1
    assert user_content[0]["type"] == "image_url"
