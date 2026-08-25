"""Wiring: VLMClient streaming + _extract's use of the §5.2 guard.

Covers the three outcomes callers must be able to tell apart (loop / over
budget / clean) and the guard's cost claim — that an abort actually stops the
generation rather than just relabelling a completed one.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.structured_page import (
    REPETITION_ABORT,
    _extract,
    build_text_messages,
)

GOOD_PAGE = {
    "metadata": {"title": "T", "authors": [], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "s",
    "semantic_chunks": [{"chunk_id": "1", "title": "A", "content": "body",
                         "section_type": "introduction"}],
    "figures": [],
    "tables": [],
}


class FakeStreamClient:
    """Emits a fixed body as deltas, recording how much was actually consumed."""

    model = "shrew-ocr-preview"

    def __init__(self, body, finish_reason="stop"):
        self.body = body
        self.finish_reason = finish_reason
        self.emitted = 0
        self.calls = 0

    def chat_completion_stream(self, messages, max_tokens=None, temperature=None,
                               timeout=None, extra_params=None, on_delta=None,
                               wall_clock_s=None):
        self.calls += 1
        parts, acc, stop = [], "", None
        for i in range(0, len(self.body), 100):
            piece = self.body[i:i + 100]
            parts.append(piece)
            acc += piece
            self.emitted += len(piece)
            if on_delta is not None:
                stop = on_delta(piece, acc)
                if stop:
                    break
        return {"choices": [{"message": {"content": acc},
                             "finish_reason": stop or self.finish_reason}]}


class FakeBlockingClient:
    """No streaming support — the guard must degrade, not fail the page."""

    model = "shrew-ocr-preview"

    def __init__(self, body):
        self.body = body
        self.calls = 0

    def chat_completion(self, messages, max_tokens=None, temperature=None,
                        timeout=None, extra_params=None, **kw):
        self.calls += 1
        return {"choices": [{"message": {"content": self.body},
                             "finish_reason": "stop"}]}


def test_clean_page_streams_through_unchanged():
    client = FakeStreamClient(json.dumps(GOOD_PAGE))
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    assert res["ok"] and res["status"] == "ok"
    assert res["data"]["metadata"]["title"] == "T"
    assert res["repetition_abort"] is False
    assert res["loop_guard"]["aborted"] is False
    assert client.calls == 1, "a clean page must not take the retry tier"


def test_loop_aborts_early_and_is_flagged():
    loop = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 6000
    client = FakeStreamClient(loop)
    res = _extract(build_text_messages("x"), client, max_tokens=20000)

    assert not res["ok"] and res["status"] == "degenerate"
    assert res["repetition_abort"] is True
    assert res["degenerate"] is True
    # The cost claim: the guard must stop generation, not relabel a finished one.
    assert client.emitted < len(loop), "abort did not shorten the generation"
    assert "repetition_abort" in res["error"]


def test_blocking_client_still_works():
    """A server that cannot stream falls back rather than failing every page."""
    client = FakeBlockingClient(json.dumps(GOOD_PAGE))
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    assert res["ok"] and res["loop_guard"] is None
    assert client.calls == 1


def test_over_budget_is_not_reported_as_a_loop():
    """finish_reason=length must stay distinguishable from an abort: the remedy
    is a bigger budget, not a retry."""
    client = FakeStreamClient('{"metadata": {"title": "half a pa',
                              finish_reason="length")
    res = _extract(build_text_messages("x"), client, max_tokens=20000)
    assert res["status"] == "overlong_failed"
    assert res["repetition_abort"] is False
    assert res["degenerate"] is False


# ── the real SSE parser ─────────────────────────────────────────────────────


class _SSEHandler(BaseHTTPRequestHandler):
    BODY = "hello world, this is a streamed page"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(self.BODY), 7):
            evt = {"choices": [{"delta": {"content": self.BODY[i:i + 7]}}]}
            self.wfile.write(f"data: {json.dumps(evt)}\r\n\r\n".encode())
        self.wfile.write(b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\r\n\r\n')
        self.wfile.write(b"data: [DONE]\r\n\r\n")
        self.wfile.flush()

    def log_message(self, *a):
        pass


@pytest.fixture
def sse_server():
    srv = HTTPServer(("127.0.0.1", 0), _SSEHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_stream_parser_against_a_real_sse_response(sse_server):
    """Exercises the wire format rather than a mock: `data:` prefixes, CRLF
    separators, the terminal [DONE] sentinel and a delta-less finish event."""
    from app.vlm_client import VLMClient

    seen = []
    client = VLMClient(sse_server, "shrew-ocr-preview")
    res = client.chat_completion_stream(
        [{"role": "user", "content": "x"}],
        on_delta=lambda chunk, acc: seen.append(chunk) and None,
    )
    choice = res["choices"][0]
    assert choice["message"]["content"] == _SSEHandler.BODY
    assert choice["finish_reason"] == "stop"
    assert "".join(seen) == _SSEHandler.BODY


def test_on_delta_can_abort_a_real_stream(sse_server):
    from app.vlm_client import VLMClient

    client = VLMClient(sse_server, "shrew-ocr-preview")
    res = client.chat_completion_stream(
        [{"role": "user", "content": "x"}],
        on_delta=lambda chunk, acc: REPETITION_ABORT if len(acc) >= 14 else None,
    )
    choice = res["choices"][0]
    assert choice["finish_reason"] == REPETITION_ABORT
    assert 14 <= len(choice["message"]["content"]) < len(_SSEHandler.BODY)


# ── wall clock starts at first byte, not submission ─────────────────────────


class _SlowFirstByteHandler(_SSEHandler):
    """Simulates deep-queue scheduling: a long wait BEFORE the first byte,
    then a fast healthy stream. Queue wait must not count against the
    generation wall clock (measured: 21 false aborts in one congested hour,
    VHR 2026-08-15)."""

    QUEUE_S = 0.7

    def do_POST(self):
        import time as _t
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _t.sleep(self.QUEUE_S)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(self.BODY), 7):
            evt = {"choices": [{"delta": {"content": self.BODY[i:i + 7]}}]}
            self.wfile.write(f"data: {json.dumps(evt)}\r\n\r\n".encode())
        self.wfile.write(b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\r\n\r\n')
        self.wfile.write(b"data: [DONE]\r\n\r\n")
        self.wfile.flush()


@pytest.fixture
def slow_sse_server():
    srv = HTTPServer(("127.0.0.1", 0), _SlowFirstByteHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_wall_clock_excludes_queue_wait(slow_sse_server):
    """wall_clock_s smaller than the queue wait but larger than the stream:
    submission-relative timing would abort; first-byte-relative completes."""
    from app.vlm_client import VLMClient

    client = VLMClient(slow_sse_server, "shrew-ocr-preview")
    res = client.chat_completion_stream(
        [{"role": "user", "content": "x"}],
        wall_clock_s=0.5,
    )
    choice = res["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == _SSEHandler.BODY


def test_page_context_labels_both_attempts(caplog):
    """Observability port: with page_no supplied, the first pass and the
    enforced retry produce distinguishable, page-correlated abort warnings.
    Behavior (early abort, degenerate verdict, repetition_abort flag) is
    unchanged and still asserted."""
    import logging
    loop = '{"semantic_chunks": [' + '{"chunk_id": "c", "content": "x"},' * 6000
    client = FakeStreamClient(loop)
    with caplog.at_level(logging.WARNING, logger="shrew.structured_page"):
        res = _extract(build_text_messages("x"), client, max_tokens=20000,
                       page_no=12)
    assert not res["ok"] and res["status"] == "degenerate"
    assert res["repetition_abort"] is True
    warnings = [r.getMessage() for r in caplog.records
                if "repetition_abort" in r.getMessage()]
    assert any(w.startswith("Page 12 (attempt 1/2, first pass): repetition_abort")
               for w in warnings)
    assert any(w.startswith("Page 12 (attempt 2/2, enforced retry): repetition_abort")
               for w in warnings)
