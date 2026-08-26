"""Cached readiness gate (audit issue #17).

The bug: a per-request inference probe queued behind real OCR under load,
timed out at 10s, and 503'd healthy-but-busy backends. These tests pin the
new contract: the per-request gate never runs inference, fails OPEN on a
slow/busy backend that was healthy before, and rejects only definite faults.
"""
import time

import pytest

from app import vlm_client as vc
from app.vlm_client import VLMClient


@pytest.fixture(autouse=True)
def _clear_cache():
    vc._readiness.clear()
    yield
    vc._readiness.clear()


def _client():
    return VLMClient(base_url="http://vlm.test:8000", model="shrew-ocr-preview")


def test_is_ready_never_runs_inference(monkeypatch):
    """The per-request path may hit /v1/models but must NEVER POST an
    inference — that was the whole cause of the false-503-under-load."""
    calls = []

    def fake_get(url, **kw):
        calls.append(("GET", url))
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"data": [{"id": "shrew-ocr-preview"}]}
        return R()

    def fake_post(url, **kw):
        calls.append(("POST", url))
        raise AssertionError("is_ready must not POST an inference")

    monkeypatch.setattr(vc.requests, "get", fake_get)
    monkeypatch.setattr(vc.requests, "post", fake_post)
    assert _client().is_ready() is True
    assert all(m == "GET" for m, _ in calls)


def test_warm_cache_admits_without_any_network(monkeypatch):
    """With a fresh healthy stamp (kept warm by the background refresher) the
    gate must not touch the network at all — no queuing behind OCR."""
    c = _client()
    vc.VLMClient._mark_readiness(vc._readiness_key(c.base_url, c.model), True)

    def boom(*a, **k):
        raise AssertionError("warm cache must not probe")
    monkeypatch.setattr(vc.requests, "get", boom)
    monkeypatch.setattr(vc.requests, "post", boom)
    assert c.is_ready() is True


def test_busy_backend_fails_open_when_previously_healthy(monkeypatch):
    """The incident: backend healthy but SATURATED, probe times out. If it was
    ever healthy, admit anyway — the pipeline handles per-page errors."""
    c = _client()
    # seed an 'ever healthy' but now-stale entry
    vc.VLMClient._mark_readiness(vc._readiness_key(c.base_url, c.model), True)
    vc._readiness[vc._readiness_key(c.base_url, c.model)]["ts"] = time.time() - 10_000

    def timeout_get(*a, **k):
        raise vc.requests.exceptions.Timeout("queued behind OCR")
    monkeypatch.setattr(vc.requests, "get", timeout_get)
    assert c.is_ready() is True   # fail-open


def test_never_healthy_and_unreachable_rejects(monkeypatch):
    c = _client()

    def timeout_get(*a, **k):
        raise vc.requests.exceptions.Timeout("down")
    monkeypatch.setattr(vc.requests, "get", timeout_get)
    assert c.is_ready() is False  # never proven healthy → don't admit


def test_definite_fault_model_absent_rejects(monkeypatch):
    """A DEFINITE fault (model not served) must reject even fail-open —
    it's a config error, not transient load."""
    c = _client()
    vc.VLMClient._mark_readiness(vc._readiness_key(c.base_url, c.model), True)
    vc._readiness[vc._readiness_key(c.base_url, c.model)]["ts"] = time.time() - 10_000

    def wrong_model_get(url, **kw):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"data": [{"id": "some-other-model"}]}
        return R()
    monkeypatch.setattr(vc.requests, "get", wrong_model_get)
    assert c.is_ready() is False


def test_probe_classifies_hard_vs_soft(monkeypatch):
    c = _client()

    def auth_get(url, **kw):
        class R:
            status_code = 403
            def raise_for_status(self): pass
            def json(self): return {}
        return R()
    monkeypatch.setattr(vc.requests, "get", auth_get)
    verdict, _ = c.probe(timeout=1, do_inference=False)
    assert verdict == "hard"

    def timeout_get(*a, **k):
        raise vc.requests.exceptions.Timeout()
    monkeypatch.setattr(vc.requests, "get", timeout_get)
    verdict, _ = c.probe(timeout=1, do_inference=False)
    assert verdict == "soft"
