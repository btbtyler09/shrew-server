"""/health concurrency section (v0.3.7): capacity + live conversion activity.

Acceptance list from the issue: queued->running transitions, aggregation,
effective-limit math, presence in healthy/degraded/unhealthy responses, both
endpoints tracked, shared per-worker gate, release on completion / error /
cancellation, no live inference from /health.
"""
import asyncio
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

import app.server as server_module
from app.models import PipelineResult
from app.pipeline import CancelledException
from app.server import ServerConfig, app

GOOD_PAGE_JSON = json.dumps({
    "metadata": {"title": None, "authors": [], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "s",
    "semantic_chunks": [{"chunk_id": "1", "title": "T", "content": "c",
                          "keywords": [], "section_type": "introduction"}],
    "figures": [], "tables": [],
})


class FakeVLMClient:
    ready = True

    def __init__(self, base_url=None, model=None, api_key=None, **kwargs):
        self.base_url = base_url
        self.model = model or "shrew-9b"

    def is_ready(self):
        return type(self).ready

    def readiness_snapshot(self):
        return {"ready": type(self).ready, "age_s": 0.0, "ever_ok": True}

    def probe(self, timeout=10, do_inference=None):
        return ("ok", None)

    def health_check(self, timeout=10):
        return type(self).ready

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                         timeout=None, extra_params=None):
        return {"choices": [{"finish_reason": "stop",
                              "message": {"content": GOOD_PAGE_JSON}}]}


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    try:
        from fastapi.testclient import TestClient
    except Exception as e:
        pytest.skip(f"fastapi TestClient unavailable: {e}")

    monkeypatch.setenv("SHREW_CONCURRENCY_DIR", str(tmp_path / "leases"))
    FakeVLMClient.ready = True
    monkeypatch.setattr("app.server.VLMClient", FakeVLMClient)
    monkeypatch.setattr("app.structured_pipeline.VLMClient", FakeVLMClient)
    monkeypatch.setattr(server_module, "_config", ServerConfig(
        vlm_url="http://fake-vlm", vlm_model="shrew-9b",
        workers=2, pipeline_concurrency=1, vlm_concurrency=4,
    ))
    monkeypatch.setattr(server_module, "_figure_converter", None)
    monkeypatch.setattr(server_module, "_shrew_lora_map", None)
    monkeypatch.setattr(server_module, "_shrew_lora_format", "none")
    monkeypatch.setattr(server_module, "_vlm_pool", ThreadPoolExecutor(max_workers=2))
    monkeypatch.setattr(server_module, "_pipeline_gate", threading.Semaphore(1))

    return TestClient(app)


def _upload_png():
    buf = io.BytesIO()
    Image.new("RGB", (400, 500), "white").save(buf, format="PNG")
    buf.seek(0)
    return {"file": ("doc.png", buf, "image/png")}


def _wait_until(pred, timeout=15.0, msg="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {msg}")


def _conversions(api_client):
    return api_client.get("/health").json()["concurrency"]["conversions"]


class _BlockingPipeline:
    """Stands in for run_structured_pipeline: parks until released, honoring
    progress cancellation like the real pipeline (raises CancelledException)."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()

    def __call__(self, tmp_path, output_dir, config, progress=None, raw=False,
                 client=None):
        self.entered.set()
        while not self.release.wait(timeout=0.05):
            if progress is not None and progress.is_cancelled():
                raise CancelledException()
        return PipelineResult("# ok", {}, {"total_pages": 1, "total_figures": 0,
                                           "total_time_seconds": 0.0})


# ── capacity math + presence in every status ───────────────────────────────


def test_health_reports_capacity_and_effective_limits(api_client):
    body = api_client.get("/health").json()
    conc = body["concurrency"]
    assert conc["workers"] == 2
    assert conc["pipeline"] == {"per_worker_limit": 1, "effective_limit": 2}
    assert conc["vlm"] == {"limit": 4, "cross_process": True, "in_flight": 0}
    assert conc["conversions"] == {"running": 0, "queued": 0}


def test_unhealthy_response_still_carries_concurrency(api_client):
    FakeVLMClient.ready = False
    resp = api_client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unhealthy"
    assert body["concurrency"]["pipeline"]["effective_limit"] == 2


def test_degraded_response_still_carries_concurrency(api_client, monkeypatch):
    monkeypatch.setattr(server_module, "empty_200_streak",
                        lambda: server_module.EMPTY_200_ALERT_THRESHOLD)
    resp = api_client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert "concurrency" in body


def test_health_carries_no_secrets_in_concurrency(api_client):
    conc = api_client.get("/health").json()["concurrency"]
    blob = json.dumps(conc)
    assert "http" not in blob and "/" not in blob.replace("\\/", "")
    assert set(conc) == {"workers", "pipeline", "vlm", "conversions"}


# ── live activity through the endpoints ─────────────────────────────────────


def test_convert_transitions_queued_to_running_and_releases(api_client, monkeypatch):
    fake = _BlockingPipeline()
    monkeypatch.setattr(server_module, "run_structured_pipeline", fake)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(api_client.post, "/v1/convert", files=_upload_png())
        assert fake.entered.wait(10), "first conversion never started"
        _wait_until(lambda: _conversions(api_client) == {"running": 1, "queued": 0},
                    msg="first request running")

        # Second request queues behind the size-1 gate: queued, NOT running.
        fut2 = pool.submit(api_client.post, "/v1/convert", files=_upload_png())
        _wait_until(lambda: _conversions(api_client) == {"running": 1, "queued": 1},
                    msg="second request queued")

        fake.release.set()
        assert fut1.result(timeout=30).status_code == 200
        assert fut2.result(timeout=30).status_code == 200
    _wait_until(lambda: _conversions(api_client) == {"running": 0, "queued": 0},
                msg="all leases released")


def test_stream_and_convert_share_one_pipeline_gate(api_client, monkeypatch):
    """With a size-1 gate held by a non-stream conversion, a STREAM conversion
    must queue — the streaming route has no separate capacity."""
    fake = _BlockingPipeline()
    monkeypatch.setattr(server_module, "run_structured_pipeline", fake)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(api_client.post, "/v1/convert", files=_upload_png())
        assert fake.entered.wait(10)
        _wait_until(lambda: _conversions(api_client)["running"] == 1,
                    msg="convert running")

        fut2 = pool.submit(api_client.post, "/v1/convert/stream", files=_upload_png())
        _wait_until(lambda: _conversions(api_client) == {"running": 1, "queued": 1},
                    msg="stream queued behind convert")

        fake.release.set()
        assert fut1.result(timeout=30).status_code == 200
        assert fut2.result(timeout=30).status_code == 200
    _wait_until(lambda: _conversions(api_client) == {"running": 0, "queued": 0},
                msg="all leases released")


def test_stream_conversion_is_tracked(api_client, monkeypatch):
    fake = _BlockingPipeline()
    monkeypatch.setattr(server_module, "run_structured_pipeline", fake)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(api_client.post, "/v1/convert/stream", files=_upload_png())
        assert fake.entered.wait(10), "stream conversion never started"
        _wait_until(lambda: _conversions(api_client) == {"running": 1, "queued": 0},
                    msg="stream running")
        fake.release.set()
        assert fut.result(timeout=30).status_code == 200
    _wait_until(lambda: _conversions(api_client) == {"running": 0, "queued": 0},
                msg="stream lease released")


def test_pipeline_error_releases_lease(api_client, monkeypatch):
    def _boom(*a, **k):
        raise ValueError("synthetic pipeline failure")
    monkeypatch.setattr(server_module, "run_structured_pipeline", _boom)
    resp = api_client.post("/v1/convert", files=_upload_png())
    assert resp.status_code == 500
    assert _conversions(api_client) == {"running": 0, "queued": 0}


def test_queued_request_cancelled_by_deadline_releases_lease(api_client, monkeypatch):
    """A request cancelled while WAITING for capacity (deadline fires in the
    queue) must release its queued lease and answer 499 — the running
    conversion is untouched."""
    monkeypatch.setenv("SHREW_CONVERT_DEADLINE_S", "1")
    fake = _BlockingPipeline()
    monkeypatch.setattr(server_module, "run_structured_pipeline", fake)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(api_client.post, "/v1/convert", files=_upload_png())
        assert fake.entered.wait(10)
        _wait_until(lambda: _conversions(api_client)["running"] == 1,
                    msg="first running")
        # Queued request hits the 1s deadline before capacity frees up.
        # (The deadline also cancels fut1's watcher-cancelled pipeline: the
        # fake honors cancellation, so fut1 comes back 499 as well.)
        fut2 = pool.submit(api_client.post, "/v1/convert", files=_upload_png())
        assert fut2.result(timeout=30).status_code == 499
        _wait_until(lambda: _conversions(api_client)["queued"] == 0,
                    msg="queued lease released")
        fake.release.set()
        fut1.result(timeout=30)
    _wait_until(lambda: _conversions(api_client) == {"running": 0, "queued": 0},
                msg="all leases released")


def test_health_never_calls_inference(api_client, monkeypatch):
    calls = []
    monkeypatch.setattr(FakeVLMClient, "chat_completion",
                        lambda self, *a, **k: calls.append(1))
    for _ in range(3):
        api_client.get("/health")
    assert calls == []


def test_vlm_gate_serializes_across_clients(monkeypatch, tmp_path):
    """VLM_CONCURRENCY=1: two concurrent chat_completion calls must serialize
    on the machine-wide slot gate — never overlap."""
    import app.vlm_client as vc
    monkeypatch.setenv("SHREW_CONCURRENCY_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("VLM_CONCURRENCY", "1")

    active = []
    overlap = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        active.append(1)
        if len(active) > 1:
            overlap.append(len(active))
        time.sleep(0.15)
        active.pop()

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"finish_reason": "stop",
                                      "message": {"content": "{}"}}],
                        "usage": {}}
        return R()

    monkeypatch.setattr(vc.requests, "post", _fake_post)
    client = vc.VLMClient("http://unused", "m")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(client.chat_completion, [{"role": "user", "content": "x"}])
                for _ in range(3)]
        for f in futs:
            f.result(timeout=30)
    assert overlap == [], f"calls overlapped despite VLM_CONCURRENCY=1: {overlap}"
