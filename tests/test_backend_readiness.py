"""v0.3.15 (GitLab #28): readiness must fail CLOSED once the pipeline has seen
consecutive transport errors and a socket re-probe cannot confirm the backend.

Found live 2026-09-15: after a model-serve swap, /health kept reporting "ok" with a
warm `vlm_readiness` cache (ever_ok) while every /v1/convert page came back
`transport_error` — a gate run would have read as a 0% first pass. The v0.3.4
fail-open rule (a busy backend must not reject queued work) is kept for the
un-tripped case; the trip adds the missing "the backend is actually gone" signal.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

import app.server as server
import app.vlm_client as vc
from app.server import ServerConfig, app
from app.vlm_client import VLMClient

URL, MODEL = "http://fake-backend", "shrew-ocr-preview"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    vc._readiness.clear(); vc.reset_transport()
    monkeypatch.setattr(vc, "VLM_TRANSPORT_TRIP", 3)
    yield
    vc._readiness.clear(); vc.reset_transport()


def _warm_cache():
    VLMClient._mark_readiness(vc._readiness_key(URL, MODEL), True)  # fresh healthy stamp, ever_ok


def test_untripped_keeps_the_fail_open_rule(monkeypatch):
    _warm_cache()
    c = VLMClient(base_url=URL, model=MODEL)
    monkeypatch.setattr(c, "probe", lambda **k: ("soft", "timeout"))
    assert c.is_ready() is True                       # warm cache fast path
    vc.note_transport_error(URL, MODEL); vc.note_transport_error(URL, MODEL)
    assert vc.transport_snapshot(URL, MODEL)["tripped"] is False
    assert c.is_ready() is True                       # below the trip: still admitted


def test_trip_bypasses_the_cache_and_fails_closed(monkeypatch):
    _warm_cache()
    c = VLMClient(base_url=URL, model=MODEL)
    probes = []
    monkeypatch.setattr(c, "probe", lambda **k: probes.append(1) or ("soft", "connection refused"))
    for _ in range(3):
        vc.note_transport_error(URL, MODEL)
    snap = vc.transport_snapshot(URL, MODEL)
    assert snap["tripped"] is True and snap["consecutive"] == 3
    assert c.is_ready() is False                      # ever_ok no longer admits
    assert probes, "a tripped client must re-probe the socket instead of trusting the cache"


def test_successful_probe_clears_the_trip(monkeypatch):
    _warm_cache()
    c = VLMClient(base_url=URL, model=MODEL)
    for _ in range(3):
        vc.note_transport_error(URL, MODEL)
    monkeypatch.setattr(c, "probe", lambda **k: ("ok", None))
    assert c.is_ready() is True
    snap = vc.transport_snapshot(URL, MODEL)
    assert snap["tripped"] is False and snap["consecutive"] == 0


def test_a_completed_page_resets_the_streak():
    vc.note_transport_error(URL, MODEL); vc.note_transport_error(URL, MODEL)
    vc.note_transport_ok(URL, MODEL)
    vc.note_transport_error(URL, MODEL)
    assert vc.transport_snapshot(URL, MODEL)["tripped"] is False
    assert vc.transport_snapshot(URL, MODEL)["consecutive"] == 1
    assert vc.transport_snapshot(URL, MODEL)["recent"] == 3  # window count keeps all three


def test_snapshot_carries_transport_errors():
    _warm_cache()
    vc.note_transport_error(URL, MODEL)
    s = VLMClient(base_url=URL, model=MODEL).readiness_snapshot()
    assert s["ready"] is True and s["transport_errors"]["consecutive"] == 1
    assert s["transport_errors"]["trip_at"] == 3


# ── /health ──────────────────────────────────────────────────────────────────

@pytest.fixture
def health_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("SHREW_CONCURRENCY_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("SHREW_TELEMETRY_DIR", str(tmp_path / "tel"))
    monkeypatch.setattr(server, "_config", ServerConfig(
        vlm_url=URL, vlm_model=MODEL, workers=1, pipeline_concurrency=1, vlm_concurrency=4))
    monkeypatch.setattr(server, "_vlm_pool", ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(server, "_pipeline_gate", threading.Semaphore(1))
    return TestClient(app)


def test_health_reports_degraded_with_counts_when_tripped(health_client, monkeypatch):
    _warm_cache()
    monkeypatch.setattr(VLMClient, "probe", lambda self, **k: ("soft", "connection refused"))
    r = health_client.get("/health")
    assert r.status_code == 200 and r.json()["transport_errors"]["consecutive"] == 0
    for _ in range(3):
        vc.note_transport_error(URL, MODEL)
    r = health_client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded" and body["transport_errors"]["tripped"] is True
    assert "consecutive transport errors" in body["detail"]


def test_health_recovers_when_the_socket_comes_back(health_client, monkeypatch):
    _warm_cache()
    for _ in range(3):
        vc.note_transport_error(URL, MODEL)
    monkeypatch.setattr(VLMClient, "probe", lambda self, **k: ("ok", None))
    r = health_client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["transport_errors"]["tripped"] is False


# ── pipeline wiring ──────────────────────────────────────────────────────────

def test_pipeline_feeds_transport_outcomes(tmp_path, monkeypatch):
    from app import structured_pipeline as sp
    from app.models import PipelineConfig

    class Boom:
        base_url, model = URL, MODEL

    monkeypatch.setattr(sp, "extract_page", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("refused")))
    monkeypatch.setattr(sp, "prepare_image_bucketed", lambda img, g: (img.copy(), "B1"))
    monkeypatch.setattr(sp, "cached_glyph_height", lambda *a, **k: 10.0)
    png = tmp_path / "p.png"; Image.new("RGB", (100, 100), "white").save(png)
    cfg = PipelineConfig()
    for i in range(3):
        r = sp._process_one_page(i + 1, str(png), cfg, str(tmp_path), Boom())
        assert r["status"] == "transport_error"
    assert vc.transport_snapshot(URL, MODEL)["tripped"] is True
    monkeypatch.setattr(sp, "extract_page", lambda *a, **k: {"ok": True, "data": {}, "status": "ok",
                                                             "error": None, "attempts": 1, "raw_len": 10})
    sp._process_one_page(4, str(png), cfg, str(tmp_path), Boom())
    assert vc.transport_snapshot(URL, MODEL)["consecutive"] == 0
