"""HTTP-level acceptance for the run-telemetry wiring (GitLab #24): a real
conversion leaves a durable content-free trace, a failing one records died_at,
and /health surfaces live in-flight progress."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

import app.server as server
from app.server import ServerConfig, app

GOOD = json.dumps({"metadata": {"title": "T", "authors": [], "organization": None,
                                "year": None, "doc_type": "report"}, "summary": "s",
                   "semantic_chunks": [], "figures": [], "tables": []})


class FakeVLM:
    ready = True

    def __init__(self, *a, **k):
        self.model = "shrew-9b"

    def is_ready(self):
        return True

    def readiness_snapshot(self):
        return {"ready": True, "age_s": 0.0, "ever_ok": True}

    def health_check(self, timeout=10):
        return True

    def chat_completion(self, *a, **k):
        return {"choices": [{"finish_reason": "stop", "message": {"content": GOOD}}]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    tel = tmp_path / "tel"
    monkeypatch.setenv("SHREW_TELEMETRY_DIR", str(tel))
    monkeypatch.setenv("SHREW_CONCURRENCY_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("SHREW_TELEMETRY_SAMPLE_S", "0")  # no background sampler
    monkeypatch.setattr("app.server.VLMClient", FakeVLM)
    monkeypatch.setattr("app.structured_pipeline.VLMClient", FakeVLM)
    monkeypatch.setattr(server, "_config", ServerConfig(
        vlm_url="http://fake", vlm_model="shrew-9b",
        workers=1, pipeline_concurrency=1, vlm_concurrency=4))
    monkeypatch.setattr(server, "_figure_converter", None)
    monkeypatch.setattr(server, "_shrew_lora_map", None)
    monkeypatch.setattr(server, "_shrew_lora_format", "none")
    monkeypatch.setattr(server, "_vlm_pool", ThreadPoolExecutor(max_workers=2))
    monkeypatch.setattr(server, "_pipeline_gate", threading.Semaphore(1))
    return TestClient(app), tel


def _png(tmp_path):
    p = tmp_path / "doc.png"
    Image.new("RGB", (1700, 2200), "white").save(p)
    return p


def _traces(tel_dir):
    return [json.loads(l) for f in tel_dir.glob("*.jsonl")
            for l in open(f) if l.strip()]


def test_convert_leaves_a_durable_content_free_trace(client, tmp_path):
    tc, tel_dir = client
    with open(_png(tmp_path), "rb") as f:
        r = tc.post("/v1/convert", files={"file": ("doc.png", f, "image/png")},
                    data={"pipeline_mode": "structured"})
    assert r.status_code == 200, r.text

    evs = _traces(tel_dir)
    kinds = [e["ev"] for e in evs]
    assert kinds[0] == "open" and kinds[-1] == "done"
    assert evs[-1]["status"] == "done"
    phases = {e.get("phase") for e in evs if e["ev"] == "phase"}
    assert {"rasterize_done", "transcribe_start", "serialize_done"} <= phases
    for e in evs:  # content-free: every string is a short controlled token
        for v in e.values():
            if isinstance(v, str):
                assert len(v) <= 40 and "\n" not in v


def test_failed_conversion_records_died_at_not_message(client, tmp_path, monkeypatch):
    tc, tel_dir = client
    secret = "boom near 'Confidential: account 4111-1111-1111-1111'"

    def _explode(*a, **k):
        raise RuntimeError(secret)
    monkeypatch.setattr(server, "run_structured_pipeline", _explode)

    with open(_png(tmp_path), "rb") as f:
        r = tc.post("/v1/convert", files={"file": ("doc.png", f, "image/png")},
                    data={"pipeline_mode": "structured"})
    assert r.status_code == 500

    evs = _traces(tel_dir)
    died = [e for e in evs if e["ev"] == "died_at"]
    assert len(died) == 1 and died[0]["exc_type"] == "RuntimeError"
    assert died[0]["category"] == "other"
    blob = json.dumps(evs)
    assert "4111" not in blob and "Confidential" not in blob and secret not in blob


def test_health_exposes_live_active_runs(client, tmp_path):
    tc, tel_dir = client
    from app import telemetry
    tr = telemetry.RunTrace("inflight", total_pages=3500, dir=str(tel_dir),
                            sample_interval_s=0)
    tr.phase("transcribe_start")
    tr.mark_pages_done(1800)
    tr.publish_live()
    try:
        h = tc.get("/health").json()
        active = h["concurrency"]["conversions"].get("active", [])
        mine = [a for a in active if a["total_pages"] == 3500]
        assert mine and mine[0]["phase"] == "transcribe_start"
        assert mine[0]["pages_done"] == 1800
    finally:
        tr.close("done")
