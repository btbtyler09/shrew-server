"""Server-level tests: /v1/convert routes through the v2 structured pipeline
by default, exposes the new `tables` key, and shares one rasterize lock with
the legacy pipeline.

The model is never hit — VLMClient is monkeypatched everywhere it gets
constructed (server.py's preflight health check AND structured_pipeline's
per-page extraction calls) with a fake that returns canned 5-key JSON.
"""

import asyncio
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.pipeline as pipeline_module
import app.rasterizer as rasterizer_module
import app.server as server_module
from app.server import ServerConfig, app

GOOD_PAGE_JSON = json.dumps({
    "metadata": {"title": "Doc Title", "authors": ["A"], "organization": None,
                 "year": None, "doc_type": "report"},
    "summary": "Page 1 summary.",
    "semantic_chunks": [{"chunk_id": "1", "title": "Intro",
                          "content": "Some intro content.", "keywords": [],
                          "section_type": "introduction"}],
    "figures": [],
    "tables": [],
})


class FakeVLMClient:
    """Stands in for app.vlm_client.VLMClient everywhere it's constructed —
    server.py's preflight health check AND structured_pipeline's per-page
    extraction. Never touches the network."""

    def __init__(self, base_url=None, model=None, api_key=None, **kwargs):
        self.base_url = base_url
        self.model = model or "shrew-9b"
        self.api_key = api_key

    def health_check(self, timeout=10):
        return True

    def chat_completion(self, messages, max_tokens=8192, temperature=0.2,
                         timeout=None, extra_params=None):
        return {"choices": [{"finish_reason": "stop",
                              "message": {"content": GOOD_PAGE_JSON}}]}


@pytest.fixture
def api_client(monkeypatch):
    # Bypass real network calls: server.py's preflight VLM health check and
    # structured_pipeline's per-page model calls both construct VLMClient.
    monkeypatch.setattr("app.server.VLMClient", FakeVLMClient)
    monkeypatch.setattr("app.structured_pipeline.VLMClient", FakeVLMClient)

    # Minimal server globals so the endpoints don't 503 — mirrors what
    # lifespan() sets, without hitting real docling/VLM auto-detect at
    # startup (TestClient() without a `with` block never runs lifespan).
    monkeypatch.setattr(server_module, "_config", ServerConfig(
        vlm_url="http://fake-vlm", vlm_model="shrew-9b",
    ))
    monkeypatch.setattr(server_module, "_figure_converter", None)
    monkeypatch.setattr(server_module, "_shrew_lora_map", None)
    monkeypatch.setattr(server_module, "_shrew_lora_format", "none")
    monkeypatch.setattr(server_module, "_vlm_pool", ThreadPoolExecutor(max_workers=2))
    monkeypatch.setattr(server_module, "_pipeline_sem", asyncio.Semaphore(3))
    monkeypatch.setattr(server_module, "_pipeline_gate", threading.Semaphore(3))

    return TestClient(app)


def _upload_png():
    buf = io.BytesIO()
    Image.new("RGB", (400, 500), "white").save(buf, format="PNG")
    buf.seek(0)
    return {"file": ("doc.png", buf, "image/png")}


STAGE3_KEYS = ("metadata", "summary", "semantic_chunks", "tables")
ALWAYS_KEYS = ("markdown", "images", "processing_log")


def test_convert_default_pipeline_mode_routes_through_structured(api_client):
    """Omitting pipeline_mode should route through run_structured_pipeline
    (default is now "structured"), producing the full v2 response shape."""
    resp = api_client.post("/v1/convert", files=_upload_png())
    assert resp.status_code == 200
    body = resp.json()
    for key in (*ALWAYS_KEYS, *STAGE3_KEYS):
        assert key in body, f"missing {key!r} in response: {sorted(body)}"


def test_convert_explicit_structured_mode(api_client):
    resp = api_client.post(
        "/v1/convert", data={"pipeline_mode": "structured"}, files=_upload_png(),
    )
    assert resp.status_code == 200
    body = resp.json()
    for key in (*ALWAYS_KEYS, *STAGE3_KEYS):
        assert key in body


def test_convert_skip_stage3_omits_stage3_keys_and_tables(api_client):
    resp = api_client.post(
        "/v1/convert", data={"skip_stage3": "true"}, files=_upload_png(),
    )
    assert resp.status_code == 200
    body = resp.json()
    for key in ALWAYS_KEYS:
        assert key in body
    for key in STAGE3_KEYS:
        assert key not in body


def test_rasterize_lock_is_shared_between_legacy_and_structured_pipelines():
    """Proves the thread-safety fix: pipeline.py's legacy lock and the
    module-level RASTERIZE_LOCK that structured_pipeline.py wraps
    prepare_pages() with are literally the same lock object."""
    assert pipeline_module._rasterize_lock is rasterizer_module.RASTERIZE_LOCK
