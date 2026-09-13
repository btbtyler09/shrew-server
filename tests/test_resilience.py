"""v0.3.12 resilience, root-caused from the 3,500-page handbook run:

- vLLM's engine died at page ~2014 (GPU OOM); the next ~1,490 pages all came
  back transport_error, yet shrew-server ground through every one for ~an hour
  and returned HTTP 200 on a 44%-empty document. A dead backend must
  CIRCUIT-BREAK: abort fast with a clear error, not fake success.
- A page that fails transcription must NEVER silently vanish (the user's
  "we never skip pages"): it ships its rendered image instead.
"""
import json

import pytest
from PIL import Image

from app.assembly import assemble_document
from app.models import PipelineConfig
from app.structured_pipeline import (
    ModelBackendDownError,
    build_structured_json,
    run_structured_pipeline,
    synthesize_markdown,
)


def _page(page, ok=True, status="ok"):
    if ok:
        data = {"metadata": {"title": None, "authors": [], "organization": None,
                             "year": None, "doc_type": None},
                "summary": "s", "semantic_chunks": [], "figures": [], "tables": []}
        return {"page": page, "ok": True, "data": data, "status": "ok"}
    return {"page": page, "ok": False, "data": None, "status": status}


# ── never-skip: a failed page ships its rendered image ───────────────────────


def test_failed_page_ships_its_render_not_nothing(tmp_path):
    # page 2 failed transcription; its hires render exists.
    render = tmp_path / "p2.png"
    Image.new("RGB", (120, 160), "white").save(render)
    p1 = tmp_path / "p1.png"
    Image.new("RGB", (120, 160), "white").save(p1)
    hires = {1: str(p1), 2: str(render)}
    doc = assemble_document(
        "docid", "f.pdf", "pdf",
        [_page(1, ok=True), _page(2, ok=False, status="degenerate")],
        hires_images=hires, stitch=False, crops_dir=str(tmp_path / "crops"))

    fails = [f for f in doc["figures"] if "Page 2" in (f.get("caption") or "")]
    assert len(fails) == 1, doc["figures"]
    f = fails[0]
    assert f["figure_id"] and f["page"] == 2
    assert f["crop_path"] == str(render), "must point at the page render"
    assert "degenerate" in f["caption"]

    # it renders into markdown (an image ref, not a blank page) ...
    md = synthesize_markdown(doc)
    fig_index = {fg["figure_id"]: i for i, fg in enumerate(doc["figures"], 1)}
    assert f"(img:{fig_index[f['figure_id']]})" in md
    # ... and into the structured images[] with real png data
    sj = build_structured_json(doc, total_pages=2)
    img = [i for i in sj["images"] if i["page"] == 2]
    assert img and img[0]["data"] and img[0]["format"] == "png"


def test_ok_pages_get_no_page_fail_figure(tmp_path):
    render = tmp_path / "p1.png"
    Image.new("RGB", (80, 100), "white").save(render)
    doc = assemble_document("d", "f.pdf", "pdf", [_page(1, ok=True)],
                            hires_images={1: str(render)}, stitch=False,
                            crops_dir=str(tmp_path / "c"))
    assert not any("transcription unavailable" in (f.get("caption") or "")
                   for f in doc["figures"])


# ── circuit-break: a dead backend aborts fast, no false 200 ──────────────────


class _DeadBackendClient:
    """Every call fails as if the model server is gone."""
    model = "shrew-ocr-preview"

    def __init__(self):
        self.calls = 0

    def chat_completion(self, *a, **k):
        self.calls += 1
        raise ConnectionError("connection refused (engine dead)")


def _multipage_pdf(path, n):
    pages = [Image.new("RGB", (1000, 1300), "white") for _ in range(n)]
    pages[0].save(path, save_all=True, append_images=pages[1:])


def test_backend_death_circuit_breaks_instead_of_grinding(tmp_path, monkeypatch):
    monkeypatch.setenv("SHREW_BACKEND_DOWN_STREAK", "3")
    pdf = tmp_path / "book.pdf"
    _multipage_pdf(pdf, 12)  # far more pages than the streak
    client = _DeadBackendClient()
    # concurrency=1 makes the abort deterministic: fail x3 -> break.
    cfg = PipelineConfig(vlm_url="http://unused", vlm_model="shrew-ocr-preview",
                         vlm_concurrency=1)

    with pytest.raises(ModelBackendDownError):
        run_structured_pipeline(str(pdf), str(tmp_path / "out"), cfg, client=client)

    # It must abort at the streak, not grind through all 12 pages.
    assert client.calls <= 4, f"kept calling a dead backend: {client.calls}"


def test_sporadic_failures_do_not_trip_the_breaker(tmp_path, monkeypatch):
    """A backend that fails a couple pages then recovers is NOT a dead backend
    — the run completes, failed pages ride through as image placeholders."""
    monkeypatch.setenv("SHREW_BACKEND_DOWN_STREAK", "5")
    pdf = tmp_path / "book.pdf"
    _multipage_pdf(pdf, 6)

    good = json.dumps({"metadata": {"title": None, "authors": [], "organization": None,
                                    "year": None, "doc_type": None},
                       "summary": "s", "semantic_chunks": [], "figures": [], "tables": []})

    class _Flaky:
        model = "shrew-ocr-preview"
        def __init__(self): self.n = 0
        def chat_completion(self, *a, **k):
            self.n += 1
            if self.n in (1, 2):  # two early failures, then healthy
                raise ConnectionError("blip")
            return {"choices": [{"finish_reason": "stop", "message": {"content": good}}]}

    res = run_structured_pipeline(
        str(pdf), str(tmp_path / "out"),
        PipelineConfig(vlm_url="x", vlm_model="m", vlm_concurrency=1),
        client=_Flaky())
    assert res.structured_json  # completed, no raise
    # the two failed pages rode through as image placeholders, not dropped
    fails = [i for i in res.structured_json["images"]
             if "transcription unavailable" in (i.get("caption") or "")]
    assert len(fails) == 2
