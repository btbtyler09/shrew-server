"""Rasterizer crash isolation (work-prod hardening).

Production failure being pinned down: shrew-server exited 139 (SIGSEGV in
native code) while rasterizing an 889-page PDF under overlapped
/v1/convert/stream requests — no OOM, 37 MB process peak. Two defects:

1. Lifetime: the final loop iteration's page/bitmap locals outlived
   doc.close(), so their finalizers touched a freed PDFium document.
2. Containment: PDFium finalizers run on whichever thread drops the last
   reference, so RASTERIZE_LOCK can't serialize native teardown — and any
   PDFium fault killed the whole server with every in-flight request.

The fix routes all PDFium work through a spawned worker process. These tests
pin the contract: a worker segfault or hang becomes a RuntimeError for that
one document, and the server process keeps serving the next one.
"""
import os

import pypdfium2
import pytest

from app.rasterizer import (
    RASTERIZE_LOCK,
    get_page_count_and_dims,
    prepare_pages,
)


@pytest.fixture
def two_page_pdf(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    doc = pypdfium2.PdfDocument.new()
    doc.new_page(612, 792)   # US letter
    doc.new_page(612, 1008)  # US legal — distinct dims to assert per-page
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_isolated_roundtrip_renders_and_reports_dims(two_page_pdf, tmp_path):
    out = tmp_path / "out"
    page_images, total, dims = prepare_pages(two_page_pdf, str(out))
    assert total == 2
    assert set(page_images) == {1, 2}
    for display, hires in page_images.values():
        assert display.exists() and hires.exists()
    assert dims[1] == (612, 792)
    assert dims[2] == (612, 1008)


def test_worker_segfault_is_contained_and_server_recovers(
        two_page_pdf, tmp_path, monkeypatch):
    # The env var crosses the spawn boundary (monkeypatched attrs don't) and
    # makes the worker SIGSEGV itself before touching PDFium.
    monkeypatch.setenv("SHREW_RASTER_TEST_CRASH", "1")
    with pytest.raises(RuntimeError, match="worker crashed"):
        prepare_pages(two_page_pdf, str(tmp_path / "a"))

    # The contract that matters in production: the process that just lost a
    # worker serves the next document normally.
    monkeypatch.delenv("SHREW_RASTER_TEST_CRASH")
    page_images, total, _ = prepare_pages(two_page_pdf, str(tmp_path / "b"))
    assert total == 2 and len(page_images) == 2


def test_hung_worker_times_out_and_lock_is_released(
        two_page_pdf, tmp_path, monkeypatch):
    monkeypatch.setenv("SHREW_RASTER_TEST_HANG", "60")
    monkeypatch.setenv("SHREW_RASTER_TIMEOUT_S", "2")
    with pytest.raises(RuntimeError, match="timed out"):
        prepare_pages(two_page_pdf, str(tmp_path / "a"))

    # The throttle lock must not stay held by the dead call.
    assert RASTERIZE_LOCK.acquire(timeout=1)
    RASTERIZE_LOCK.release()

    monkeypatch.delenv("SHREW_RASTER_TEST_HANG")
    monkeypatch.delenv("SHREW_RASTER_TIMEOUT_S")
    _, total, _ = prepare_pages(two_page_pdf, str(tmp_path / "b"))
    assert total == 2


def test_inproc_escape_hatch(two_page_pdf, tmp_path, monkeypatch):
    monkeypatch.setenv("SHREW_RASTER_INPROC", "1")
    page_images, total, dims = prepare_pages(two_page_pdf, str(tmp_path / "out"))
    assert total == 2 and set(page_images) == {1, 2}
    assert dims[2] == (612, 1008)


def test_count_and_dims_without_rendering(two_page_pdf):
    total, dims = get_page_count_and_dims(two_page_pdf)
    assert total == 2
    assert dims == {1: (612, 792), 2: (612, 1008)}


def test_high_page_count_repeated_rasterization(tmp_path):
    """Regression for the production exit-139: the crash needed a large
    document (heap churn makes the freed-parent access actually fault) and
    fired at the END of rasterization — 'Rasterized 889/889' then SIGSEGV.
    Two consecutive full-document runs in one process pin clean completion
    including the teardown boundary where the old code died."""
    pdf_path = tmp_path / "big.pdf"
    doc = pypdfium2.PdfDocument.new()
    for _ in range(200):
        doc.new_page(612, 792)
    doc.save(str(pdf_path))
    doc.close()

    for run in ("a", "b"):
        page_images, total, dims = prepare_pages(
            str(pdf_path), str(tmp_path / run), low_dpi=40, high_dpi=60)
        assert total == 200
        assert len(page_images) == 200
        assert len(dims) == 200
