"""v0.3.2 hardening batch — regression tests for audit issues #5–#14.

Each test pins one fix from the 2026-08-25 robustness audit (GitLab
shrew-server-private issues #5–#14). The archetype for the batch was the
production PDFium exit-139: protections that silently weren't there.
"""
import os
import time

import pytest
from PIL import Image

from app import spreadsheet_extract as se
from app import text_extract as te
from app import structured_pipeline as spl
from app.rasterizer import prepare_image_pages, run_libreoffice


# ── #5: workbook picture captioning ─────────────────────────────────────────


def _fake_media(tmp_path):
    p = tmp_path / "img0.png"
    Image.new("RGB", (64, 64), "white").save(str(p))
    return [{"path": str(p), "sheet": "S1", "index": 0}]


def test_caption_attaches_from_extract_page_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.spreadsheet_extract.extract_spreadsheet_media",
        lambda path, output_dir, max_images=20: _fake_media(tmp_path))
    monkeypatch.setattr(
        spl, "extract_page",
        lambda *a, **k: {"ok": True,
                         "data": {"figures": [{"caption": "A bar chart"}],
                                  "summary": "unused"}})
    out = spl._extract_and_caption_media("wb.xlsx", str(tmp_path), client=object())
    assert len(out) == 1
    # v0.3.1 checked res["structured"] (a key that never exists) — captions
    # could never attach.
    assert "A bar chart" in out[0]["caption"]


def test_caption_failure_degrades_without_nameerror(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.spreadsheet_extract.extract_spreadsheet_media",
        lambda path, output_dir, max_images=20: _fake_media(tmp_path))

    def _boom(*a, **k):
        raise RuntimeError("VLM down")
    monkeypatch.setattr(spl, "extract_page", _boom)
    # v0.3.1: the except handler referenced an undefined `logger` → NameError
    # → the whole workbook conversion 500'd instead of degrading.
    out = spl._extract_and_caption_media("wb.xlsx", str(tmp_path), client=object())
    assert len(out) == 1
    assert out[0]["caption"].startswith("Embedded image")


# ── #6: spreadsheet grid clamps ─────────────────────────────────────────────


def test_sparse_far_cell_does_not_allocate_used_range(tmp_path):
    """A1 + a value at (50000, 300) used to demand a 15M-cell grid; the
    crafted-worst-case (XFD1048576) was 17 billion cells — OOM kill."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "header"
    ws.cell(row=50000, column=300, value="far")
    t0 = time.time()
    md = se._build_data_table(ws)
    assert time.time() - t0 < 60
    assert md.startswith("|")
    # Clamped: SPREADSHEET_MAX_ROWS data rows, SPREADSHEET_MAX_COLS columns.
    header_cells = md.splitlines()[0].count("|") - 1
    assert header_cells <= se.SPREADSHEET_MAX_COLS
    assert len(md.splitlines()) <= se.SPREADSHEET_MAX_ROWS + 2
    wb.close()


def test_zip_size_preflight_rejects_declared_bombs(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "wb.xlsx"
    wb = openpyxl.Workbook()
    wb.active["A1"] = "x" * 1000
    wb.save(str(p))
    wb.close()
    monkeypatch.setattr(se, "SPREADSHEET_MAX_UNCOMPRESSED_MB", 0.0001)
    with pytest.raises(ValueError, match="zip-bomb"):
        se.check_workbook_zip_size(str(p))
    # Non-zip input passes through (loader reports its own error).
    txt = tmp_path / "not_a_zip.xlsx"
    txt.write_text("hello")
    se.check_workbook_zip_size(str(txt))


# ── #8: image inputs are isolated + frame-capped ────────────────────────────


def _multiframe_tiff(tmp_path, n):
    frames = [Image.new("RGB", (100, 80), (i * 20 % 255, 0, 0)) for i in range(n)]
    p = tmp_path / "pages.tiff"
    frames[0].save(str(p), save_all=True, append_images=frames[1:])
    return str(p)


def test_multiframe_tiff_roundtrip_through_isolation(tmp_path):
    path = _multiframe_tiff(tmp_path, 3)
    page_images, total, dims = prepare_image_pages(path, str(tmp_path / "out"))
    assert total == 3 and set(page_images) == {1, 2, 3}
    for d, h in page_images.values():
        assert d.exists() and h.exists()


def test_tiff_frame_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_MAX_FRAMES", "2")
    path = _multiframe_tiff(tmp_path, 5)
    page_images, total, _ = prepare_image_pages(path, str(tmp_path / "out"))
    assert total == 2 and set(page_images) == {1, 2}


def test_image_decoder_crash_is_contained(tmp_path, monkeypatch):
    """Same containment contract as the PDF arm: a native codec fault costs
    one conversion, not the server (the archetype exit-139 class)."""
    monkeypatch.setenv("SHREW_RASTER_TEST_CRASH", "1")
    path = _multiframe_tiff(tmp_path, 2)
    with pytest.raises(RuntimeError, match="worker crashed"):
        prepare_image_pages(path, str(tmp_path / "out"))
    monkeypatch.delenv("SHREW_RASTER_TEST_CRASH")
    _, total, _ = prepare_image_pages(path, str(tmp_path / "out2"))
    assert total == 2


# ── #9: text-family size bound ──────────────────────────────────────────────


def test_oversize_text_rejected_not_truncated(tmp_path, monkeypatch):
    p = tmp_path / "big.txt"
    p.write_text("hello world\n" * 100)
    monkeypatch.setattr(te, "MAX_TEXT_MB", 0.000001)
    with pytest.raises(ValueError, match="MAX_TEXT_MB"):
        te.extract_txt(str(p))
    monkeypatch.setattr(te, "MAX_TEXT_MB", 50.0)
    assert "hello world" in te.extract_txt(str(p))


# ── #10: LibreOffice isolation + group kill ─────────────────────────────────


def test_libreoffice_timeout_kills_process_group(tmp_path):
    """The wrapper's children must die with it: an orphaned soffice.bin
    holding the profile lock used to poison every later conversion."""
    fake = tmp_path / "fake_soffice"
    marker = tmp_path / "grandchild.pid"
    fake.write_text(
        "#!/bin/sh\n"
        f"sleep 300 & echo $! > {marker}\n"
        "wait\n")
    fake.chmod(0o755)
    src = tmp_path / "doc.rtf"
    src.write_text("x")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        run_libreoffice("pdf", str(src), str(tmp_path), timeout=1.0,
                        binary=str(fake))
    assert time.time() - t0 < 10
    # the grandchild must be dead too (process-group SIGKILL)
    time.sleep(0.2)
    pid = int(marker.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_libreoffice_profiles_are_unique_and_cleaned(tmp_path):
    fake = tmp_path / "fake_ok"
    profiles = tmp_path / "seen.txt"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$2" >> {profiles}\n'  # $2 = -env:UserInstallation=...
        "exit 0\n")
    fake.chmod(0o755)
    src = tmp_path / "doc.rtf"
    src.write_text("x")
    run_libreoffice("pdf", str(src), str(tmp_path), binary=str(fake))
    run_libreoffice("pdf", str(src), str(tmp_path), binary=str(fake))
    lines = profiles.read_text().strip().splitlines()
    assert len(lines) == 2 and lines[0] != lines[1]
    for line in lines:
        assert line.startswith("-env:UserInstallation=file://")
        assert not os.path.exists(line.split("file://", 1)[1])


# ── #11: startup sweep ──────────────────────────────────────────────────────


def test_stale_tmp_sweep(monkeypatch, tmp_path):
    from app import server
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    old_dir = tmp_path / "shrew_oldrun"
    old_dir.mkdir()
    (old_dir / "page.png").write_bytes(b"x")
    old_up = tmp_path / "shrew_up_stale.pdf"
    old_up.write_bytes(b"x")
    fresh = tmp_path / "shrew_live"
    fresh.mkdir()
    other = tmp_path / "unrelated.file"
    other.write_bytes(b"x")
    stale = time.time() - 48 * 3600
    os.utime(old_dir, (stale, stale))
    os.utime(old_up, (stale, stale))

    removed = server._sweep_stale_tmp(max_age_h=24)
    assert removed == 2
    assert not old_dir.exists() and not old_up.exists()
    assert fresh.exists() and other.exists()
