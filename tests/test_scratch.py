"""RAM-only document scratch (app/scratch.py): placement, refusal, /health, 507."""

import asyncio
import errno
import io
import os
import tempfile

import pytest

from app import scratch


def _is_tmpfs(path):
    return os.path.isdir(path) and scratch.fs_type(path) in ("tmpfs", "ramfs")


@pytest.fixture
def isolated(monkeypatch):
    """Let a test call configure()/enforce() with its own env; restore the
    process-wide tempfile/TMPDIR redirection afterwards."""
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for var in ("TMPDIR", "TMP", "TEMP", "SHREW_SCRATCH_DIR", "SHREW_ALLOW_DISK_SCRATCH",
                "SHREW_RAW_DUMP_DIR"):
        if var in os.environ:
            monkeypatch.setenv(var, os.environ[var])
        else:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(scratch, "_configured_dir", scratch._configured_dir)
    monkeypatch.setattr(scratch, "_configure_error", scratch._configure_error)
    return monkeypatch


def test_import_redirects_tempfile_into_scratch():
    import app.server  # noqa: F401  (import performs the redirection)
    d = scratch.status()["dir"]
    assert tempfile.gettempdir() == d
    assert os.environ["TMPDIR"] == d
    fd, p = tempfile.mkstemp(prefix="shrew_up_")
    os.close(fd)
    try:
        assert os.path.dirname(p) == d
    finally:
        os.unlink(p)
    assert oct(os.stat(d).st_mode & 0o777) == "0o700"


def test_refuses_disk_scratch(isolated, tmp_path):
    if _is_tmpfs(str(tmp_path)):
        pytest.skip("pytest tmp dir is itself tmpfs here")
    isolated.setenv("SHREW_SCRATCH_DIR", str(tmp_path / "s"))
    with pytest.raises(RuntimeError, match="refuses to start.*not tmpfs/ramfs"):
        scratch.enforce()


def test_allow_disk_scratch_override(isolated, tmp_path):
    if _is_tmpfs(str(tmp_path)):
        pytest.skip("pytest tmp dir is itself tmpfs here")
    isolated.setenv("SHREW_SCRATCH_DIR", str(tmp_path / "s"))
    isolated.setenv("SHREW_ALLOW_DISK_SCRATCH", "1")
    st = scratch.enforce()
    assert st["in_memory"] is False and st["ok"] is True and st["allow_disk"] is True
    assert tempfile.gettempdir() == str(tmp_path / "s")


@pytest.mark.skipif(not _is_tmpfs("/dev/shm"), reason="no tmpfs /dev/shm")
def test_tmpfs_scratch_accepted(isolated):
    isolated.setenv("SHREW_SCRATCH_DIR", "/dev/shm/shrew-test-scratch")
    st = scratch.enforce()
    assert st["fs"] == "tmpfs" and st["in_memory"] and st["ok"]
    assert st["total_bytes"] > 0 and st["free_bytes"] >= 0


def test_raw_dump_dir_on_disk_refused(isolated, tmp_path):
    if _is_tmpfs(str(tmp_path)) or not _is_tmpfs("/dev/shm"):
        pytest.skip("needs disk tmp_path and tmpfs /dev/shm")
    isolated.setenv("SHREW_SCRATCH_DIR", "/dev/shm/shrew-test-scratch")
    isolated.setenv("SHREW_RAW_DUMP_DIR", str(tmp_path / "raw"))
    with pytest.raises(RuntimeError, match="SHREW_RAW_DUMP_DIR"):
        scratch.enforce()


def test_owned_by_other_uid_or_symlink_is_an_error(isolated, tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    isolated.setenv("SHREW_SCRATCH_DIR", str(link))
    isolated.setenv("SHREW_ALLOW_DISK_SCRATCH", "1")
    with pytest.raises(RuntimeError, match="not a real directory"):
        scratch.enforce()


def test_is_full_error_walks_the_chain():
    inner = OSError(errno.ENOSPC, "No space left on device")
    try:
        try:
            raise inner
        except OSError as e:
            raise RuntimeError("render failed") from e
    except RuntimeError as outer:
        assert scratch.is_full_error(outer)
    assert scratch.is_full_error(OSError(errno.EDQUOT, "quota"))
    assert not scratch.is_full_error(None)


def test_child_env_points_tmp_and_home_at_scratch():
    env = scratch.child_env(home="/dev/shm/x/profile")
    d = scratch.status()["dir"]
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == d
    assert env["HOME"] == env["XDG_CACHE_HOME"] == env["XDG_CONFIG_HOME"] == "/dev/shm/x/profile"


def test_libreoffice_child_gets_scratch_env(tmp_path):
    from app.rasterizer import run_libreoffice
    seen = tmp_path / "env.txt"
    fake = tmp_path / "fake_env"
    fake.write_text(f'#!/bin/sh\necho "$TMPDIR|$HOME|$XDG_CACHE_HOME" > {seen}\nexit 0\n')
    fake.chmod(0o755)
    src = tmp_path / "doc.rtf"
    src.write_text("x")
    run_libreoffice("pdf", str(src), str(tmp_path), binary=str(fake))
    tmpdir, home, cache = seen.read_text().strip().split("|")
    assert tmpdir == scratch.status()["dir"]
    assert home == cache and os.path.dirname(home) == tmpdir
    assert os.path.basename(home).startswith("shrew_lo_")


class _Upload:
    def __init__(self, data, name="doc.pdf"):
        self.filename = name
        self.file = io.BytesIO(data)


def test_upload_enospc_is_507(monkeypatch):
    from fastapi import HTTPException
    from app import server

    def full(fd, chunk):
        raise OSError(errno.ENOSPC, "No space left on device")
    d = scratch.status()["dir"]
    before = set(os.listdir(d))
    monkeypatch.setattr(server.os, "write", full)
    with pytest.raises(HTTPException) as ei:
        server._save_upload(_Upload(b"%PDF-1.4 x"))
    assert ei.value.status_code == 507
    assert set(os.listdir(d)) - before == set()   # partial upload removed


def test_upload_mkstemp_enospc_is_507(monkeypatch):
    from fastapi import HTTPException
    from app import server

    def full(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(server.tempfile, "mkstemp", full)
    with pytest.raises(HTTPException) as ei:
        server._save_upload(_Upload(b"x"))
    assert ei.value.status_code == 507


def test_body_parse_enospc_maps_to_507_other_400s_unchanged():
    from fastapi import HTTPException
    from starlette.requests import Request
    from app import server

    req = Request({"type": "http", "method": "POST", "path": "/v1/convert", "headers": []})
    full = HTTPException(status_code=400, detail="There was an error parsing the body")
    full.__cause__ = OSError(errno.ENOSPC, "No space left on device")
    r = asyncio.run(server._body_parse_full_handler(req, full))
    assert r.status_code == 507 and b"scratch_full" in r.body
    plain = HTTPException(status_code=400, detail="bad page range")
    r = asyncio.run(server._body_parse_full_handler(req, plain))
    assert r.status_code == 400 and b"bad page range" in r.body


def test_handler_is_registered():
    from starlette.exceptions import HTTPException as SHE
    from app import server
    assert server.app.exception_handlers.get(SHE) is server._body_parse_full_handler
