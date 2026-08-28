"""Cross-process conversion-activity tracking (v0.3.7, /health concurrency).

Crash-safe flock lease directory: one locked lease file per conversion,
queued-<uuid>.lease atomically renamed to running-<uuid>.lease, exclusive
flock held for the lease lifetime. An unlocked lease belongs to a crashed
worker and is pruned. Counts aggregate across uvicorn worker processes
because the directory — not process memory — is the source of truth.
"""
import os
import stat
import subprocess
import sys
import textwrap

import pytest

from app import concurrency as cc


@pytest.fixture(autouse=True)
def lease_dir(tmp_path, monkeypatch):
    d = tmp_path / "leases"
    monkeypatch.setenv("SHREW_CONCURRENCY_DIR", str(d))
    return d


# ── lease lifecycle ─────────────────────────────────────────────────────────


def test_lease_lifecycle_queued_running_release(lease_dir):
    lease = cc.Lease()
    files = os.listdir(lease_dir)
    assert len(files) == 1 and files[0].startswith("queued-")
    assert cc.snapshot() == {"running": 0, "queued": 1}

    lease.mark_running()
    files = os.listdir(lease_dir)
    assert len(files) == 1 and files[0].startswith("running-")
    assert cc.snapshot() == {"running": 1, "queued": 0}

    lease.release()
    assert os.listdir(lease_dir) == []
    assert cc.snapshot() == {"running": 0, "queued": 0}


def test_release_is_idempotent(lease_dir):
    lease = cc.Lease()
    lease.mark_running()
    lease.release()
    lease.release()  # second release must be a no-op, not an error
    assert cc.snapshot() == {"running": 0, "queued": 0}


def test_release_from_queued_state(lease_dir):
    """A conversion cancelled while still waiting for capacity releases its
    queued lease without ever running."""
    lease = cc.Lease()
    lease.release()
    assert os.listdir(lease_dir) == []


def test_multiple_leases_aggregate(lease_dir):
    l1, l2, l3 = cc.Lease(), cc.Lease(), cc.Lease()
    l1.mark_running()
    l2.mark_running()
    assert cc.snapshot() == {"running": 2, "queued": 1}
    for lease in (l1, l2, l3):
        lease.release()
    assert cc.snapshot() == {"running": 0, "queued": 0}


# ── crash safety ────────────────────────────────────────────────────────────


_ORPHAN_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {src!r})
    from app import concurrency as cc
    lease = cc.Lease()
    lease.mark_running()
    # exit WITHOUT release: the kernel drops the flock with the process,
    # leaving an unlocked lease file — a crashed worker's footprint.
""")


def _spawn_orphan(lease_dir):
    src = os.path.join(os.path.dirname(cc.__file__), "..")
    subprocess.run(
        [sys.executable, "-c", _ORPHAN_SCRIPT.format(src=os.path.abspath(src))],
        env={**os.environ, "SHREW_CONCURRENCY_DIR": str(lease_dir)},
        check=True, timeout=30,
    )


def test_crashed_worker_lease_is_pruned(lease_dir):
    _spawn_orphan(lease_dir)
    files = os.listdir(lease_dir)
    assert len(files) == 1 and files[0].startswith("running-")
    # The snapshot detects the lock is free -> dead holder -> pruned, not counted.
    assert cc.snapshot() == {"running": 0, "queued": 0}
    assert os.listdir(lease_dir) == []


def test_live_lease_survives_prune(lease_dir):
    """prune must NOT touch leases whose holders are alive (their flock is
    held) — a restarting worker sweeps only the dead."""
    _spawn_orphan(lease_dir)
    live = cc.Lease()
    live.mark_running()
    cc.prune_stale()
    assert cc.snapshot() == {"running": 1, "queued": 0}
    live.release()


def test_startup_prune_clears_stale_counts(lease_dir):
    """'No stale count remains after a worker crash or server restart.'"""
    for _ in range(3):
        _spawn_orphan(lease_dir)
    cc.prune_stale()
    assert os.listdir(lease_dir) == []
    assert cc.snapshot() == {"running": 0, "queued": 0}


# ── security ────────────────────────────────────────────────────────────────


def test_lease_dir_mode_and_ownership(lease_dir):
    cc.ensure_dir()
    st = os.lstat(lease_dir)
    assert stat.S_ISDIR(st.st_mode)
    assert stat.S_IMODE(st.st_mode) == 0o700
    assert st.st_uid == os.geteuid()


def test_lease_file_mode(lease_dir):
    lease = cc.Lease()
    path = os.path.join(lease_dir, os.listdir(lease_dir)[0])
    st = os.lstat(path)
    assert stat.S_ISREG(st.st_mode)
    assert stat.S_IMODE(st.st_mode) == 0o600
    assert st.st_uid == os.geteuid()
    lease.release()


def test_snapshot_ignores_symlinks_and_foreign_files(lease_dir):
    cc.ensure_dir()
    (lease_dir / "not-a-lease.txt").write_text("x")
    os.symlink("/etc/hostname", lease_dir / "running-fake.lease")
    assert cc.snapshot() == {"running": 0, "queued": 0}


def test_snapshot_with_missing_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SHREW_CONCURRENCY_DIR", str(tmp_path / "never-created"))
    assert cc.snapshot() == {"running": 0, "queued": 0}


def test_new_lease_falls_back_to_null_on_unusable_dir(tmp_path, monkeypatch):
    """Activity tracking is advisory: an unusable lease dir must yield a no-op
    lease, never an exception into the conversion path."""
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the lease DIR should be")
    monkeypatch.setenv("SHREW_CONCURRENCY_DIR", str(blocker))
    lease = cc.new_lease()
    assert isinstance(lease, cc.NullLease)
    lease.mark_running()  # all no-ops, no error
    lease.release()


# ── cross-process VLM slot gate ─────────────────────────────────────────────


def test_vlm_slot_acquire_release(lease_dir):
    s = cc.acquire_vlm_slot(2)
    assert s is not None
    assert cc.vlm_in_flight() == 1
    s2 = cc.acquire_vlm_slot(2)
    assert cc.vlm_in_flight() == 2
    # Limit reached: a non-blocking attempt reports no slot.
    assert cc.acquire_vlm_slot(2, block=False) is None
    s.release()
    assert cc.vlm_in_flight() == 1
    s3 = cc.acquire_vlm_slot(2, block=False)
    assert s3 is not None
    s2.release(); s3.release()
    assert cc.vlm_in_flight() == 0


def test_vlm_slot_release_is_idempotent(lease_dir):
    s = cc.acquire_vlm_slot(1)
    s.release()
    s.release()
    assert cc.vlm_in_flight() == 0


_SLOT_HOLDER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {src!r})
    from app import concurrency as cc
    s = cc.acquire_vlm_slot(1)
    print("held", flush=True)
    time.sleep({hold})
""")


def test_vlm_slot_is_cross_process(lease_dir):
    """A slot held by ANOTHER process consumes the shared limit — the whole
    point: N uvicorn workers cannot exceed VLM_CONCURRENCY combined."""
    import subprocess as sp
    src = os.path.abspath(os.path.join(os.path.dirname(cc.__file__), ".."))
    proc = sp.Popen(
        [sys.executable, "-c", _SLOT_HOLDER.format(src=src, hold=15)],
        env={**os.environ, "SHREW_CONCURRENCY_DIR": str(lease_dir)},
        stdout=sp.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        assert cc.vlm_in_flight() == 1
        assert cc.acquire_vlm_slot(1, block=False) is None, \
            "limit-1 slot held by another process must block us"
    finally:
        proc.kill()
        proc.wait(timeout=10)
    # Kernel dropped the dead process's flock: slot immediately reusable.
    s = cc.acquire_vlm_slot(1, block=False)
    assert s is not None
    s.release()


def test_vlm_slot_zero_limit_disables_gate(lease_dir):
    assert cc.acquire_vlm_slot(0) is None
    assert cc.acquire_vlm_slot(0, block=False) is None


def test_vlm_slot_blocking_waits_for_free_slot(lease_dir):
    import threading as th
    s = cc.acquire_vlm_slot(1)
    got = {}

    def _waiter():
        got["slot"] = cc.acquire_vlm_slot(1)

    t = th.Thread(target=_waiter)
    t.start()
    t.join(timeout=0.3)
    assert t.is_alive(), "acquire must still be waiting while the slot is held"
    s.release()
    t.join(timeout=10)
    assert not t.is_alive() and got["slot"] is not None
    got["slot"].release()
