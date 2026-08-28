"""Cross-process conversion-activity tracking for /health (v0.3.7).

Uvicorn workers are separate processes, so in-process counters cannot answer
"is this server saturated?" — the lease DIRECTORY is the shared source of
truth. One lease file per conversion:

- ``queued-<uuid>.lease`` is created (exclusively, locked) when a conversion
  is admitted and starts waiting for the worker's pipeline gate;
- it is atomically renamed to ``running-<uuid>.lease`` when the gate is
  acquired (the flock rides the inode, so the lock survives the rename);
- it is unlinked and closed on completion, failure, cancellation, or
  disconnect.

Crash safety is the flock itself: the kernel drops a process's locks when it
dies, so an UNLOCKED lease file is by definition a crashed worker's footprint.
``snapshot()`` prunes those instead of counting them, and ``prune_stale()``
runs at startup — no stale count can survive a crash or restart, while live
workers' leases (locks held) are never touched.

Security: directory 0700 and files 0600, owned by the service user; opens use
O_NOFOLLOW | O_CLOEXEC (and O_EXCL on create); symlinks, foreign-owned and
irregular files are ignored. Lease files are empty — nothing secret to leak.

The directory defaults to <tmpdir>/shrew-concurrency; override with
SHREW_CONCURRENCY_DIR (multiple shrew-server instances on one box should get
separate directories, or their counts merge).
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import stat
import tempfile
import time
import uuid

logger = logging.getLogger("shrew.concurrency")

_SUFFIX = ".lease"
_OPEN_FLAGS = os.O_NOFOLLOW | os.O_CLOEXEC


def _dir() -> str:
    return os.environ.get(
        "SHREW_CONCURRENCY_DIR",
        os.path.join(tempfile.gettempdir(), "shrew-concurrency"))


def ensure_dir() -> str:
    """Create/validate the lease directory: 0700, our uid, a real directory."""
    d = _dir()
    os.makedirs(d, mode=0o700, exist_ok=True)
    st = os.lstat(d)
    if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
        raise RuntimeError(f"lease path {d} is not a directory")
    if st.st_uid != os.geteuid():
        raise RuntimeError(f"lease dir {d} is not owned by the service user")
    if stat.S_IMODE(st.st_mode) != 0o700:
        os.chmod(d, 0o700)
    return d


class Lease:
    """One conversion's liveness token. Create = queued; mark_running() after
    the pipeline gate is acquired; release() always (finally). The exclusive
    flock is held for the whole lease lifetime."""

    def __init__(self):
        d = ensure_dir()
        name = uuid.uuid4().hex
        self._path = os.path.join(d, f"queued-{name}{_SUFFIX}")
        self._name = name
        self._fd = os.open(
            self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _OPEN_FLAGS, 0o600)
        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._released = False

    def mark_running(self) -> None:
        if self._released:
            return
        new_path = os.path.join(_dir(), f"running-{self._name}{_SUFFIX}")
        os.rename(self._path, new_path)  # atomic; the flock rides the inode
        self._path = new_path

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        except OSError as e:  # never let cleanup break a conversion path
            logger.warning(f"lease unlink failed: {e}")
        try:
            os.close(self._fd)  # drops the flock
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class NullLease:
    """No-op stand-in when lease creation fails (disk full, bad perms):
    activity tracking is advisory — it must never fail a conversion."""

    def mark_running(self) -> None:
        pass

    def release(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def new_lease():
    """A Lease, or a NullLease if the lease dir is unusable (logged)."""
    try:
        return Lease()
    except Exception as e:  # noqa: BLE001 — advisory layer
        logger.warning(f"conversion lease unavailable: {e}")
        return NullLease()


def snapshot(prune: bool = True) -> dict:
    """Aggregate counts across ALL workers: {"running": n, "queued": m}.

    A lease whose flock can be taken has no living holder — pruned (when
    prune=True), never counted. Symlinks, foreign-owned and irregular files
    are ignored entirely.
    """
    counts = {"running": 0, "queued": 0}
    d = _dir()
    try:
        entries = os.listdir(d)
    except FileNotFoundError:
        return counts
    for name in entries:
        if name.startswith("queued-") and name.endswith(_SUFFIX):
            state = "queued"
        elif name.startswith("running-") and name.endswith(_SUFFIX):
            state = "running"
        else:
            continue
        path = os.path.join(d, name)
        try:
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
                continue
            fd = os.open(path, os.O_RDONLY | _OPEN_FLAGS)
        except OSError:
            continue  # vanished (released/renamed mid-scan) or unopenable
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EACCES):
                    counts[state] += 1  # locked -> holder alive
                continue
            # Lock acquired: the holder is dead. Prune the footprint.
            if prune:
                try:
                    os.unlink(path)
                    logger.info(f"pruned stale lease {name} (crashed worker)")
                except FileNotFoundError:
                    pass
        finally:
            os.close(fd)
    return counts


def prune_stale() -> None:
    """Startup sweep: drop crashed workers' leases, keep live ones."""
    snapshot(prune=True)


# ── cross-process VLM slot gate ─────────────────────────────────────────────
# VLM_CONCURRENCY must bound in-flight VLM calls across ALL uvicorn workers —
# the model server has a fixed number of serving slots, and N spawned workers
# each holding their own N-sized semaphore multiply the real load by N (the
# audit-#14 bug). The gate is a row of slot files: holding slot i = holding
# an exclusive flock on slot-i.lock. Crash-safe like the leases (the kernel
# drops a dead process's flocks), no shared parent process required, and the
# same 0700/0600 + O_NOFOLLOW|O_CLOEXEC discipline.


def _slots_dir() -> str:
    d = os.path.join(ensure_dir(), "vlm-slots")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


class VlmSlot:
    """A held VLM serving slot. release() in a finally, always."""

    def __init__(self, fd: int):
        self._fd = fd
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            os.close(self._fd)  # drops the flock; the slot file persists
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def _try_slot(d: str, i: int):
    path = os.path.join(d, f"slot-{i:03d}.lock")
    try:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | _OPEN_FLAGS, 0o600)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return VlmSlot(fd)


def acquire_vlm_slot(limit: int, block: bool = True,
                     poll_s: float = 0.025) -> VlmSlot | None:
    """Take one of `limit` machine-wide VLM slots.

    limit <= 0 disables the gate (returns None — caller proceeds ungated).
    block=True waits until a slot frees (a VLM call parked here is exactly a
    call that would otherwise be queueing inside the saturated model server);
    block=False returns None when all slots are held.
    """
    if limit <= 0:
        return None
    d = _slots_dir()
    waiting_logged = False
    while True:
        for i in range(limit):
            slot = _try_slot(d, i)
            if slot is not None:
                return slot
        if not block:
            return None
        if not waiting_logged:
            logger.debug("VLM gate full (cross-process), waiting for a slot...")
            waiting_logged = True
        time.sleep(poll_s)


def vlm_in_flight() -> int:
    """How many VLM slots are currently held, machine-wide. Counts every
    slot file ever created (a stale higher-limit file is only counted while
    some process actually holds its lock)."""
    try:
        d = _slots_dir()
        names = os.listdir(d)
    except OSError:
        return 0
    held = 0
    for name in names:
        if not (name.startswith("slot-") and name.endswith(".lock")):
            continue
        path = os.path.join(d, name)
        try:
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
                continue
            fd = os.open(path, os.O_RDONLY | _OPEN_FLAGS)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EACCES):
                    held += 1
        finally:
            os.close(fd)
    return held
