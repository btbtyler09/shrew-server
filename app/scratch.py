"""RAM-only scratch space for uploaded documents and everything derived from them.

Uploaded documents can be sensitive, so their bytes — the upload itself, the
multipart spool, rendered page images, LibreOffice output and profile, crops,
and the per-request output dir — must never reach persistent disk.

How:
- ``configure()`` (run when ``app.server`` is imported, before any request)
  points Python's ``tempfile`` module and the ``TMPDIR``/``TMP``/``TEMP``
  environment variables at ``SHREW_SCRATCH_DIR`` (default ``/dev/shm/shrew``).
  Every ``tempfile`` user in this process — our own code, Starlette's
  ``UploadFile`` spool, and third-party libraries — and every child process
  (LibreOffice, PDF tools) that honours ``TMPDIR`` therefore writes there.
- ``enforce()`` (run first thing at startup) checks with ``statfs`` that the
  scratch dir is on tmpfs or ramfs and refuses to start otherwise, unless
  ``SHREW_ALLOW_DISK_SCRATCH=1`` is set explicitly.
- ``status()`` is reported in ``/health``.
- ``is_full_error()`` lets request handlers turn a full scratch dir into a
  clean 507 for that request. There is no fallback to disk.

Swap: tmpfs pages can be swapped out. Run the container with swap disabled
(``memswap_limit`` equal to ``mem_limit``) or on a host without swap; see the
README "Data handling" section.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import logging
import os
import stat
import tempfile

logger = logging.getLogger("shrew.scratch")

DEFAULT_SCRATCH_DIR = "/dev/shm/shrew"

# statfs f_type magics (linux/magic.h)
_TMPFS_MAGIC = 0x01021994
_RAMFS_MAGIC = 0x858458F6
_IN_MEMORY_MAGICS = {_TMPFS_MAGIC: "tmpfs", _RAMFS_MAGIC: "ramfs"}
_IN_MEMORY_MOUNT_TYPES = {"tmpfs", "ramfs"}

# Failures that crossed a process boundary (the rasterizer child, LibreOffice)
# arrive as message text, not an OSError; match the libc strings for ENOSPC/EDQUOT.
_FULL_TEXT = ("[Errno 28]", "No space left on device", "[Errno 122]", "Disk quota exceeded")

_configured_dir: str | None = None
# The system temp dir as it was before configure() redirected tempfile.
# Content-free state that must outlive a request (telemetry traces, VLM-slot
# lease files) keeps living here, not in the document scratch dir.
_SYSTEM_TMP = tempfile.gettempdir()
_configure_error: str | None = None


def scratch_dir() -> str:
    return (os.environ.get("SHREW_SCRATCH_DIR") or "").strip() or DEFAULT_SCRATCH_DIR


def current_dir() -> str:
    return _configured_dir or os.path.abspath(scratch_dir())


def system_tmp() -> str:
    return _SYSTEM_TMP


def child_env(home: str | None = None) -> dict:
    """Environment for a child process (LibreOffice, PDF tools): temp files go
    to the scratch dir; with ``home``, HOME/XDG dirs point there too so the
    child keeps no per-user state on disk."""
    env = dict(os.environ)
    if _configured_dir:          # server process; the plain CLI keeps the caller's TMPDIR
        d = _configured_dir
        env.update(TMPDIR=d, TMP=d, TEMP=d)
    if home:
        env.update(HOME=home, XDG_CONFIG_HOME=home, XDG_CACHE_HOME=home, XDG_DATA_HOME=home)
    return env


def allow_disk() -> bool:
    return (os.environ.get("SHREW_ALLOW_DISK_SCRATCH") or "").strip().lower() in ("1", "true", "yes")


def configure() -> str:
    """Create the scratch dir (0700) and route all temp files into it.

    Idempotent. Never raises: a failure is recorded and ``enforce()`` refuses
    to start with it, so importing the module stays safe (tests, CLI help).
    """
    global _configured_dir, _configure_error
    d = os.path.abspath(scratch_dir())
    try:
        os.makedirs(d, mode=0o700, exist_ok=True)
        st = os.lstat(d)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise OSError(f"{d} is not a real directory")
        if st.st_uid != os.getuid():
            raise OSError(f"{d} is owned by uid {st.st_uid}, not this process (uid {os.getuid()})")
        if st.st_mode & 0o077:
            os.chmod(d, 0o700)
    except OSError as e:
        _configure_error = f"cannot use scratch dir {d}: {e}"
        logger.error(_configure_error)
        return d
    _configure_error = None
    tempfile.tempdir = d
    for var in ("TMPDIR", "TMP", "TEMP"):
        os.environ[var] = d
    _configured_dir = d
    return d


def _statfs_magic(path: str) -> int | None:
    """f_type of the filesystem holding ``path`` via libc statfs, or None.

    ``struct statfs`` starts with ``f_type`` as a C long on Linux for both
    glibc and musl (x86_64 and aarch64), which is all this needs.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
        buf = ctypes.create_string_buffer(256)  # larger than struct statfs on any 64-bit libc
        if libc.statfs(os.fsencode(path), buf) != 0:
            return None
        return ctypes.c_long.from_buffer(buf).value & 0xFFFFFFFF
    except (OSError, AttributeError, TypeError):
        return None


def _mount_fstype(path: str) -> str | None:
    """Filesystem type of the longest mount point containing ``path`` (fallback)."""
    try:
        real = os.path.realpath(path)
        best, best_type = "", None
        with open("/proc/self/mountinfo", encoding="utf-8") as f:
            for line in f:
                left, _, right = line.partition(" - ")
                fields = left.split()
                if len(fields) < 5 or not right:
                    continue
                mnt = fields[4].replace("\\040", " ")
                if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
                    best, best_type = mnt, right.split()[0]
        return best_type
    except OSError:
        return None


def fs_type(path: str) -> str:
    magic = _statfs_magic(path)
    if magic is not None:
        return _IN_MEMORY_MAGICS.get(magic, f"0x{magic:x}")
    return _mount_fstype(path) or "unknown"


def status() -> dict:
    """Content-free description of the scratch dir, for /health and logs."""
    d = _configured_dir or os.path.abspath(scratch_dir())
    out: dict = {"dir": d, "allow_disk": allow_disk()}
    if _configure_error:
        out.update(fs="unknown", in_memory=False, ok=False, error=_configure_error)
        return out
    fst = fs_type(d)
    in_mem = fst in _IN_MEMORY_MOUNT_TYPES
    out.update(fs=fst, in_memory=in_mem, ok=in_mem or out["allow_disk"])
    rd = raw_dump_status()
    if rd:
        out["raw_dump"] = rd
    try:
        sv = os.statvfs(d)
        out["total_bytes"] = sv.f_blocks * sv.f_frsize
        out["free_bytes"] = sv.f_bavail * sv.f_frsize
    except OSError:
        pass
    return out


def _nearest_existing(path: str) -> str:
    p = os.path.abspath(path)
    while not os.path.exists(p) and p != os.path.dirname(p):
        p = os.path.dirname(p)
    return p


def raw_dump_status() -> dict | None:
    """SHREW_RAW_DUMP_DIR (debug) appends raw model output — document
    content — to files there. Same in-memory rule as the scratch dir."""
    d = (os.environ.get("SHREW_RAW_DUMP_DIR") or "").strip()
    if not d:
        return None
    fst = fs_type(_nearest_existing(d))
    return {"dir": os.path.abspath(d), "fs": fst, "in_memory": fst in _IN_MEMORY_MOUNT_TYPES}


def enforce() -> dict:
    """Refuse to start unless the scratch dir is in memory (or disk is explicitly allowed)."""
    configure()
    rd = raw_dump_status()
    if rd and not rd["in_memory"]:
        msg = (f"SHREW_RAW_DUMP_DIR={rd['dir']} is on '{rd['fs']}', not tmpfs/ramfs; it receives "
               f"raw model output (document content).")
        if not allow_disk():
            raise RuntimeError(f"shrew-server refuses to start: {msg} Unset it, point it at a "
                               f"tmpfs, or set SHREW_ALLOW_DISK_SCRATCH=1.")
        logger.warning(f"SHREW_ALLOW_DISK_SCRATCH=1: {msg}")
    st = status()
    if st.get("error"):
        raise RuntimeError(f"shrew-server refuses to start: {st['error']}")
    if not st["in_memory"]:
        msg = (f"scratch dir {st['dir']} is on '{st['fs']}', not tmpfs/ramfs. Uploaded documents "
               f"would be written to disk. Mount a tmpfs there (see README 'Data handling') or set "
               f"SHREW_SCRATCH_DIR to an in-memory path.")
        if not st["allow_disk"]:
            raise RuntimeError(f"shrew-server refuses to start: {msg} "
                               f"Set SHREW_ALLOW_DISK_SCRATCH=1 to accept disk scratch anyway.")
        logger.warning(f"SHREW_ALLOW_DISK_SCRATCH=1: {msg}")
    else:
        logger.info(f"scratch: {st['dir']} on {st['fs']} "
                    f"({st.get('free_bytes', 0) / 2**20:.0f} MiB free of {st.get('total_bytes', 0) / 2**20:.0f} MiB)")
    return st


def is_full_error(exc: BaseException | None) -> bool:
    """True if ``exc`` or anything in its cause/context chain is a full-disk
    error: an OSError with ENOSPC/EDQUOT, or a message carrying that error's
    text (failures relayed from a child process lose the errno)."""
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        msg = str(e)
        if any(t in msg for t in _FULL_TEXT):
            return True
        e = e.__cause__ or e.__context__
    return False


def full_body() -> dict:
    """The one 507 body every scratch-full path returns."""
    return {"error": FULL_DETAIL, "detail": FULL_DETAIL, "reason": "scratch_full"}


FULL_DETAIL = ("scratch space full: uploaded documents are held in memory only and the "
               "in-memory scratch dir has no room for this request; retry later or raise the "
               "scratch tmpfs size")
