"""Content-free, crash-surviving per-run diagnostics (GitLab #24).

A 3,500-page conversion that dies mid-transcription used to leave NO evidence:
every timing and gate stat lived in ``processing_log``, which is only returned
in the HTTP response — so a job that never returns destroyed its own post-mortem.

``RunTrace`` fixes that by appending events to a durable JSONL file *as they
happen* (each line flushed + fsync-friendly), plus a background sampler that
records server resource state on a heartbeat even when no page is completing.
A hard kill (the OOM-killer's SIGKILL) leaves the backlog intact; a caught
exception writes a ``died_at`` marker naming the phase. Reading the last lines
tells you WHERE and in WHAT server state the run died.

Privacy invariant — this trace records SERVER STATE and FAILURE EVIDENCE only,
never document content:

- ``page()`` takes a page NUMBER and a status ENUM. There is no text
  parameter, so transcribed text cannot enter by construction.
- Errors are recorded as a fixed CATEGORY plus the exception CLASS NAME.
  ``str(exc)`` is never stored — a JSON-parse error echoes model output, and
  that output is document content.
- Runs are keyed by the caller's ``run_id`` (the doc_id hash), never the
  source filename.

Directory: SHREW_TELEMETRY_DIR, else <tmpdir>/shrew-telemetry. It must be
DURABLE — never the per-request temp dir, which is deleted on cleanup and
would take the evidence with it. SHREW_TELEMETRY=0 disables the layer (a
NullTrace stands in, every method a no-op).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import tempfile
import threading
import time

logger = logging.getLogger("shrew.telemetry")

# Status values a page may report — the ONLY strings page() accepts. Mirrors
# the gate outcomes in structured_page/_gate so the trace can never carry a
# free-form (content-bearing) status.
PAGE_STATUSES = frozenset({
    "ok", "schema", "degenerate", "overlong_failed", "empty_completion",
    "empty", "parse", "failed", "transport_error", "oversize", "cancelled",
    "image_fallback",
})

# Phase markers the pipeline may emit. A closed vocabulary keeps phase names
# from ever becoming a channel for content.
PHASES = frozenset({
    "upload", "rasterize_start", "rasterize_done", "transcribe_start",
    "transcribe_done", "assemble_start", "stitch", "fidelity", "json_build",
    "serialize_start", "serialize_done", "render_raw",
})

_LIVE_SUFFIX = ".live.json"
_OPEN_FLAGS = os.O_NOFOLLOW | os.O_CLOEXEC


def _dir(override: str | None = None) -> str:
    return override or os.environ.get(
        "SHREW_TELEMETRY_DIR",
        os.path.join(tempfile.gettempdir(), "shrew-telemetry"))


def _category(exc: BaseException) -> str:
    """Map an exception to a fixed, content-free category. Never inspects the
    message text (which can carry document content)."""
    if isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, (TimeoutError,)):
        return "timeout"
    # socket.timeout is an OSError subclass on some versions; check name too.
    if type(exc).__name__ in ("timeout", "ReadTimeout", "ConnectTimeout",
                              "ReadTimeoutError"):
        return "timeout"
    if isinstance(exc, (ConnectionError, ConnectionResetError, BrokenPipeError)):
        return "connection"
    if type(exc).__name__ in ("ConnectionError", "ChunkedEncodingError",
                              "ProtocolError", "IncompleteRead"):
        return "connection"
    # JSON / parse errors — the message echoes model output, so category only.
    if type(exc).__name__ in ("JSONDecodeError",) or isinstance(exc, ValueError) \
            and type(exc).__name__ == "JSONDecodeError":
        return "parse"
    try:
        import json as _json
        if isinstance(exc, _json.JSONDecodeError):
            return "parse"
    except Exception:  # noqa: BLE001
        pass
    if isinstance(exc, ValueError):
        return "parse"
    if isinstance(exc, OSError):
        if getattr(exc, "errno", None) in (errno.ENOSPC,):
            return "disk_full"
        if getattr(exc, "errno", None) in (errno.EMFILE, errno.ENFILE):
            return "fd_exhausted"
        return "os"
    if type(exc).__name__ == "CancelledException":
        return "cancelled"
    return "other"


def _rss_mb() -> float:
    """Current resident set size in MiB. /proc on Linux; getrusage fallback."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)  # kB -> MiB
    except OSError:
        pass
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 1024, 1)  # Linux ru_maxrss is kB
    except Exception:  # noqa: BLE001
        return 0.0


def _open_fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0


def _disk_free_mb(path: str) -> float:
    try:
        import shutil
        return round(shutil.disk_usage(path).free / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        return 0.0


class RunTrace:
    """One conversion's diagnostic trace. Thread-safe; every write flushed."""

    def __init__(self, run_id: str, total_pages: int | None = None,
                 dir: str | None = None, sample_interval_s: float = 5.0):
        self.run_id = str(run_id)
        self.total_pages = total_pages
        self._pages_done = 0
        self._phase = "upload"
        self._interval = float(sample_interval_s)
        self._start = time.time()
        self._lock = threading.Lock()
        self._closed = False
        self._sampler: threading.Thread | None = None
        self._stop = threading.Event()

        self._dir = _dir(dir)
        os.makedirs(self._dir, mode=0o700, exist_ok=True)
        # run_id is a hex hash from the caller; keep it filename-safe anyway.
        safe = "".join(c for c in self.run_id if c.isalnum() or c in "-_")[:64] or "run"
        pid = os.getpid()
        self._path = os.path.join(self._dir, f"{int(self._start)}-{safe}-{pid}.jsonl")
        self._live_path = os.path.join(self._dir, f"{safe}-{pid}{_LIVE_SUFFIX}")
        self._fh = open(self._path, "a", encoding="utf-8")
        self._emit("open", total_pages=total_pages, pid=pid)

    # ── internals ────────────────────────────────────────────────────────────

    def _emit(self, ev: str, **fields) -> None:
        rec = {"t": round(time.time(), 3), "ev": ev,
               "elapsed_s": round(time.time() - self._start, 2)}
        rec.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            if self._closed:
                return
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError as e:
                logger.debug(f"trace write failed: {e}")

    # ── public API (content-free by construction) ────────────────────────────

    @property
    def current_phase(self) -> str:
        return self._phase

    def phase(self, name: str) -> None:
        if name not in PHASES:
            raise ValueError(f"unknown phase {name!r}")
        self._phase = name
        self._emit("phase", phase=name, pages_done=self._pages_done)
        self.publish_live()

    def page(self, page_no: int, status: str, ms: int,
             bucket: str | None = None, retry_tier: str | None = None,
             fallback: bool = False) -> None:
        if not isinstance(page_no, int) or isinstance(page_no, bool):
            raise TypeError("page_no must be an int (a position, not content)")
        if status not in PAGE_STATUSES:
            raise ValueError(f"unknown page status {status!r}")
        if retry_tier is not None and retry_tier not in ("coerced", "fallback"):
            raise ValueError(f"unknown retry_tier {retry_tier!r}")
        self._emit("page", page=page_no, status=status, ms=int(ms),
                   bucket=bucket, retry_tier=retry_tier,
                   fallback=bool(fallback) or None)

    def mark_pages_done(self, n: int) -> None:
        self._pages_done = int(n)

    def sample(self) -> None:
        self._emit("sample",
                   rss_mb=_rss_mb(), open_fds=_open_fds(),
                   threads=threading.active_count(),
                   disk_free_mb=_disk_free_mb(self._dir),
                   pages_done=self._pages_done, total_pages=self.total_pages)
        self.publish_live()

    def died(self, phase: str, exc: BaseException) -> None:
        # phase may be any of PHASES or a caller label; never content.
        self._emit("died_at", phase=str(phase)[:40],
                   exc_type=type(exc).__name__[:40], category=_category(exc),
                   pages_done=self._pages_done, total_pages=self.total_pages,
                   rss_mb=_rss_mb(), disk_free_mb=_disk_free_mb(self._dir))

    def publish_live(self) -> None:
        """Write the compact in-flight state for /health to read cross-process.
        Numbers + the current phase enum only."""
        rec = {"run_id": self.run_id, "phase": self._phase,
               "pages_done": self._pages_done, "total_pages": self.total_pages,
               "rss_mb": _rss_mb(), "elapsed_s": round(time.time() - self._start, 1),
               "pid": os.getpid()}
        tmp = self._live_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rec, f, separators=(",", ":"))
            os.replace(tmp, self._live_path)  # atomic
        except OSError:
            pass

    def start_sampler(self) -> None:
        if self._interval <= 0 or self._sampler is not None:
            return

        def _loop():
            while not self._stop.wait(self._interval):
                self.sample()

        self._sampler = threading.Thread(target=_loop, name=f"trace-{self.run_id}",
                                         daemon=True)
        self._sampler.start()

    def close(self, status: str) -> None:
        with self._lock:
            if self._closed:
                return
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
        self._emit("done", status=str(status)[:40], pages_done=self._pages_done,
                   total_pages=self.total_pages, rss_mb=_rss_mb())
        with self._lock:
            self._closed = True
            try:
                self._fh.close()
            except OSError:
                pass
        try:
            os.unlink(self._live_path)  # clear in-flight state
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.died(self._phase, exc)
            self.close("failed")
        else:
            self.close("done")
        return False


class NullTrace:
    """No-op trace when telemetry is disabled — every method a silent no-op."""
    run_id = ""
    current_phase = ""

    def phase(self, *a, **k): pass
    def page(self, *a, **k): pass
    def mark_pages_done(self, *a, **k): pass
    def sample(self, *a, **k): pass
    def died(self, *a, **k): pass
    def publish_live(self, *a, **k): pass
    def start_sampler(self, *a, **k): pass
    def close(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


def new_trace(run_id: str, total_pages: int | None = None,
              sample_interval_s: float = 5.0):
    """A RunTrace, or a NullTrace when disabled/unavailable (never fatal —
    telemetry must not fail a conversion)."""
    if os.environ.get("SHREW_TELEMETRY", "1") == "0":
        return NullTrace()
    try:
        tr = RunTrace(run_id, total_pages=total_pages,
                      sample_interval_s=sample_interval_s)
        tr.start_sampler()
        return tr
    except Exception as e:  # noqa: BLE001
        logger.warning(f"telemetry unavailable: {e}")
        return NullTrace()


def read_live(dir: str | None = None) -> list[dict]:
    """Every in-flight run's compact state (for /health). Stale files whose
    process is gone are ignored (best-effort: we cannot always tell, so a
    consumer treats these as advisory)."""
    d = _dir(dir)
    out: list[dict] = []
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for name in names:
        if not name.endswith(_LIVE_SUFFIX):
            continue
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        pid = rec.get("pid")
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)  # process alive?
            except ProcessLookupError:
                continue  # crashed worker — skip its stale live file
            except OSError:
                pass
        out.append(rec)
    return out
