"""RunTrace: a content-free, crash-surviving per-run diagnostic trace (GitLab
#24). Records SERVER STATE and failure evidence only — page numbers, status
enums, phase names, resource metrics — never document content, never raw
exception strings, never the source filename.
"""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from app import telemetry


def _events(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_trace_writes_durable_jsonl_keyed_by_run_id(tmp_path):
    tr = telemetry.RunTrace("abc123", total_pages=10, dir=str(tmp_path),
                            sample_interval_s=0)
    tr.phase("rasterize_done")
    tr.page(1, "ok", 1200, bucket="b_768")
    tr.close("done")

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert "abc123" in files[0].name
    evs = _events(files[0])
    kinds = [e["ev"] for e in evs]
    assert kinds[0] == "open" and kinds[-1] == "done"
    assert {"phase", "page"} <= set(kinds)


def test_events_are_flushed_immediately_not_buffered(tmp_path):
    # A crash loses only the in-flight line, never the backlog: each event must
    # hit disk as it happens.
    tr = telemetry.RunTrace("run", dir=str(tmp_path), sample_interval_s=0)
    tr.phase("transcribe_start")
    path = next(tmp_path.glob("*.jsonl"))
    assert any(e["ev"] == "phase" for e in _events(path)), "not flushed"
    tr.close("done")


def test_page_event_rejects_document_content(tmp_path):
    """The API must make leaking page text structurally impossible: page()
    takes a NUMBER and status enum — there is no text parameter, and a
    non-int page or unknown status is refused."""
    tr = telemetry.RunTrace("r", dir=str(tmp_path), sample_interval_s=0)
    with pytest.raises((TypeError, ValueError)):
        tr.page("some transcribed sentence", "ok", 10)  # page must be int
    with pytest.raises((TypeError, ValueError)):
        tr.page(1, "The document says...", 10)  # status must be a known enum
    tr.close("done")


def test_no_event_field_contains_free_text(tmp_path):
    tr = telemetry.RunTrace("r", total_pages=3, dir=str(tmp_path),
                            sample_interval_s=0)
    tr.phase("assemble_start")
    tr.page(2, "degenerate", 50, bucket="b_1024", retry_tier="fallback",
            fallback=True)
    tr.sample()
    tr.close("done")
    path = next(tmp_path.glob("*.jsonl"))
    # Every string value in every event is a short controlled token (enum,
    # phase, key) — no field carries a sentence of document text.
    for e in _events(path):
        for v in e.values():
            if isinstance(v, str):
                assert len(v) <= 40 and "\n" not in v, f"suspicious string: {v!r}"


def test_died_records_category_and_class_not_message(tmp_path):
    tr = telemetry.RunTrace("r", dir=str(tmp_path), sample_interval_s=0)
    secret = "PARSE FAILED near 'Patient SSN 123-45-6789 diagnosis...'"
    try:
        raise ValueError(secret)
    except ValueError as e:
        tr.died("json_build", e)
    tr.close("failed")
    path = next(tmp_path.glob("*.jsonl"))
    died = [e for e in _events(path) if e["ev"] == "died_at"]
    assert len(died) == 1
    d = died[0]
    assert d["phase"] == "json_build"
    assert d["exc_type"] == "ValueError"
    assert d["category"] == "parse"          # mapped from the exception
    # the raw message (which carries document content) must NOT appear anywhere
    blob = json.dumps(_events(path))
    assert "SSN" not in blob and "123-45-6789" not in blob and secret not in blob


@pytest.mark.parametrize("exc,cat", [
    (MemoryError(), "oom"),
    (TimeoutError(), "timeout"),
    (ConnectionResetError(), "connection"),
    (json.JSONDecodeError("x", "doc", 0), "parse"),
    (OSError(28, "No space left on device"), "disk_full"),
    (RuntimeError(), "other"),
])
def test_error_categories(tmp_path, exc, cat):
    assert telemetry._category(exc) == cat


def test_resource_sample_is_numeric_server_state(tmp_path):
    tr = telemetry.RunTrace("r", total_pages=5, dir=str(tmp_path),
                            sample_interval_s=0)
    tr.mark_pages_done(3)
    tr.sample()
    tr.close("done")
    path = next(tmp_path.glob("*.jsonl"))
    s = [e for e in _events(path) if e["ev"] == "sample"][-1]
    for k in ("rss_mb", "open_fds", "threads", "disk_free_mb", "elapsed_s"):
        assert isinstance(s[k], (int, float)) and s[k] >= 0, k
    assert s["pages_done"] == 3 and s["total_pages"] == 5


def test_background_sampler_emits_without_page_calls(tmp_path):
    # A run that hangs mid-transcription (no page completes) must still leave a
    # heartbeat of resource samples — that is how an OOM/disk stall is caught.
    tr = telemetry.RunTrace("r", dir=str(tmp_path), sample_interval_s=0.05)
    tr.start_sampler()
    time.sleep(0.25)
    tr.close("done")
    path = next(tmp_path.glob("*.jsonl"))
    assert sum(1 for e in _events(path) if e["ev"] == "sample") >= 2


def test_live_state_is_readable_cross_process(tmp_path):
    tr = telemetry.RunTrace("liverun", total_pages=100, dir=str(tmp_path),
                            sample_interval_s=0)
    tr.phase("transcribe_start")
    tr.mark_pages_done(42)
    tr.publish_live()
    live = telemetry.read_live(str(tmp_path))
    one = [r for r in live if r["run_id"] == "liverun"][0]
    assert one["phase"] == "transcribe_start"
    assert one["pages_done"] == 42 and one["total_pages"] == 100
    assert isinstance(one["rss_mb"], (int, float))
    tr.close("done")
    # live entry is cleared when the run ends
    assert not [r for r in telemetry.read_live(str(tmp_path)) if r["run_id"] == "liverun"]


def test_trace_survives_hard_kill_of_the_process(tmp_path):
    """SIGKILL mid-run (the OOM-killer's signal): the backlog written before
    the kill must remain on disk and parseable — that is the whole point."""
    script = (
        "import time\n"
        "from app import telemetry\n"
        f"tr = telemetry.RunTrace('killed', total_pages=9, dir={str(tmp_path)!r}, sample_interval_s=0)\n"
        "tr.phase('transcribe_start')\n"
        "tr.page(1, 'ok', 10)\n"
        "tr.page(2, 'ok', 10)\n"
        "print('READY', flush=True)\n"
        "time.sleep(30)\n"
    )
    p = subprocess.Popen([sys.executable, "-c", script],
                         stdout=subprocess.PIPE, env={**os.environ})
    assert p.stdout.readline().strip() == b"READY"
    p.kill()
    p.wait(timeout=5)
    path = next(tmp_path.glob("*killed*.jsonl"))
    kinds = [e["ev"] for e in _events(path)]
    assert kinds.count("page") == 2 and "phase" in kinds
    # no clean "done" — a post-mortem sees the trace simply stops
    assert "done" not in kinds
