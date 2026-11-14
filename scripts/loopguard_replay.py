#!/usr/bin/env python3
"""Offline replay of shrew-server's streaming loop guard over saved model outputs.

    python3 scripts/loopguard_replay.py --rows corpus.jsonl [--labels labels.jsonl] [--out rows.jsonl]
    python3 scripts/loopguard_replay.py --clean clean.jsonl --loop loop.jsonl      (the 2026-11-09 form)

Rows: jsonl {"src", "id", "text", "finish"?} — raw completions (a SHREW_RAW_DUMP_DIR
dump: use its "text" and "finish_reason") or page JSON rebuilt from a response.
Labels: jsonl {"src", "id", "label"}, label = "loop" | "clean" | anything else;
labels come from reading the text, never from how the generation finished (a
page that ends with "stop" can be a loop that stopped by itself).

Every text is streamed through the server's own RepetitionGuard in 4-char
deltas, twice: with the first v0.5 development guard ("old": 2k zlib window,
8k zlib window > 12, 60k cap) and with the current v0.5 parameters ("new"). A
text that a guard cut while it was being generated (finish repetition_abort /
guard) is also judged once at its last character, because the serving check
grid is not the replay's. Reports the old x new table per label, every
disagreement, and the highest scores that did not fire.

This is the gate guard changes are accepted with: re-run it on new dumps before
touching a threshold.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app import structured_page as sp  # noqa: E402

OLD = {"long_window": 8000, "long_threshold": 12.0, "length_cap_chars": 60000}   # the first v0.5 development guard
CUT = ("repetition_abort", "guard")


def replay(text: str, params: dict, cut: bool = False):
    """(guard object, abort position or None) for one text."""
    g = sp.RepetitionGuard(**params)
    for i in range(0, len(text), 4):
        if g(text[i:i + 4], text[:i + 4]):
            return g, min(i + 4, len(text))
    if cut and len(text) >= g.window:
        g._next_check = 0
        if g("", text):
            return g, len(text)
    return g, None


def _one(row: dict) -> dict:
    text = row["text"]
    cut = row.get("finish") in CUT
    g_old, at_old = replay(text, OLD, cut)
    g_new, at_new = replay(text, sp.v05_guard_params(), cut)
    ev = dict(g_new.evidence or {})
    ev.pop("tail", None)
    return {"src": row.get("src"), "id": row.get("id"), "finish": row.get("finish"), "chars": len(text),
            "old": [at_old, g_old.abort_guard], "new": [at_new, g_new.abort_guard],
            "old_max": {"window": round(g_old.max_ratio, 2), "long_window": round(g_old.long_max_ratio, 2)},
            "new_max": {"periodic": g_new.periodic_max, "counter": g_new.counter_max,
                        "repeated": g_new.repeat_max, "numeric": g_new.numeric_max},
            "evidence": ev or None}


def _work(line: str) -> dict:
    logging.disable(logging.WARNING)            # the guard logs every abort
    return _one(json.loads(line))


def _lines(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows")
    ap.add_argument("--labels")
    ap.add_argument("--clean")
    ap.add_argument("--loop")
    ap.add_argument("--out", help="write one result row per text (jsonl)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    labels = {}
    sources = []
    if a.rows:
        sources.append((a.rows, None))
    for path, lab in ((a.clean, "clean"), (a.loop, "loop")):
        if path:
            sources.append((path, lab))
    if not sources:
        ap.error("give --rows, or --clean and --loop")
    if a.labels:
        for line in _lines(a.labels):
            r = json.loads(line)
            labels[(r.get("src"), r["id"])] = r["label"]
    results = []
    out = open(a.out, "w") if a.out else None
    with mp.Pool(a.jobs) as pool:
        for path, lab in sources:
            for r in pool.imap(_work, _lines(path), chunksize=40):
                r["label"] = lab or labels.get((r["src"], r["id"]), "unlabelled")
                results.append(r)
                if out:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
    if out:
        out.close()
    table = Counter((r["label"], r["old"][1] or "-", r["new"][1] or "-") for r in results)
    by_label = {}
    for lab in sorted({r["label"] for r in results}):
        rows = [r for r in results if r["label"] == lab]
        by_label[lab] = {"n": len(rows),
                         "old_aborts": sum(1 for r in rows if r["old"][1]),
                         "new_aborts": sum(1 for r in rows if r["new"][1]),
                         "old_only": [[r["src"], r["id"], *r["old"]] for r in rows if r["old"][1] and not r["new"][1]],
                         "new_only": [[r["src"], r["id"], *r["new"]] for r in rows if r["new"][1] and not r["old"][1]]}
    quiet = [r for r in results if not r["new"][1]]
    print(json.dumps({
        "params": {"window": sp.STREAM_GUARD_WINDOW_CHARS, "ratio": sp.STREAM_GUARD_RATIO,
                   "consecutive": sp.STREAM_GUARD_CONSECUTIVE, "check_every": sp.STREAM_GUARD_CHECK_EVERY_CHARS,
                   "old": OLD, "new": sp.v05_guard_params()},
        "texts": len(results),
        "old_x_new": {f"{lab} | old {o} | new {n}": c for (lab, o, n), c in sorted(table.items())},
        "by_label": by_label,
        "highest_not_fired": {k: sorted(((r["new_max"][k], r["label"], r["src"], r["id"]) for r in quiet),
                                        key=lambda x: -x[0])[:5]
                              for k in ("periodic", "counter", "repeated", "numeric")},
    }, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
