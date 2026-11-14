"""v0.5 document passes that use the trained tool commands.

figure_check  ``classify_figure`` on each cropped figure: a crop the model calls
              ``artifact`` (stamp, logo, barcode, seal, decoration, scanner
              noise) is removed from the document's figures. Its record stays
              in processing_log.figure_check.
heading_fix   ``fix_headings`` over the document's markdown heading outline
              (every ``#`` line inside chunk content, in reading order): the
              model re-levels headings that were transcribed one page at a
              time; text is never changed.

Both are best-effort: a transport error, an unparsed answer, or an answer that
would change heading text leaves the document exactly as it was, and is
counted in the report. Neither pass can fail a document.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from . import contract as ct
from .preprocess import prepare_image_bucketed
from .generation import get_generation_params
from .structured_page import DecodePolicy, _sampling, build_command_messages, page_input_key

logger = logging.getLogger("shrew.v05_tools")

CLASSIFY_MAX_TOKENS = 8
CLASSIFY_TIMEOUT_S = float(os.environ.get("SHREW_FIGURE_CHECK_TIMEOUT", "120"))
# Training envelope for fix_headings rows (4,926 rows): median 5 lines,
# p99 15, max 26; max 1,748 chars. A longer outline is sent in windows inside
# that envelope.
HEADING_WINDOW_LINES = int(os.environ.get("SHREW_HEADING_WINDOW_LINES", "24"))
HEADING_WINDOW_CHARS = int(os.environ.get("SHREW_HEADING_WINDOW_CHARS", "1700"))
HEADING_TIMEOUT_S = float(os.environ.get("SHREW_HEADING_FIX_TIMEOUT", "120"))

_HEADING_LINE = re.compile(r"^#{1,6} \S")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _ask(client, messages, *, max_tokens, timeout, decode: "DecodePolicy | None") -> str:
    """One tool call with the server's single decode policy: the first-pass
    recipe (v0.4.0: temperature 0.4, presence penalty 0.3, fixed seed derived
    from the input). There is no greedy default anywhere."""
    decode = decode or DecodePolicy.from_env("v0.5")
    params = get_generation_params(client.model, decode.stage)
    temp, extra = _sampling(params["temperature"], params["extra_params"], decode.try1_temp,
                            decode, page_input_key(messages), 0)
    result = client.chat_completion(messages, max_tokens=max_tokens, temperature=temp,
                                    extra_params=extra, timeout=timeout)
    return result["choices"][0]["message"].get("content") or ""


# --------------------------------------------------------------------------- figure_check

def _checkable(f: dict) -> bool:
    fid = f.get("figure_id", "")
    # Page-failure renders and embedded spreadsheet media are not model figures.
    return bool(f.get("crop_path")) and f.get("bbox") is not None and not (
        fid.endswith("_pagefail") or "_embedded_" in fid)


def _classify_one(f: dict, client, decode=None) -> tuple[str | None, str | None]:
    """(verdict, error) for one figure crop."""
    try:
        with Image.open(f["crop_path"]) as im:
            enh, _bucket = prepare_image_bucketed(im.convert("RGB"))
        model_png = f["crop_path"][:-4] + "_check_model.png"
        enh.save(model_png)
        msgs = build_command_messages(ct.CMD_CLASSIFY_FIGURE, image_path=model_png)
        out = _ask(client, msgs, timeout=CLASSIFY_TIMEOUT_S, decode=decode,
                   max_tokens=ct.v05_max_tokens(ct.CMD_CLASSIFY_FIGURE, size=enh.size,
                                                cap=CLASSIFY_MAX_TOKENS))
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    v = ct.parse_figure_verdict(out)
    return (v, None) if v else (None, f"unparsed answer {out[:40]!r}")


def figure_check(doc: dict, client, *, concurrency: int = 4, decode=None) -> dict:
    """Drop figures the model classifies as artifacts. Mutates doc["figures"]."""
    todo = [f for f in doc["figures"] if _checkable(f)]
    report = {"checked": len(todo), "figures": 0, "artifacts": 0, "errors": 0, "dropped": []}
    if not todo:
        return report
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(lambda f: _classify_one(f, client, decode), todo))
    drop = set()
    for f, (verdict, err) in zip(todo, results):
        if err:
            report["errors"] += 1
            logger.warning("figure_check %s: %s (kept)", f["figure_id"], err)
        elif verdict == "artifact":
            report["artifacts"] += 1
            drop.add(f["figure_id"])
            report["dropped"].append({"figure_id": f["figure_id"], "page": f.get("page"),
                                      "caption": f.get("caption"), "bbox": f.get("bbox")})
        else:
            report["figures"] += 1
    if drop:
        doc["figures"] = [f for f in doc["figures"] if f["figure_id"] not in drop]
    return report


# --------------------------------------------------------------------------- heading_fix

def heading_lines(doc: dict) -> list[tuple[int, int, str]]:
    """(chunk index, line index, line) for every markdown heading line in chunk
    content, in document order; fenced code blocks are skipped."""
    out = []
    for ci, c in enumerate(doc["chunks"]):
        fenced = False
        for li, line in enumerate((c.get("content") or "").split("\n")):
            if _FENCE.match(line):
                fenced = not fenced
                continue
            if not fenced and _HEADING_LINE.match(line):
                out.append((ci, li, line.rstrip()))
    return out


def _windows(lines: list[str]) -> list[tuple[int, int]]:
    spans, start, chars = [], 0, 0
    for i, ln in enumerate(lines):
        if i > start and (i - start >= HEADING_WINDOW_LINES
                          or chars + len(ln) + 1 > HEADING_WINDOW_CHARS):
            spans.append((start, i))
            start, chars = i, 0
        chars += len(ln) + 1
    spans.append((start, len(lines)))
    return spans


def heading_fix(doc: dict, client, *, decode=None) -> dict:
    """Re-level the document's heading outline. Mutates chunk content in place."""
    found = heading_lines(doc)
    report = {"headings": len(found), "calls": 0, "changed": 0, "rejected": 0, "errors": 0}
    if len(found) < 2:                 # a single heading has no outline to fix
        return report
    lines = [ln for _, _, ln in found]
    fixed = list(lines)
    for a, b in _windows(lines):
        sent = lines[a:b]
        if len(sent) < 2:
            continue
        text = "\n".join(sent)
        report["calls"] += 1
        try:
            msgs = build_command_messages(ct.CMD_FIX_HEADINGS, text=text)
            out = _ask(client, msgs, timeout=HEADING_TIMEOUT_S, decode=decode,
                       max_tokens=ct.v05_max_tokens(ct.CMD_FIX_HEADINGS, text=text, cap=4096))
        except Exception as e:
            report["errors"] += 1
            logger.warning("heading_fix window %d-%d: %s: %s", a, b, type(e).__name__, e)
            continue
        got = ct.parse_fixed_headings(sent, out)
        if got is None:
            report["rejected"] += 1
            logger.warning("heading_fix window %d-%d rejected (text or count changed)", a, b)
            continue
        fixed[a:b] = got
    by_chunk: dict[int, list[str]] = {}
    for (ci, li, old), new in zip(found, fixed):
        if new == old:
            continue
        body = by_chunk.setdefault(ci, (doc["chunks"][ci].get("content") or "").split("\n"))
        body[li] = new
        report["changed"] += 1
    for ci, body in by_chunk.items():
        doc["chunks"][ci]["content"] = "\n".join(body)
    return report
