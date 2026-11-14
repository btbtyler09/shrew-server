"""Content-aware loop statistics for the v0.5 streaming guard.

A compression ratio over raw output cannot tell "the model is re-emitting the
same content" from "the content is legitimately regular": a correct 68-row
table is mostly ``<td>..</td>`` markup and short cell values, and compresses as
hard as a loop does (OmniDocBench page-213e6a57, 8k-window zlib 12.2, a false
abort). The two statistics here look at what is being repeated instead.

Content model
    letters   language: the thing a loop re-emits
    numbers   data: a table of different numbers is new information
    markup, JSON keys and punctuation runs: structure, removed first

``periodic``   The longest exactly-repeating suffix of the structure-free,
               number-masked tail. Its score is the content re-emitted verbatim
               in that run: letters, plus digits of numbers that never change.
               It only counts when the masked numbers are constant or step
               regularly (a counter). Rows that differ in a name, or in numbers
               that follow no constant step, are not periodic. A stepping
               number never scores: a conversion table whose every column
               steps by a constant is real, and re-emits nothing.
``repeated``   Characters of the last ``WINDOW`` content characters that sit in
               units (sentence, line, table row, JSON string) already emitted
               verbatim at least ``REPEAT_MIN_PRIOR`` times earlier in the same
               output, counted only while almost none of the window is new.
               Catches long cycles that drift and so never repeat exactly. A
               page that repeats boilerplate around new items (legal notices)
               keeps bringing new units and scores 0.

Both are pure functions of the text so far, so an offline replay of a saved
completion reproduces the serving decision exactly
(``scripts/loopguard_replay.py``).
"""
from __future__ import annotations

import re
from collections import Counter

RAW_TAIL = 64000          # raw chars examined for the periodic tail (markup-heavy tables shrink ~6x)
PERIODIC_WINDOW = 16000   # content chars the periodic tail is searched in (240 numbered rows fit)
WINDOW = 8000             # content chars the repeated statistic is measured over
MIN_REPEATS = 3           # a periodic run is at least this many periods
REGULAR_MAX = 0.10        # share of number steps allowed to break a slot's constant step
UNIT_MIN_LETTERS = 12     # shorter units (cell values, labels) are not evidence
REPEAT_MIN_PRIOR = 3      # a unit counts once it was already emitted this many times
NOVEL_MAX = 0.10          # ...and only while at most this share of the window's units is new
ROW_MAX_CONTENT = 200     # a numbered unit re-emitting less than this per repeat is a row, not a paragraph

_B = "\x00"               # unit boundary left where structure was removed
_ESC = re.compile(r"\\[nrt]")
_ROW_END = re.compile(r"</tr>|<br\s*/?>|</p>|</li>")
_TAG = re.compile(r"<[^<>]{0,80}>")
_KEY = re.compile(r'"[A-Za-z_]{1,24}":|\bnull\b')
_JSON_SEP = re.compile(r'"\s*[,}\]]+|[{\[]\s*"')
_PUNCT_RUN = re.compile(r"([^\w\s]|_)\1{2,}")
_WS = re.compile(r"[ \t\r\n]+")
_PARTIAL = re.compile(r"(?:<[^<>]{0,80}|\\)\Z")     # a tag or escape the stream has not finished yet
_SENTENCE = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])")
_CJK_NUM = "〇零一二三四五六七八九十百千两"
_NUM = re.compile(
    r"\d+(?:\.\d+)?"
    r"|[" + _CJK_NUM + r"]+"
    r"|\b(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b"
)
_CJK_DIGIT = {c: i for i, c in enumerate("零一二三四五六七八九")}
_CJK_DIGIT.update({"〇": 0, "两": 2})
_CJK_UNIT = {"十": 10, "百": 100, "千": 1000}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def content(raw: str, units: bool = False) -> str:
    """``raw`` with markup removed: JSON escapes, HTML tags, runs of one
    punctuation mark. ``units=True`` also removes JSON keys and separators and
    leaves a boundary character where a line, table row or JSON string ended.
    The periodic view keeps the keys: a run of near-empty chunks or figure
    entries re-emits them, and they are all there is to see."""
    b = _B if units else " "
    s = _ESC.sub(b, raw)
    s = _ROW_END.sub(b, s)
    s = _TAG.sub(" ", s)
    if units:
        s = _KEY.sub(b, s)
        s = _JSON_SEP.sub(b, s)
    s = s.replace('\\"', '"')
    s = _PUNCT_RUN.sub(r"\1", s)
    return _WS.sub(" ", s)


def _number(tok: str) -> tuple[float, int]:
    """(value, decimals) of an Arabic, Roman or CJK numeral token."""
    c = tok[0]
    if c.isdigit():
        return float(tok[:15]), (len(tok) - tok.index(".") - 1 if "." in tok else 0)
    if c in _ROMAN:
        total = 0
        for a, nxt in zip(tok, tok[1:] + " "):
            total += -_ROMAN[a] if _ROMAN.get(nxt, 0) > _ROMAN[a] else _ROMAN[a]
        return float(total), 0
    if not any(k in tok for k in _CJK_UNIT):          # positional: 二〇二五
        total = 0
        for ch in tok:
            total = total * 10 + _CJK_DIGIT.get(ch, 0)
        return float(total), 0
    total = cur = 0                                    # 四十五, 一百零三
    for ch in tok:
        if ch in _CJK_DIGIT:
            cur = _CJK_DIGIT[ch]
        else:
            total += (cur or 1) * _CJK_UNIT[ch]
            cur = 0
    return float(total + cur), 0


def _z(s: str) -> list[int]:
    n = len(s)
    z = [0] * n
    left = right = 0
    for i in range(1, n):
        if i < right:
            k = z[i - left]
            if k < right - i:
                z[i] = k
                continue
            k = right - i
        else:
            k = 0
        while i + k < n and s[k] == s[i + k]:
            k += 1
        z[i] = k
        if i + k > right:
            left, right = i, i + k
    return z


def _periodic_tail(t: str) -> tuple[int, int]:
    """(re-emitted chars, period) of the longest suffix of ``t`` that repeats
    with one period at least MIN_REPEATS times; (0, 0) when there is none."""
    z = _z(t[::-1])
    best = (0, 0)
    for period in range(1, len(t) // MIN_REPEATS + 1):
        run = z[period]
        if run >= (MIN_REPEATS - 1) * period and run > best[0]:
            best = (run, period)
    return best


def _numbers(nums: list[tuple[float, int, int]], per_period: int) -> tuple[float, int, bool]:
    """Numbers of a periodic run, ``per_period`` slots per period.

    Returns (irregular, constant_chars, stepping): the share of steps that
    break their slot's constant step (0.0 = every number is constant or counts
    regularly), the characters per period of numbers that never change, and
    whether any slot counts (its usual step is not zero).
    """
    periods = len(nums) // per_period if per_period else 0
    if periods < MIN_REPEATS:
        return 0.0, 0, False
    tail = nums[-periods * per_period:]
    total = bad = constant = 0
    stepping = False
    for slot in range(per_period):
        col = tail[slot::per_period]
        decimals = max(d for _, d, _ in col)
        values = [v for v, _, _ in col]
        steps = sorted(b - a for a, b in zip(values, values[1:]))
        median = steps[len(steps) // 2]
        tol = 1.5 * 10 ** -decimals if decimals else 0.0
        bad += sum(abs(s - median) > tol for s in steps)
        total += len(steps)
        if steps[0] == 0 == steps[-1]:
            constant += col[-1][2]
        elif abs(median) > tol:
            stepping = True
    return bad / total, constant, stepping


def _unmasked_start(st: str, spans: list[tuple[int, int]], masked_len: int) -> int:
    """Offset in ``st`` where its last ``masked_len`` masked characters begin
    (each number in ``spans`` is one masked character)."""
    pos, need = len(st), masked_len
    for start, end in reversed(spans):
        if need <= pos - end:
            break
        need -= pos - end
        pos = end
        if need:
            need -= 1
            pos = start
    return max(0, pos - need)


def periodic(raw: str) -> dict:
    """Periodic-tail statistic of the output so far.

    ``score`` = content re-emitted verbatim by the periodic run: its letters
    plus the digits of numbers that never change. 0 when the tail is not
    periodic or its numbers are irregular (data, not a loop). A number that
    steps (a counter, a linear table column) is allowed but never scores.

    ``stepping`` = some number in the run counts (``Item 7 ...``, ``Item 8
    ...``; every looping chunk does, through its chunk_id). ``numbered_rows`` =
    a stepping run that re-emits less than ROW_MAX_CONTENT characters per
    repeat. A model does loop that way, but a payment schedule, an answer
    sheet or an exhibit list of lots is the same shape (the legitimate cases
    re-emit up to ~110 characters per row), so the guard holds numbered rows to a
    higher threshold than a run that repeats identically or in paragraphs.

    ``numeric_repeats`` = cycles of a run that holds numbers and no letters at
    all (``4012, 4013, 4014, ...``), or of numbered rows. Such a run has
    little or no content to score and cannot be told from a number grid, a
    numbered list or a table whose columns all step by a constant, except by
    its length.

    ``period`` is measured on the number-masked text; ``unit`` is the last
    period as it was emitted (``period_chars`` long).
    """
    st = content(_PARTIAL.sub("", raw[-RAW_TAIL:]))[-PERIODIC_WINDOW:]
    spans = [m.span() for m in _NUM.finditer(st) if m.group()]
    masked = _NUM.sub(lambda m: "0" if m.group() else "", st)
    run, period = _periodic_tail(masked)
    out = {"score": 0, "numeric_repeats": 0.0, "chars": run, "period": period, "repeats": 0.0,
           "irregular": 0.0, "stepping": False, "numbered_rows": False, "window_chars": len(st), "unit": "", "period_chars": 0}
    if not period:
        return out
    unit = masked[-period:]
    out["repeats"] = round((run + period) / period, 1)
    out["unit"] = st[_unmasked_start(st, spans, period):]
    out["period_chars"] = len(out["unit"])
    per_period = unit.count("0")
    constant = 0.0
    k = 1
    if per_period:
        covered = masked[-(run + period):].count("0")
        nums = [_number(st[a:b]) + (b - a,) for a, b in spans[-covered:]]
        # Numbers that cycle (three rows repeating) are regular at a multiple of
        # the period: judge at the multiple that explains them best.
        irregular, k, per_k, stepping = min(
            (r[0], k, r[1], r[2]) for k, r in
            ((k, _numbers(nums, per_period * k)) for k in (1, 2, 3, 4)
             if k == 1 or run + period >= MIN_REPEATS * k * period))
        out["irregular"] = round(irregular, 3)
        out["stepping"] = stepping
        constant = per_k / k
    if out["irregular"] <= REGULAR_MAX:
        letters = sum(c.isalpha() for c in unit)
        out["score"] = int(run * (letters + constant) / period)
        out["numbered_rows"] = out["stepping"] and letters + constant < ROW_MAX_CONTENT
        if per_period and (not letters or out["numbered_rows"]):
            # Counted in whole cycles of the numbers: a 60-row conversion table
            # with four columns is 60 repeats, not 240.
            out["numeric_repeats"] = round(out["repeats"] / k, 1)
    return out


def repeated(raw: str) -> dict:
    """Re-emitted-unit statistic of the output so far.

    ``score`` = characters of the window in units already emitted at least
    REPEAT_MIN_PRIOR times, counted only while almost nothing in the window is
    new (``novel`` <= NOVEL_MAX). A page of legal notices repeats its
    boilerplate a dozen times, but every notice also brings its own address
    and case number; a model cycling through the same notices brings nothing.
    """
    s = content(raw, units=True)
    floor = len(s) - WINDOW
    seen: Counter = Counter()
    hits: Counter = Counter()
    pos = chars = total = novel = 0
    for segment in s.split(_B):
        for piece in _SENTENCE.split(segment):
            pos += len(piece)
            unit = piece.strip()
            if sum(c.isalpha() for c in unit) < UNIT_MIN_LETTERS:
                continue
            # Units are compared as emitted, numbers included. Masking them
            # here would turn every numbered list that shares a phrase (a valve
            # list, an answer sheet) into "re-emitted" text; the cost is that a
            # loop changing a number on every repeat is left to the length cap.
            prior = seen[unit]
            if pos > floor:
                total += len(unit)
                if not prior:
                    novel += len(unit)
                elif prior >= REPEAT_MIN_PRIOR:
                    chars += len(unit)
                    hits[unit] += 1
            seen[unit] = prior + 1
        pos += 1
    share = novel / total if total else 1.0
    top = [{"times": seen[u], "unit": u[:160]} for u, _ in hits.most_common(3)]
    return {"score": chars if share <= NOVEL_MAX else 0, "chars": chars, "novel": round(share, 3),
            "window_chars": min(len(s), WINDOW), "top": top}
