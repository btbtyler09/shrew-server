"""Wrap bare LaTeX in display delimiters (v0.5 post-process).

v0.5 sometimes writes a derivation as raw LaTeX with no ``$``/``$$`` around it:

    \\begin{aligned} a &= b \\\\ c &= d \\end{aligned}
    A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}.
    \\frac{f'(x)}{f(x)} = -\\frac{4x - 1}{2x - 1}

Downstream renderers and the OmniDocBench matcher only see a display formula
when it is delimited, so these score as "omitted" (class 3a, 10-11: +2.16 CDM
on v0.4 when wrapped).  ``wrap_bare_latex`` is a pure, deterministic,
idempotent text -> text function:

1. A math environment (``\\begin{aligned}..\\end{aligned}``, matrices, cases,
   align, ...) that sits outside any math delimiter, code fence, or table row
   is wrapped.  If the rest of its line is pure math (no prose), the whole line
   becomes one ``$$..$$`` display; if the line also holds prose, only the
   environment is wrapped, inline (``mixed="inline"``, default) or as
   ``$$..$$`` (``mixed="display"``).
2. A line that is pure LaTeX (at least one math command, no prose, no
   delimiters) is wrapped in ``$$..$$``.

Never touched: code fences, existing ``$..$`` / ``$$..$$`` / ``\\(..\\)`` /
``\\[..\\]`` spans, headings, list items, blockquotes, table rows, HTML lines,
strikethrough, and non-math environments (enumerate, tabular, ...).  Text
outside the wrapped spans is byte-identical, and a wrap only inserts ``$``
characters, so ``text.replace("$", "")`` of the output equals that of the input.
"""
from __future__ import annotations

import re

MATH_ENVS = frozenset({
    "aligned", "align", "alignat", "flalign", "gather", "gathered", "multline",
    "equation", "eqnarray", "split", "cases", "array", "matrix", "pmatrix",
    "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "subarray",
    "math", "displaymath", "dcases", "rcases",
})

_BEGIN = re.compile(r"\\begin\{([A-Za-z]+\*?)\}")
_FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", re.S | re.M)
_PROTECTED = [
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\\\[.*?\\\]", re.S),
    re.compile(r"\\\(.*?\\\)", re.S),
    re.compile(r"(?<![\\$])\$(?!\$)[^$\n]+?(?<![\\$])\$(?!\$)"),
]
_CMD = re.compile(r"\\([A-Za-z]+)")
_TEXTLIKE = re.compile(r"\\(?:text|mathrm|mbox|operatorname|textbf|textit|mathbf|mathit|mathsf|mathtt)\s*\{[^{}]*\}")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WORD = re.compile(r"[A-Za-z]{2,}")
# Words that legitimately sit between formulas on a display line.
_GLUE = frozenset({"and", "or", "for", "if", "then", "where", "with", "when", "so", "is", "at",
                   "in", "on", "of", "by", "to", "as", "iff", "else", "otherwise", "mod"})
# Multi-letter identifiers that are math, not prose.
_MATHWORD = frozenset({"dx", "dy", "dz", "dt", "du", "dv", "dw", "dr", "ds", "dp", "dq", "dtheta",
                       "det", "tr", "rank", "dim", "ker", "gcd", "lcm", "max", "min", "sup", "inf",
                       "sgn", "arg", "exp", "log", "ln", "sin", "cos", "tan", "cot", "sec", "csc",
                       "sinh", "cosh", "tanh", "lim", "Pr", "Var", "Cov", "cdf", "pdf", "mol", "kg",
                       "Re", "Im", "Hom", "End", "Aut", "Ext", "Tor", "id", "im", "mod"})
_LINE_SKIP = re.compile(r"^\s*(#|\||<|>|[-*+]\s|\d+[.)]\s|~~|!\[|\[\^|=====|-----)")
_LEAD = "\u2234\u2235"  # therefore / because


def _spans_of(text: str, pats) -> list[tuple[int, int]]:
    out = []
    for p in pats:
        out.extend((m.start(), m.end()) for m in p.finditer(text))
    return out


def _protected(text: str) -> list[tuple[int, int]]:
    """Sorted, merged spans that must not be edited (fences, existing math)."""
    sp = _spans_of(text, [_FENCE])
    # delimiter spans are computed outside fences only
    masked = list(text)
    for a, b in sp:
        for i in range(a, b):
            if masked[i] != "\n":
                masked[i] = " "
    sp += _spans_of("".join(masked), _PROTECTED)
    sp.sort()
    merged: list[list[int]] = []
    for a, b in sp:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _inside(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def _env_spans(text: str, prot) -> list[tuple[int, int]]:
    """Top-level math \\begin{..}..\\end{..} spans outside protected regions."""
    out, i = [], 0
    while True:
        m = _BEGIN.search(text, i)
        if not m:
            return out
        s, name = m.start(), m.group(1)
        if _inside(s, prot):
            i = m.end()
            continue
        # match nested same-name environments by depth
        tok = re.compile(r"\\(begin|end)\{" + re.escape(name) + r"\}")
        depth, e = 0, None
        for t in tok.finditer(text, s):
            depth += 1 if t.group(1) == "begin" else -1
            if depth == 0:
                e = t.end()
                break
        if e is None:                      # unclosed: leave alone
            i = m.end()
            continue
        if name.rstrip("*") in MATH_ENVS and not any(_inside(k, prot) for k in (s, e - 1)):
            out.append((s, e))
        i = e


def _has_prose(residue: str) -> bool:
    r = _TEXTLIKE.sub(" ", residue)
    r = _CMD.sub(" ", r)
    if _CJK.search(r):
        return True
    words = [w for w in re.findall(r"[A-Za-z]+", r)]
    toks = [w for w in words if len(w) >= 2 and w.lower() not in _MATHWORD and w not in _MATHWORD]
    toks = [w for w in toks if w.lower() not in _GLUE]
    if not toks:
        return False
    # a single bare identifier-like token (AB, Ab, xy, dx) is math; real prose has >=2 words
    long = [w for w in toks if len(w) >= 4]
    return len(toks) >= 2 or bool(long)


def _pure_math(s: str) -> bool:
    return not _has_prose(s) and "$" not in s


def _wrap_line(body: str) -> str:
    return "$$" + body.strip() + "$$"


def _line_extent(text: str, s: int, e: int) -> tuple[int, int]:
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    return ls, (len(text) if le < 0 else le)


def _wrap_envs(text: str, mixed: str) -> str:
    prot = _protected(text)
    spans = _env_spans(text, prot)
    if not spans:
        return text
    # cluster spans that share a line (no newline between consecutive spans)
    clusters: list[list[tuple[int, int]]] = []
    for sp in spans:
        if clusters and "\n" not in text[clusters[-1][-1][1]:sp[0]]:
            clusters[-1].append(sp)
        else:
            clusters.append([sp])
    edits: list[tuple[int, int, str]] = []
    for cl in clusters:
        ls, le = _line_extent(text, cl[0][0], cl[-1][1])
        line = text[ls:le]
        if _LINE_SKIP.match(line) or any(a < le and b > ls for a, b in prot):
            continue
        residue = text[ls:cl[0][0]]
        for (a1, b1), (a2, _b2) in zip(cl, cl[1:]):
            residue += " " + text[b1:a2]
        residue += " " + text[cl[-1][1]:le]
        if _pure_math(residue.replace(_LEAD[0], " ").replace(_LEAD[1], " ")):
            lead = line[:len(line) - len(line.lstrip())]
            edits.append((ls, le, lead + _wrap_line(line)))
        else:
            for a, b in cl:
                body = text[a:b]
                edits.append((a, b, ("$" + body + "$") if mixed == "inline" else ("$$" + body + "$$")))
    for a, b, new in sorted(edits, reverse=True):
        text = text[:a] + new + text[b:]
    return text


def _wrap_pure_lines(text: str) -> str:
    prot = _protected(text)
    out, pos = [], 0
    for line in text.split("\n"):
        end = pos + len(line)
        keep = True
        t = line.strip()
        if (t and not _LINE_SKIP.match(line) and "$" not in t and "\\begin" not in t
                and not any(a < end and b > pos for a, b in prot)):
            cmds = [c for c in _CMD.findall(t) if c not in ("begin", "end")]
            body = t.lstrip(_LEAD + " ")
            if cmds and body and _pure_math(t.replace(_LEAD[0], " ").replace(_LEAD[1], " ")):
                lead = line[:len(line) - len(line.lstrip())]
                out.append(lead + _wrap_line(t))
                keep = False
        if keep:
            out.append(line)
        pos = end + 1
    return "\n".join(out)


def wrap_bare_latex(text, *, mixed: str = "inline"):
    """Return ``text`` with bare LaTeX wrapped in math delimiters (see module doc).

    ``mixed``: how an environment on a line that also contains prose is wrapped,
    "inline" ($..$) or "display" ($$..$$).  Non-str input is returned unchanged.
    """
    if not isinstance(text, str) or "\\" not in text:
        return text
    if mixed not in ("inline", "display"):
        raise ValueError("mixed must be 'inline' or 'display'")
    return _wrap_pure_lines(_wrap_envs(text, mixed))


def wrap_record(rec: dict, *, mixed: str = "inline") -> tuple[dict, dict]:
    """Apply ``wrap_bare_latex`` to a saved /v1/convert-style record, offline.

    Returns ``(new_record, stats)``.  Wraps ``semantic_chunks[].content`` and
    ``markdown`` (assembled from the chunks); every other key is shared with the
    input.  ``stats`` = {"chunks_changed", "markdown_changed", "wraps"}.
    """
    new = dict(rec)
    changed = 0
    chunks = rec.get("semantic_chunks")
    if isinstance(chunks, list):
        new_chunks = []
        for c in chunks:
            nc = dict(c) if isinstance(c, dict) else c
            if isinstance(c, dict) and isinstance(c.get("content"), str):
                w = wrap_bare_latex(c["content"], mixed=mixed)
                if w != c["content"]:
                    nc["content"] = w
                    changed += 1
            new_chunks.append(nc)
        new["semantic_chunks"] = new_chunks
    md = rec.get("markdown")
    md_changed = 0
    if isinstance(md, str):
        w = wrap_bare_latex(md, mixed=mixed)
        if w != md:
            new["markdown"] = w
            md_changed = 1
    stats = {"chunks_changed": changed, "markdown_changed": md_changed}
    return new, stats


def env_settings() -> tuple[bool, str]:
    """(enabled, mixed) from SHREW_LATEX_WRAP / SHREW_LATEX_WRAP_MIXED. Raises
    ValueError on a malformed value: the server calls this at startup so a typo
    stops the boot instead of failing each request after the model work is done."""
    import os
    raw = os.environ.get("SHREW_LATEX_WRAP", "1").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        enabled = True
    elif raw in ("0", "false", "no", "off"):
        enabled = False
    else:
        raise ValueError(f"SHREW_LATEX_WRAP must be one of 1/0, true/false, yes/no, on/off, got {raw!r}")
    mixed = os.environ.get("SHREW_LATEX_WRAP_MIXED", "inline").strip().lower()
    if mixed not in ("inline", "display"):
        raise ValueError(f"SHREW_LATEX_WRAP_MIXED must be 'inline' or 'display', got {mixed!r}")
    return enabled, mixed
