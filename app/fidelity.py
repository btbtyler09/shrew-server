"""Deterministic-source fidelity cross-check (flag-only).

Origin: a user's workspace agent reported a document "typo" —
DCRrectifierControl_PH, extra R — that did not exist in the source Word file.
Reproduced 2026-08-28 on the real pipeline (shrew fine-tune, temp 0,
deterministic): the table *transcription* was faithful, but the model's
generated summary blended the caption ("DCR instance mapping") with the type
name and emitted DCRRectifierControl_PH. Copying is safe; composing prose
ABOUT precision strings is the hazard.

The check: build a vocabulary of "precision tokens" from a deterministic
extraction of the source (office → LibreOffice PDF text layer, born-digital
PDF → text layer), then scan every model-generated output field for precision
tokens that don't exist in the source. Two detection modes:

- unknown: identifier-shaped tokens (camelCase/underscore, len >= 6) absent
  from the source are flagged outright — they are specific enough that
  absence is itself evidence.
- near-miss: ANY precision token (digit-bearing codes: 7.15, IEC-61850-7-4,
  0x1A2B, M8x1.25, v2.14.3, 22.5; ALL-CAPS acronyms: SCADA, DNP3) absent from
  the source but within a small edit distance of a same-class source token —
  direct corruption evidence for classes too noisy to flag on absence alone.

Precision guards (false positives are the failure mode here):
- plain prose is never checked — paraphrasing English is the model's job;
- bare integers of <= 3 digits are exempt (summaries legitimately compose
  counts the source never writes: "lists 30 instances");
- distance budget scales with token length (<=1 under 7 chars, <=2 at 7+);
- near-miss candidates come from the same token class only.

Results ride processing_log["fidelity"]. Flagged IDENTIFIERS with a unique
source match are deterministically corrected in place before rendering (see
apply_corrections — evidence-backed: the source is ground truth for its own
identifiers); numbers, codes, and acronyms stay flag-only. Every correction is
recorded on its flag. SHREW_FIDELITY_CORRECT=0 keeps flags but rewrites
nothing; SHREW_FIDELITY=0 disables the layer entirely.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("shrew.fidelity")

# Compound token: alnum runs joined by internal . _ - (so "22.5 Nm" yields
# "22.5", never "22.5."; "IEC-61850-7-4" survives whole).
_TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._\-]*[A-Za-z0-9])?")
_CAMEL_RE = re.compile(r"[a-z][A-Z]")

# At most this many distinct flags per document — a report, not a firehose.
MAX_FLAGS = 100


def _classify(token: str) -> str | None:
    """Precision class of a token, or None for prose (never checked)."""
    has_digit = any(c.isdigit() for c in token)
    # Identifier-shaped: underscore or a camelCase boundary, long enough to be
    # specific. Checked FIRST so DCRectifierStatus01 / FooBarBazQuux_99 land
    # here (unknown-flaggable), not in the near-miss-only code class.
    if len(token) >= 6 and ("_" in token or _CAMEL_RE.search(token)):
        return "ident"
    if has_digit:
        return "code"
    if len(token) >= 3 and token.isupper():
        return "acronym"
    return None


def _budget(token: str) -> int:
    """Edit-distance budget for near-miss matching."""
    return 2 if len(token) >= 7 else 1


_BARE_NUMERIC_RE = re.compile(r"[0-9]+(?:[.,\-][0-9]+)*")


def _is_bare_numeric(token: str) -> bool:
    """Purely numeric (decimals, ranges, dates): 22.4, 7.5, 2024, 1-5.
    A distance-1 neighbor of a number proves little, and rewriting a value is
    the one unforgivable failure — these are flag-only, never corrected."""
    return bool(_BARE_NUMERIC_RE.fullmatch(token))


def _is_boundary(v: str, i: int) -> bool:
    """Is position i a segment boundary of token v? Segments split on
    separators (. _ -), camelCase transitions (incl. acronym->Word: the R in
    DCRectifier), and digit/letter transitions."""
    if i == 0 or i == len(v):
        return True
    a, b = v[i - 1], v[i]
    if a in "._-" or b in "._-":
        return True
    if a.islower() and b.isupper():
        return True
    if a.isupper() and b.isupper() and i + 1 < len(v) and v[i + 1].islower():
        return True
    return a.isdigit() != b.isdigit()


def _segment_aligned(token: str, v: str) -> bool:
    """Does token appear in v as a run of WHOLE segments? DCRectifierStatus
    inside DCRectifierStatus01 is a legit family reference (ends at the digit
    boundary); DCRectifierControl_P inside ..._PH is truncation (ends
    mid-segment) and must NOT be exempted."""
    start = v.find(token)
    while start != -1:
        if _is_boundary(v, start) and _is_boundary(v, start + len(token)):
            return True
        start = v.find(token, start + 1)
    return False


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (small strings; row-wise DP)."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract_vocab(text: str | None) -> set[str]:
    """All precision tokens in a deterministic source extraction."""
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text) if _classify(t) is not None}


def _closest(token: str, vocab_by_class: dict[str, list[str]],
             cls: str) -> tuple[str | None, int | None, bool]:
    """Best same-class vocab match within the distance budget.

    Returns (match, distance, ambiguous) — ambiguous means a DIFFERENT vocab
    token sits at the same best distance, so a correction cannot know which
    was meant. Full scan, no early exit: ambiguity detection needs every
    candidate at the best distance.
    """
    budget = _budget(token)
    best, best_d, ties = None, budget + 1, 0
    for cand in vocab_by_class.get(cls, ()):
        if abs(len(cand) - len(token)) > budget:
            continue
        d = edit_distance(token, cand)
        if d < best_d:
            best, best_d, ties = cand, d, 1
        elif d == best_d and cand != best:
            ties += 1
    if best is None:
        return None, None, False
    return best, best_d, ties > 1


def _vocab_by_class(vocab: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for v in vocab:
        cls = _classify(v)
        if cls:
            out.setdefault(cls, []).append(v)
    return out


def _vocab_by_lower(vocab: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for v in vocab:
        out.setdefault(v.lower(), set()).add(v)
    return out


def check_output(text: str | None, vocab: set[str],
                 _by_class: dict[str, list[str]] | None = None,
                 _lower: dict[str, set[str]] | None = None) -> list[dict]:
    """Flag precision tokens in one output text that the source cannot back.

    Returns [{token, closest, distance, class, ambiguous, case_only}] in
    first-occurrence order, deduplicated. Empty vocab (no deterministic
    source) flags nothing.
    """
    if not text or not vocab:
        return []
    by_class = _by_class if _by_class is not None else _vocab_by_class(vocab)
    if _lower is None:
        _lower = _vocab_by_lower(vocab)
    flagged: list[dict] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(text):
        if token in seen or token in vocab:
            continue
        cls = _classify(token)
        if cls is None:
            continue
        # Bare short integers are aggregation noise ("30 instances"), never
        # corruption evidence.
        if cls == "code" and token.isdigit() and len(token) <= 3:
            continue
        # A segment-aligned stem of a source token is backed by the source:
        # summaries legitimately name FAMILIES ("the DCRectifierStatus
        # instances", "firmware v2.14") when the source only writes fuller
        # forms (DCRectifierStatus01, v2.14.3). Alignment matters: a token
        # ending mid-segment (DCRectifierControl_P) is truncation, not a stem.
        # Found live 2026-08-28.
        if any(_segment_aligned(token, v) for v in vocab):
            continue
        # Case rescue: a unique case-insensitive match is corruption of the
        # CASING only (DCRECTIFIERCONTROL_PH). Case flips blow past the edit
        # budget, so this must be its own check — and it is the safest
        # correction of all.
        ci = _lower.get(token.lower()) if _lower is not None else None
        if ci is not None and len(ci) == 1:
            m = next(iter(ci))
            seen.add(token)
            flagged.append({"token": token, "closest": m,
                            "distance": edit_distance(token, m), "class": cls,
                            "ambiguous": False, "case_only": True})
            continue
        closest, dist, ambiguous = _closest(token, by_class, cls)
        if cls == "ident" or closest is not None:
            seen.add(token)
            flagged.append({"token": token, "closest": closest, "distance": dist,
                            "class": cls, "ambiguous": ambiguous,
                            "case_only": False})
    return flagged


def iter_output_texts(doc: dict):
    """(where, text) for every model-GENERATED or transcribed field of an
    assembled doc record. Missing keys are fine — yields what exists."""
    if doc.get("doc_summary"):
        yield "doc_summary", doc["doc_summary"]
    meta = doc.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("title"):
        yield "metadata.title", str(meta["title"])
    for c in doc.get("chunks") or []:
        where = f"chunk[{c.get('chunk_id')}] (page {c.get('page')})"
        if c.get("title"):
            yield where + ".title", c["title"]
        if c.get("content"):
            yield where, c["content"]
    for i, t in enumerate(doc.get("tables") or []):
        yield f"table[{i}] (pages {t.get('pages') or t.get('page')})", \
            t.get("flat_text") or t.get("html") or ""
    for i, f in enumerate(doc.get("figures") or []):
        if f.get("caption"):
            yield f"figure[{i}].caption (page {f.get('page')})", f["caption"]


def check_document(doc: dict, source_text: str | None) -> dict | None:
    """Cross-check an assembled doc against the deterministic source text.

    Returns {"flagged": [{token, closest, distance, where, count}],
    "vocab_size", "checked"} — or None when there is no deterministic source
    (scanned PDF, image upload): unavailable, not "all clear".
    """
    if not source_text or not source_text.strip():
        return None
    vocab = extract_vocab(source_text)
    by_class = _vocab_by_class(vocab)
    lower_map = _vocab_by_lower(vocab)
    merged: dict[str, dict] = {}
    checked = 0
    for where, text in iter_output_texts(doc):
        checked += 1
        for f in check_output(text, vocab, by_class, lower_map):
            if f["token"] in merged:
                merged[f["token"]]["count"] += 1
            elif len(merged) < MAX_FLAGS:
                merged[f["token"]] = {**f, "where": where, "count": 1}
    report = {"flagged": list(merged.values()), "vocab_size": len(vocab),
              "checked": checked}
    if report["flagged"]:
        logger.warning(
            "fidelity: %d precision token(s) not backed by the source: %s",
            len(report["flagged"]),
            ", ".join(f"{f['token']}"
                      + (f" (source: {f['closest']})" if f["closest"] else "")
                      for f in report["flagged"][:10]))
    return report


def _correctable(f: dict) -> bool:
    """Per-class correction risk gate — see apply_corrections docstring."""
    if f.get("case_only"):
        return True
    cls = f.get("class")
    if cls == "ident":
        return True
    return cls == "code" and not _is_bare_numeric(f["token"])


def _field_refs(doc: dict):
    """(container, key, where) for every mutable output text field — the
    write-side twin of iter_output_texts. Table html AND flat_text are both
    yielded: a correction must land in whichever representation a consumer
    reads."""
    if doc.get("doc_summary"):
        yield doc, "doc_summary", "doc_summary"
    meta = doc.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("title"):
        yield meta, "title", "metadata.title"
    for c in doc.get("chunks") or []:
        where = f"chunk[{c.get('chunk_id')}]"
        if c.get("title"):
            yield c, "title", where + ".title"
        if c.get("content"):
            yield c, "content", where
    for i, t in enumerate(doc.get("tables") or []):
        for key in ("html", "flat_text"):
            if t.get(key):
                yield t, key, f"table[{i}].{key}"
    for i, f in enumerate(doc.get("figures") or []):
        if f.get("caption"):
            yield f, "caption", f"figure[{i}].caption"


def apply_corrections(doc: dict, report: dict | None) -> int:
    """Deterministically repair flagged IDENTIFIERS in place; returns the
    number of distinct tokens corrected.

    Substitution is evidence-backed, not model-backed: the deterministic
    source is ground truth for its own precision strings, so when a flagged
    token has a UNIQUE source match and the corrupt form exists nowhere in
    the source, replacing it restores the source spelling. Per-class risk
    gating (see _correctable):
    - identifiers and structured codes (letters+digits: versions, hex, part
      codes, standard refs) are corrected;
    - case-only mismatches are corrected for every class — the safest
      correction of all;
    - bare numerics never are (22.4 vs 22.5 might be a legitimately
      different value; rewriting a number is the one unforgivable failure);
    - pure acronyms never are (source says HTTP, model says HTTPS — possibly
      the model CORRECTLY adding knowledge);
    - unique matches only — an ambiguous tie means we cannot know which
      source token was meant;
    - word-boundary substitution — never rewrites inside a longer token.
    Every flag gains "corrected": True/False so the report shows exactly what
    was rewritten. Disable with SHREW_FIDELITY_CORRECT=0 (flags remain).
    """
    if not report or not report.get("flagged"):
        return 0
    enabled = os.environ.get("SHREW_FIDELITY_CORRECT", "1") != "0"
    subs = {}
    for f in report["flagged"]:
        f.setdefault("corrected", False)
        if (enabled and f.get("closest") and not f.get("ambiguous")
                and _correctable(f)):
            subs[f["token"]] = f
    if not subs:
        return 0
    pattern = re.compile(
        r"(?<![A-Za-z0-9._\-])("
        + "|".join(re.escape(t) for t in sorted(subs, key=len, reverse=True))
        + r")(?![A-Za-z0-9._\-])")
    corrected: set[str] = set()

    def _sub(m):
        corrected.add(m.group(1))
        return subs[m.group(1)]["closest"]

    for container, key, _where in _field_refs(doc):
        container[key] = pattern.sub(_sub, container[key])
    for t in corrected:
        subs[t]["corrected"] = True
        logger.warning("fidelity: corrected %s -> %s (distance %s)",
                       t, subs[t]["closest"], subs[t]["distance"])
    return len(corrected)
