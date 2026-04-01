"""Deterministic markdown post-processor.

Cleans up agent output with regex-based passes:
- Deduplication of repeated blocks (page boundary overlaps)
- Artifact removal (<!-- image -->, stray page numbers, repeated headers)
- Reference cleanup (- - - artifacts)
- Citation style normalization
- Subscript/superscript normalization
- Whitespace normalization
"""

import logging
import re
from collections import Counter
from difflib import SequenceMatcher

logger = logging.getLogger("shrew.postprocess")


def postprocess_markdown(text: str, dedup: bool = False) -> str:
    """Run all post-processing passes on markdown text.

    Args:
        dedup: Enable expensive dedup passes (paragraph, block, long-line).
            Useful for conventional/docling mode where page-boundary overlaps
            are common. Off by default for VLM mode.
    """
    original_len = len(text)

    text = _dedup_consecutive_lines(text)
    if dedup:
        text = _dedup_long_lines(text)
        text = _dedup_paragraphs(text)
        text = _dedup_blocks(text)
    text = _dedup_figure_captions(text)
    text = _renumber_figures(text)
    text = _remove_artifacts(text)
    text = _cleanup_references(text)
    text = _normalize_citations(text)
    text = _normalize_sub_superscripts(text)
    text = _normalize_equation_tags(text)
    text = _fix_math_headings(text)
    text = _remove_vlm_image_placeholders(text)
    text = _rejoin_hyphenated_words(text)
    text = _normalize_whitespace(text)

    logger.info(f"Post-processing: {original_len} -> {len(text)} chars")
    return text


# ── Pass -2: Consecutive Line Deduplication ───────────────────────────────────

def _dedup_consecutive_lines(text: str) -> str:
    """Remove consecutive duplicate lines (builder artifact).

    Catches cases like two identical headings back-to-back.
    Skips blank lines (those are intentional paragraph separators).
    """
    lines = text.split("\n")
    result = []
    removed = 0
    for line in lines:
        if result and line.strip() and line == result[-1]:
            removed += 1
            continue
        result.append(line)
    if removed:
        logger.info(f"Consecutive dedup: removed {removed} duplicate lines")
    return "\n".join(result)


# ── Pass -1: Intra-line Deduplication ─────────────────────────────────────────

def _dedup_long_lines(text: str, min_overlap: int = 400) -> str:
    """Detect and remove duplicated content in and across long lines.

    Catches two patterns:
    1. Intra-line: content repeated within a single very long line
    2. Cross-line: end of line N overlaps with content in line N+1
       (page boundary duplication where builder re-extracts content)
    """
    lines = text.split("\n")
    count = 0

    # Pass 1: Intra-line dedup
    for i, line in enumerate(lines):
        if len(line) < min_overlap * 2:
            continue
        n = len(line)
        best_start = -1
        best_len = 0
        for offset in range(n // 3, 2 * n // 3, 50):
            if offset + min_overlap > n:
                break
            chunk = line[offset:offset + 80]
            first_pos = line.find(chunk)
            if first_pos >= 0 and first_pos < offset - 40:
                dup_start = offset
                orig_start = first_pos
                match_len = 0
                while (dup_start + match_len < n and
                       orig_start + match_len < dup_start and
                       line[orig_start + match_len] == line[dup_start + match_len]):
                    match_len += 1
                if match_len >= min_overlap and match_len > best_len:
                    best_start = dup_start
                    best_len = match_len
        if best_start > 0 and best_len >= min_overlap:
            lines[i] = line[:best_start].rstrip()
            count += 1
            logger.info(f"Intra-line dedup: removed {best_len} chars from line {i+1}")

    # Pass 2: Cross-line overlap dedup
    for i in range(len(lines) - 1):
        line_a = lines[i]
        line_b = lines[i + 1]
        if len(line_a) < min_overlap or len(line_b) < min_overlap:
            continue
        # Check if a suffix of line_a appears in line_b
        # Test a chunk from the end of line_a
        check_len = min(200, len(line_a) // 2)
        suffix_chunk = line_a[-check_len:]
        pos = line_b.find(suffix_chunk)
        if pos < 0:
            continue
        # Found a match — extend backward to find the full overlap start in line_a
        # The overlap starts at some point in line_a and continues to its end
        overlap_start_b = pos
        overlap_in_a = len(line_a) - check_len
        # Extend backward
        while (overlap_in_a > 0 and overlap_start_b > 0 and
               line_a[overlap_in_a - 1] == line_b[overlap_start_b - 1]):
            overlap_in_a -= 1
            overlap_start_b -= 1
        overlap_len = len(line_a) - overlap_in_a
        if overlap_len >= min_overlap:
            # Remove the overlapping suffix from line_a
            lines[i] = line_a[:overlap_in_a].rstrip()
            count += 1
            logger.info(f"Cross-line dedup: removed {overlap_len} chars from end of "
                       f"line {i+1} (overlaps with line {i+2})")

    if count:
        logger.info(f"Long-line dedup: fixed {count} duplications")
    return "\n".join(lines)


# ── Pass 0: Paragraph-level Deduplication ─────────────────────────────────────

def _dedup_paragraphs(text: str, threshold: float = 0.75) -> str:
    """Remove duplicate paragraphs (split by blank lines).

    Catches page-boundary overlaps where a paragraph is re-extracted verbatim.
    Uses a lower threshold than block dedup since paragraph boundaries differ.
    """
    # Split into paragraphs (separated by 2+ newlines)
    parts = re.split(r"\n\s*\n", text)
    if len(parts) < 3:
        return text

    # Compare each paragraph against earlier ones
    # Skip numbered paragraphs (e.g., patent claims "1. ...", "2. ...") —
    # these are intentionally repetitive with legally significant variations
    _numbered_re = re.compile(r"^\d+\.\s")
    _page_tag_re = re.compile(r"</?page\s+\d+>")

    to_remove = set()
    for i in range(len(parts)):
        if i in to_remove:
            continue
        para_i = parts[i].strip()
        if len(para_i) < 60:  # Skip short paragraphs (headings, etc.)
            continue
        # Protect paragraphs containing page boundary tags
        if _page_tag_re.search(para_i):
            continue
        for j in range(i + 1, len(parts)):
            if j in to_remove:
                continue
            para_j = parts[j].strip()
            if len(para_j) < 60:
                continue
            if _page_tag_re.search(para_j):
                continue
            # Don't dedup numbered paragraphs (claims, list items)
            if _numbered_re.match(para_i) or _numbered_re.match(para_j):
                continue
            # Quick length check — skip obviously different paragraphs
            len_i, len_j = len(para_i), len(para_j)
            if len_j < len_i * threshold or len_j > len_i / threshold:
                continue
            ratio = SequenceMatcher(None, para_i, para_j).ratio()
            if ratio >= threshold:
                to_remove.add(j)
                logger.debug(f"Paragraph dedup: removing paragraph {j+1} "
                           f"(duplicate of {i+1}, ratio={ratio:.2f})")

    if to_remove:
        logger.info(f"Paragraph dedup: removed {len(to_remove)} duplicate paragraphs")
        parts = [p for i, p in enumerate(parts) if i not in to_remove]

    return "\n\n".join(parts)


# ── Pass 1: Block Deduplication ───────────────────────────────────────────────

def _dedup_blocks(text: str, min_block: int = 3, threshold: float = 0.92) -> str:
    """Remove duplicate blocks of 3+ consecutive lines.

    Keeps the first occurrence, removes later near-duplicates.
    Catches page-boundary overlaps where the builder re-extracts content.
    """
    lines = text.split("\n")
    if len(lines) < min_block * 2:
        return text

    # Build blocks of min_block consecutive non-empty lines
    # Skip page boundary tags — they're structural, not content duplicates
    _page_tag_re = re.compile(r"^</?page\s*\d*>")
    blocks = []
    for i in range(len(lines) - min_block + 1):
        block_lines = lines[i:i + min_block]
        # Skip blocks that are mostly empty
        content = "\n".join(block_lines).strip()
        if len(content) < 20:
            continue
        # Skip blocks that contain page boundary tags
        if any(_page_tag_re.match(l.strip()) for l in block_lines):
            continue
        blocks.append((i, content))

    # Find duplicate blocks — pre-filter by length to avoid O(n²) SequenceMatcher
    lines_to_remove = set()
    for i, (pos_a, content_a) in enumerate(blocks):
        if pos_a in lines_to_remove:
            continue
        len_a = len(content_a)
        for j in range(i + 1, len(blocks)):
            pos_b, content_b = blocks[j]
            if pos_b in lines_to_remove:
                continue
            # Skip overlapping blocks
            if pos_b < pos_a + min_block:
                continue
            # Quick length check — blocks with very different lengths can't match
            len_b = len(content_b)
            if len_b < len_a * threshold or len_b > len_a / threshold:
                continue
            ratio = SequenceMatcher(None, content_a, content_b).ratio()
            if ratio >= threshold:
                # Remove the later occurrence — expand to find full duplicate extent
                extend_start = pos_b
                extend_end = pos_b + min_block - 1
                # Try to extend the match forward
                while (extend_end + 1 < len(lines) and
                       pos_a + (extend_end - pos_b) + 1 < len(lines)):
                    next_orig = lines[pos_a + (extend_end - pos_b) + 1]
                    next_dup = lines[extend_end + 1]
                    if (next_orig.strip() and next_dup.strip() and
                            SequenceMatcher(None, next_orig, next_dup).ratio() >= threshold):
                        extend_end += 1
                    else:
                        break
                for k in range(extend_start, extend_end + 1):
                    lines_to_remove.add(k)
                dup_count = extend_end - extend_start + 1
                logger.debug(f"Dedup: removing lines {extend_start+1}-{extend_end+1} "
                           f"({dup_count} lines, duplicate of {pos_a+1}-{pos_a+dup_count})")

    if lines_to_remove:
        logger.info(f"Dedup: removed {len(lines_to_remove)} duplicate lines")
        lines = [l for i, l in enumerate(lines) if i not in lines_to_remove]

    return "\n".join(lines)


# ── Pass 1b: Figure Caption Deduplication ─────────────────────────────────────

_FIGURE_CAPTION_RE = re.compile(r"^\*\*Figure\s+(\d+)[.:]\*\*\s*(.*)", re.IGNORECASE)
_FIGURE_SUBCAPTION_RE = re.compile(r"^\(?([a-d])\)\s+")


def _dedup_figure_captions(text: str) -> str:
    """Remove raw OCR sub-caption text that duplicates a formatted Figure caption.

    Pattern: builder inserts **Figure N:** description, then also inserts
    raw OCR lines like "(a) Schematic...", "(b) Crystal..." from a separate
    text element. The raw lines are redundant.
    """
    lines = text.split("\n")
    lines_to_remove = set()

    i = 0
    while i < len(lines):
        m = _FIGURE_CAPTION_RE.match(lines[i].strip())
        if m:
            fig_num = m.group(1)
            # Found a formatted figure caption. Look ahead for:
            # 1. Image reference line (![...](figures/...))
            # 2. Raw sub-caption lines starting with (a), (b), etc.
            j = i + 1
            # Skip blank lines and image references
            while j < len(lines) and (
                not lines[j].strip() or
                lines[j].strip().startswith("![")
            ):
                j += 1
            # Now check if following lines are sub-captions
            removed_any = False
            while j < len(lines):
                stripped = lines[j].strip()
                if not stripped:
                    j += 1
                    continue
                # Check for sub-caption pattern: "(a) ...", "(b) ...", etc.
                if _FIGURE_SUBCAPTION_RE.match(stripped):
                    lines_to_remove.add(j)
                    removed_any = True
                    j += 1
                else:
                    break
            if removed_any:
                logger.debug(f"Figure caption dedup: removed raw sub-captions "
                           f"after Figure {fig_num} caption at line {i+1}")
        i += 1

    if lines_to_remove:
        logger.info(f"Figure caption dedup: removed {len(lines_to_remove)} "
                    f"raw sub-caption lines")
        lines = [l for i, l in enumerate(lines) if i not in lines_to_remove]

    return "\n".join(lines)


# ── Pass 1c: Figure Renumbering ──────────────────────────────────────────────

_FIGURE_LABEL_RE = re.compile(
    r"(\*\*Figure\s+)(\d+)([.:]\*\*)", re.IGNORECASE
)
_FIGURE_IMG_REF_RE = re.compile(
    r"(!\[[^\]]*\]\(figures/figure_E)\d+(\.png\))"
)


def _renumber_figures(text: str) -> str:
    """Renumber duplicate figure labels sequentially.

    If the builder outputs Figure 6, Figure 6, Figure 6 for three different
    figures, renumber them to Figure 1, Figure 2, Figure 3 (or continue from
    the last unique number).
    """
    lines = text.split("\n")
    # First pass: find all figure labels and detect duplicates
    fig_positions = []  # (line_idx, current_num)
    for i, line in enumerate(lines):
        m = _FIGURE_LABEL_RE.search(line)
        if m:
            fig_positions.append((i, int(m.group(2))))

    if len(fig_positions) < 2:
        return text

    # Check for duplicates
    nums = [n for _, n in fig_positions]
    if len(set(nums)) == len(nums):
        return text  # No duplicates

    # Renumber sequentially starting from 1
    changes = 0
    for seq, (line_idx, old_num) in enumerate(fig_positions, 1):
        if old_num != seq:
            lines[line_idx] = _FIGURE_LABEL_RE.sub(
                rf"\g<1>{seq}\g<3>", lines[line_idx], count=1
            )
            changes += 1

    if changes:
        logger.info(f"Figure renumbering: renumbered {changes} duplicate figure labels")

    return "\n".join(lines)


# ── Pass 2: Artifact Removal ─────────────────────────────────────────────────

def _remove_artifacts(text: str) -> str:
    """Remove common artifacts from docling and agent output."""
    lines = text.split("\n")
    cleaned = []
    removals = 0

    # Detect repeated short lines (running headers/footers appearing 3+ times)
    short_lines = [l.strip() for l in lines if 5 < len(l.strip()) < 60]
    line_counts = Counter(short_lines)
    repeated_headers = {l for l, c in line_counts.items() if c >= 3}

    seen_headers = set()
    for line in lines:
        stripped = line.strip()

        # Remove <!-- image --> placeholders
        if stripped == "<!-- image -->":
            removals += 1
            continue

        # Remove stray bare page numbers at end of sections
        if re.match(r"^\d{1,3}$", stripped) and not cleaned:
            # Only at very start — likely a page number
            removals += 1
            continue

        # Remove barcode-like artifacts (sequences of spaced digits)
        if re.match(r"^\d{1,2}(\s+\d{1,2}){3,}$", stripped):
            removals += 1
            continue

        # Remove repeated running headers (keep first occurrence)
        if stripped in repeated_headers:
            if stripped in seen_headers:
                removals += 1
                continue
            seen_headers.add(stripped)

        cleaned.append(line)

    # Remove trailing bare page numbers
    while cleaned and re.match(r"^\d{1,3}$", cleaned[-1].strip()):
        cleaned.pop()
        removals += 1

    if removals:
        logger.info(f"Artifacts: removed {removals} lines")

    return "\n".join(cleaned)


# ── Pass 3: Reference Cleanup ────────────────────────────────────────────────

def _cleanup_references(text: str) -> str:
    """Fix common reference formatting artifacts."""
    # Normalize "- - -" patterns in reference numbers (docling artifact)
    # e.g., "1508- - -13" → "1508-13"
    text = re.sub(r"(\d)- - -(\d)", r"\1-\2", text)
    # Also handle "- - -" with spaces
    text = re.sub(r"(\d)\s*-\s*-\s*-\s*(\d)", r"\1-\2", text)
    return text


# ── Pass 4: Citation Normalization ───────────────────────────────────────────

# Patterns for different citation styles
_CITE_FOOTNOTE = re.compile(r"\[\^(\d+)\](?!:)")  # [^1] not followed by : (not definition)
_CITE_SUP_HTML = re.compile(r"<sup>(\d[\d,\s]*)</sup>")  # <sup>1,2,3</sup>
_CITE_CARET = re.compile(r"\^(\d+)\^")  # ^1^
_CITE_BARE_SUP = re.compile(r"(?<=[a-z.)\]])(\d{1,3})(?=[.\s,])")  # bare trailing number (risky, skip)


def _normalize_citations(text: str) -> str:
    """Normalize inline citation style to the dominant style in the document."""
    # Count occurrences of each style
    counts = {
        "footnote": len(_CITE_FOOTNOTE.findall(text)),
        "sup_html": len(_CITE_SUP_HTML.findall(text)),
        "caret": len(_CITE_CARET.findall(text)),
    }

    total = sum(counts.values())
    if total < 2:
        return text  # Not enough citations to normalize

    dominant = max(counts, key=counts.get)
    logger.debug(f"Citation styles: {counts}, dominant: {dominant}")

    if dominant == "sup_html":
        # Convert [^N] → <sup>N</sup>
        text = _CITE_FOOTNOTE.sub(lambda m: f"<sup>{m.group(1)}</sup>", text)
        # Convert ^N^ → <sup>N</sup>
        text = _CITE_CARET.sub(lambda m: f"<sup>{m.group(1)}</sup>", text)
    elif dominant == "footnote":
        # Convert <sup>N</sup> → [^N] (split comma-separated)
        def sup_to_footnotes(m):
            nums = [n.strip() for n in m.group(1).split(",")]
            return "".join(f"[^{n}]" for n in nums)
        text = _CITE_SUP_HTML.sub(sup_to_footnotes, text)
        # Convert ^N^ → [^N]
        text = _CITE_CARET.sub(lambda m: f"[^{m.group(1)}]", text)
    elif dominant == "caret":
        # Convert [^N] → ^N^
        text = _CITE_FOOTNOTE.sub(lambda m: f"^{m.group(1)}^", text)
        # Convert <sup>N</sup> → ^N^ (split comma-separated)
        def sup_to_carets(m):
            nums = [n.strip() for n in m.group(1).split(",")]
            return "".join(f"^{n}^" for n in nums)
        text = _CITE_SUP_HTML.sub(sup_to_carets, text)

    return text


# ── Pass 5: Subscript/Superscript Normalization ─────────────────────────────

# Unicode subscript/superscript mappings
_SUBSCRIPT_MAP = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
}

_SUPERSCRIPT_MAP = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼",
    "n": "ⁿ", "i": "ⁱ",
}

_HTML_SUB = re.compile(r"<sub>([\w+\-=]+)</sub>")
_HTML_SUP_NONDIGIT = re.compile(r"<sup>([a-z+\-=]+)</sup>", re.IGNORECASE)


def _to_unicode_sub(text: str) -> str:
    """Convert text to Unicode subscript characters where possible."""
    result = []
    for ch in text:
        if ch.lower() in _SUBSCRIPT_MAP:
            result.append(_SUBSCRIPT_MAP[ch.lower()])
        else:
            return text  # Can't fully convert, return original
    return "".join(result)


def _normalize_sub_superscripts(text: str) -> str:
    """Convert HTML <sub>/<sup> to Unicode where possible.

    Skips documents with heavy LaTeX math (detected by $$ blocks)
    to avoid conflicting with inline $...$ notation.
    """
    has_display_math = "$$" in text

    # Always convert <sub>X</sub> to Unicode subscripts for simple cases
    def sub_replacer(m):
        content = m.group(1)
        converted = _to_unicode_sub(content)
        if converted != content:
            return converted
        return m.group(0)  # Can't convert, keep HTML

    text = _HTML_SUB.sub(sub_replacer, text)

    # Don't touch <sup> for citation numbers (handled by citation normalizer)
    # Only convert non-numeric superscripts like <sup>n</sup> to Unicode
    if not has_display_math:
        def sup_replacer(m):
            content = m.group(1)
            result = []
            for ch in content:
                if ch.lower() in _SUPERSCRIPT_MAP:
                    result.append(_SUPERSCRIPT_MAP[ch.lower()])
                else:
                    return m.group(0)
            return "".join(result)
        text = _HTML_SUP_NONDIGIT.sub(sup_replacer, text)

    return text


# ── Pass 5b: Equation Tag Normalization ───────────────────────────────────────

# Match (N) or \qquad (N) at the end of a line inside a $$ block
_TRAILING_EQ_NUM = re.compile(r"[,\s]*\\?qquad\s*\(\s*(\d+)\s*\)\s*$")
_TRAILING_EQ_NUM_BARE = re.compile(r"[,\s]*\(\s*(\d+)\s*\)\s*$")


def _normalize_equation_tags(text: str) -> str:
    """Convert inline equation numbers like (8) or \\qquad (8) to \\tag{8}.

    Works on $$ display equation blocks. Scans lines between $$ delimiters
    and normalizes trailing (N) patterns on the last content line.
    """
    lines = text.split("\n")
    count = 0
    in_eq = False
    eq_start = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("$$") and not in_eq:
            # Check for single-line equation: $$...$$
            if stripped.endswith("$$") and len(stripped) > 4:
                # Single-line display equation
                if r"\tag" not in stripped:
                    for pat in [_TRAILING_EQ_NUM, _TRAILING_EQ_NUM_BARE]:
                        m = pat.search(stripped[2:-2])
                        if m:
                            num = m.group(1)
                            inner = stripped[2:-2]
                            inner = pat.sub(f" \\\\tag{{{num}}}", inner)
                            lines[i] = f"$${inner}$$"
                            count += 1
                            break
            else:
                in_eq = True
                eq_start = i
        elif stripped == "$$" and in_eq:
            # End of multi-line equation — check previous non-empty line
            for j in range(i - 1, eq_start, -1):
                prev = lines[j].strip()
                if prev and r"\tag" not in prev:
                    for pat in [_TRAILING_EQ_NUM, _TRAILING_EQ_NUM_BARE]:
                        m = pat.search(prev)
                        if m:
                            num = m.group(1)
                            lines[j] = pat.sub(f" \\\\tag{{{num}}}", lines[j])
                            count += 1
                            break
                break
            in_eq = False

    if count:
        logger.info(f"Equation tags: normalized {count} inline equation numbers to \\tag{{}}")

    return "\n".join(lines)


# ── Pass 5c: Math-as-Heading Fix ──────────────────────────────────────────────

_MATH_HEADING_RE = re.compile(
    r"^\$\$\s*\\mathrm\s*\{([^}]+)\}\s*\$\$$"
)


def _fix_math_headings(text: str) -> str:
    r"""Convert display math that is actually a section heading back to markdown.

    Catches: $$\mathrm{III.~RESULTS~AND~DISCUSSION}$$
    Also catches spaced-out variants: $$\mathrm { I I I . ~ R E S U L T S }$$
    Converts to: ## III. RESULTS AND DISCUSSION
    """
    lines = text.split("\n")
    count = 0
    for i, line in enumerate(lines):
        m = _MATH_HEADING_RE.match(line.strip())
        if m:
            heading_text = m.group(1).strip()
            # Split on ~ (LaTeX word separators) first, then dejoin each part
            parts = re.split(r"~", heading_text)
            cleaned_parts = []
            for part in parts:
                part = part.replace("\\", "").strip()
                # Collapse "I I I ." to "III." — remove spaces between single chars
                collapsed = re.sub(r"(?<=\b\w) (?=\w\b)", "", part)
                # If that didn't help (e.g., "I I I ."), try removing all internal spaces
                tokens = part.split()
                if len(tokens) > 2 and all(len(t) <= 1 for t in tokens):
                    collapsed = "".join(tokens)
                cleaned_parts.append(collapsed.strip())
            heading_text = " ".join(p for p in cleaned_parts if p)
            heading_text = re.sub(r"\s+", " ", heading_text).strip()
            # Determine heading level from numbering style
            if re.match(r"^[IVX]+\.", heading_text):
                lines[i] = f"## {heading_text}"
            elif re.match(r"^[A-Z]\.", heading_text):
                lines[i] = f"### {heading_text}"
            else:
                lines[i] = f"## {heading_text}"
            count += 1
            logger.debug(f"Math heading fix: line {i+1} → {lines[i]}")

    if count:
        logger.info(f"Math heading fix: converted {count} display math blocks to headings")

    return "\n".join(lines)


# ── Pass 6: Heading Hierarchy Normalization ───────────────────────────────────

def _normalize_heading_hierarchy(text: str) -> str:
    """Ensure only the document title uses H1 (#). Demote other H1s to H2.

    The first H1 in the document is the title. Any subsequent H1 that isn't
    the title should be H2 (a section heading).
    """
    lines = text.split("\n")
    found_title = False
    changes = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match H1 (# ...) but not H2+ (## ...)
        if stripped.startswith("# ") and not stripped.startswith("## "):
            if not found_title:
                found_title = True  # First H1 = title, keep it
            else:
                lines[i] = line.replace("# ", "## ", 1)
                changes += 1
                logger.debug(f"Heading fix: demoted H1 to H2 at line {i+1}: {stripped[:50]}")

    if changes:
        logger.info(f"Heading hierarchy: demoted {changes} H1 headings to H2")

    return "\n".join(lines)


# ── Pass 6b: Remove VLM image placeholders ────────────────────────────────────

def _remove_vlm_image_placeholders(text: str) -> str:
    """Remove VLM-hallucinated image references, keep our img:N references.

    The VLM sometimes generates its own image markdown (e.g. ![Figure 1](image_url))
    while the pipeline separately adds proper ![caption](img:N) references.
    """
    lines = text.split("\n")
    result = []
    removals = 0
    for line in lines:
        stripped = line.strip()
        if re.match(r"^!\[.*?\]\((?!img:).+?\)$", stripped):
            removals += 1
            continue
        result.append(line)

    if removals:
        logger.info(f"Removed {removals} VLM-generated image placeholders")

    return "\n".join(result)


# ── Pass 7: Whitespace Normalization ─────────────────────────────────────────

def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for consistent formatting."""
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Ensure one blank line before headings (but not at start of document)
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i > 0 and line.startswith("#") and result and result[-1].strip():
            result.append("")
        result.append(line)

    # Strip trailing whitespace
    text = "\n".join(result).rstrip() + "\n"
    return text


# ── Rejoin Hyphenated Words ──────────────────────────────────────────────────

# Line ending with a letter + hyphen, next line starting with lowercase letter
_HYPHEN_BREAK_RE = re.compile(r"^(.+[a-zA-Z])-\s*$")
_PAGE_TAG_RE_PP = re.compile(r"</?page\s+\d+>")


def _rejoin_hyphenated_words(text: str) -> str:
    """Rejoin words split by PDF line-break hyphens.

    Detects lines ending with 'word-' followed by a line starting with
    a lowercase letter, and joins them: 'magneti-\\nzation' → 'magnetization'.

    Skips page boundary tags and markdown structures.
    """
    lines = text.split("\n")
    result = []
    i = 0
    rejoined = 0

    while i < len(lines):
        line = lines[i]

        # Don't touch page tags, headings, list items, table rows, code blocks
        if (i + 1 < len(lines)
                and not _PAGE_TAG_RE_PP.search(line)
                and not _PAGE_TAG_RE_PP.search(lines[i + 1])
                and not line.strip().startswith(("#", "|", "```", "- ", "* ", ">"))
                and not lines[i + 1].strip().startswith(("#", "|", "```", "- ", "* ", ">", "!["))):

            m = _HYPHEN_BREAK_RE.match(line)
            if m and lines[i + 1].strip() and lines[i + 1].strip()[0].islower():
                # Rejoin: "magneti-" + "zation ..." → "magnetization ..."
                # Don't append yet — the merged line may also end with a hyphen
                lines[i] = m.group(1) + lines[i + 1].strip()
                lines.pop(i + 1)
                rejoined += 1
                continue  # Re-check same index (merged line might chain)

        result.append(line)
        i += 1

    if rejoined:
        logger.debug(f"Rejoined {rejoined} hyphenated words")

    return "\n".join(result)
