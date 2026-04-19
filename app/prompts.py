"""Prompt templates for the Shrew pipeline."""

DIRECT_CONVERT_PROMPT = """\
Transcribe this document page image to clean markdown.

Rules:
- If the page has two columns, read the left column completely first, then the right column.
- Headings: # for document title, ## for sections, ### for subsections. Determine level from visual weight.
- Separate paragraphs with blank lines.
- Preserve all text precisely — do not rephrase or omit anything.
- Filler characters: runs of 3+ identical characters used as visual layout (dot leaders `. . . .`, dashes `-----`, underscores `_____`) carry no meaning. NEVER transcribe them. Use markdown table structure or whitespace to convey the relationship between elements they were padding. This rule does NOT apply to em/en dashes (— –), single hyphens in compound words, underscores in code/identifiers, or markdown horizontal rules used as section breaks.
- Math: Use LaTeX for ALL math. $...$ for inline, $$...$$ for display equations. Use standard LaTeX: \\alpha, \\beta, \\epsilon for Greek letters. Subscripts: $x_{i}$, superscripts: $x^{2}$. Fractions: $\\frac{a}{b}$. Matrices: $\\begin{pmatrix}...\\end{pmatrix}$. Display equations with numbers: $$E = mc^2 \\tag{1}$$. No tags on inline math. ALL subscripts and superscripts in LaTeX, even in prose: $J_n$, $a_0$, $e^2$, $10^{-6}$. Preserve ALL symbols, operators, and notation precisely.
- Chemical formulas: use Unicode subscripts (H₂O, CO₂).
- Inline citations: use superscript numbers: <sup>1</sup>.
- Author affiliations: superscript symbols (*, †, ‡, §).
- Tables: ALWAYS use markdown pipe tables, NEVER LaTeX tabular. Preserve all cell values precisely as shown. Use `| Header 1 | Header 2 |` format with separator row `|---|---|`. Bold row headers if visually distinct: `| **Row Header** | value |`. For multi-line cell values, use `<br>` for line breaks within cells. Use alignment markers if needed: `:---` left, `:---:` center, `---:` right.
- Tables of contents and indexes: render as a markdown table with appropriate columns (e.g. Section ID, Title, Page). Do NOT transcribe dot leaders, dashes, or fill characters between entries — use the table structure instead.
- Figures: If the page contains a figure (chart, plot, diagram, photograph), describe it concisely (2-4 sentences). For charts/plots: describe axes, trends, key data points. For diagrams: describe components and relationships. Include the figure caption if visible.
- Text fidelity: Preserve all text precisely — do not rephrase or omit anything. Fix obvious OCR-style errors (l vs 1, O vs 0) only when clearly wrong. Mark truly illegible portions with [illegible]. Do NOT fabricate content.
- Lists: proper markdown syntax (-, 1., etc.).
- Skip page headers/footers (page numbers, running heads, repeated journal headers).
- If a sentence is cut off at the page boundary, include it as-is (the next page continues it).

Return ONLY the markdown content, no commentary.\
"""

FIGURE_CLASSIFY_PROMPT = """\
You classify cropped regions from scanned document pages.

Given an image crop, respond with exactly one word: **figure** or **artifact**.

- **figure**: A meaningful visual — chart, graph, plot, diagram, photograph, schematic, map, illustration, or any image the reader needs to understand the document.
- **artifact**: A non-content element — stamp, logo, barcode, QR code, watermark, page decoration, institutional seal, letterhead graphic, or scanner noise.

Respond with a single word: figure or artifact.\
"""
