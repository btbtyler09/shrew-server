# v0.5 page JSON → markdown

`app.compose.compose_markdown(v05_json, *, page=1, page_tags=False, image_page=True) -> str`
(`app/compose.py`) is the page markdown that `/v1/convert` ships for one page of v0.5 output. It
is a pure function (no I/O, no model call) and runs the server's own code (`assemble_document` +
`synthesize_markdown`), so a scorer that imports it composes byte-for-byte what the server returns.

## Rules (in output order, per page)

1. **Chunks, in the model's emitted order.** The model's `semantic_chunks` order is the reading order.
   The 5-key output carries no other within-page ordering. Each chunk renders as:
   - `## {title}` when `title` is non-empty. This is the model's chunk title, which may be descriptive
     rather than printed text.
   - then `content` verbatim. That covers paragraphs, the content's own markdown headings, lists, and
     formulas as LaTeX (`$…$` / `$$…$$`), all exactly as the model wrote them.
2. **Figure placement.** A chunk whose content or title matches a figure's caption (≥ 0.9 similarity)
   is replaced by that figure's image ref `![caption](img:N)` at that position. The caption text is
   not repeated.
3. **Tables, in emitted order, after all chunks.** Each is the `html` verbatim. The table `caption` is
   not added, because it usually already appears as a chunk.
4. **Remaining figures, in emitted order.** On an image page with a usable box (envelope ≥ 1.5% of the
   page) the figure renders as `![caption](img:N)`. Otherwise, and always for text input, it renders
   as `[Figure: caption]` (`untitled` when there is no caption).
   - `N` is the figure's 1-based index among the page's figures.
   - Figure `description` and `chart_data` do not enter the markdown; they are in structured JSON only.
5. Blocks are joined with one blank line. `page_tags=True` keeps the `<page N>…</page N>` wrapper the
   server emits.

## What is NOT dropped

- Nothing is filtered by `section_type`. `header`, `footer` and `masthead` chunks (running heads, page
  numbers) are kept, in emitted order.
- Captions are dropped only when rule 2 places their figure.

## Not applied here (document-level server passes)

- `figure_check`: an `artifact` verdict removes that figure, and its image ref, from the document.
- `heading_fix`: re-levels `#` lines across the whole document.
- LaTeX wrap (`SHREW_LATEX_WRAP`, on by default): bare LaTeX in chunk text gets math delimiters before
  the server builds its markdown, so a page with bare LaTeX differs from this function's output by the
  inserted `$`.
- Cross-page chunk and table stitching.
- Fidelity identifier corrections (PDF text-layer only).

For a per-page benchmark these do not apply, apart from figure_check removing a figure.

## Tolerance

- Missing or ill-typed `semantic_chunks` / `figures` / `tables` are treated as empty.
- A flat 4-number box becomes a quad; a malformed `chart_data` is dropped. These are the same repairs
  the server's gate makes.
- The input dict is not modified.
