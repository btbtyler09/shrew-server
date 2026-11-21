# Changelog

## v0.5.3 — unreleased

Uploaded documents stay in memory only (README "Data handling").

- All temp files (the upload, the multipart buffer, rendered pages, LibreOffice output and
  profile, crops, output dir) go to `SHREW_SCRATCH_DIR` (default `/dev/shm/shrew`).
  `tempfile` and `TMPDIR` are redirected process-wide, so libraries and child processes
  follow.
- The server refuses to start unless the scratch dir is tmpfs/ramfs.
  `SHREW_ALLOW_DISK_SCRATCH=1` overrides this. The same rule covers `SHREW_RAW_DUMP_DIR`.
  `/health` reports a `scratch` object.
- A full scratch dir fails that request with HTTP 507 (`reason: scratch_full`), never a disk
  fallback.
- LibreOffice runs with `HOME`/`XDG_*` in its per-call profile. Font caches are warmed at
  image build.
- Compose: read-only root filesystem, tmpfs `/scratch` (sized from measured peaks) and `/tmp`.
- Run telemetry and VLM-slot leases stay in the system temp dir (no document content).

## v0.5.2 — 2026-11-14

Security update: dependency pins only. No code, API or output changes:
the same documents send byte-identical requests to the model and return the
same responses as 0.5.1 (checked on 8 PDFs, 54 pages, with a recording mock).

**Python packages (image and `pip install`):**
- `pip` (image tooling) 26.2.2 → **26.3**. pip's private copies of
  `urllib3` (2.7.0 → 2.8.0; 2 high, 1 medium) and `msgpack` (1.1.2 → 1.2.4;
  2 high) are fixed in this release. The installed `urllib3` was already 2.8.0.
- `soupsieve` 2.10 → **4.0.1** (2 high, 1 medium; 4.0 is the first release
  that fixes both highs). `beautifulsoup4` 4.15.0 supports it unchanged.
- **Alpine / minimal installs:** upgrade pip in the venv too
  (`pip install pip==26.3`, now in the README recipe) — distribution pip
  packages carry the vulnerable copies (Alpine 3.21 ships pip 25.0.1).

**Base image:** `python:3.12.15-slim-trixie` → `python:3.12.16-slim-trixie`,
rebuilt so `apt-get upgrade` picks up Debian's fixes for `apt` / `libapt-pkg`
(1 high) and `zlib1g` (2 high). Every remaining Debian finding has no fix
published yet.

**Known, not fixable by an upgrade yet:** pip 26.3 still vendors a copy of
`pkg_resources` from `setuptools` 70.3.0, which scanners report as setuptools
70.3.0 (1 high, 1 medium, 1 low). The high (CVE-2025-47273) is in
setuptools' package-index module, which pip does not ship. The installed
`setuptools` is 84.1.0. No pip release vendors a newer copy.

| Image scan (trivy) | v0.5.1 | v0.5.2 |
|---|---|---|
| Python packages | 11 (7 high), all fixable | 3 pip-vendored setuptools (1 high), no fix released |
| Debian packages | 674 (1 critical, 141 high), 27 fixable | 647 (1 critical, 137 high), 0 fixable |

## v0.5.1 — 2026-11-14

Fix: blank pages no longer fail under the v0.5 contract.

- On a blank page the first pass could hallucinate until the loop guard
  stopped it; the retry then correctly returned almost nothing, and the
  coercion-density gate (< 300 content chars and < 2% of the first pass)
  failed that correct answer as slop. Under `v0.5` the gate now measures the
  page's ink on the page render — the fraction of pixels darker than their
  local background, so paper tint, yellowing and shading do not count — and
  accepts the short retry as `ok_coerced` when ink is below
  `SHREW_BLANK_INK_MAX` (default `0.0048`). Pages with ink keep the gate.
- The bound is measured, not chosen: across OHR-Bench (8,561 pages) and
  OmniDocBench (1,651 pages), no page carrying 300 or more characters of real
  text is below 0.0048, while blank tinted, aged and hole-punched pages reach
  0.0044. Replaying a saved OHR-Bench run of the v0.5 model: 42 failed pages
  become 39 (three blank pages); OmniDocBench results are unchanged.
- New, additive: `processing_log.pages[].near_blank` (`{ink, max,
  content_chars}`, `null` unless the exemption applied).
- `v0.4` is unchanged. No new dependency (the measurement uses OpenCV and
  NumPy, already required). No configuration change is needed; an
  out-of-range `SHREW_BLANK_INK_MAX` stops the server at startup.

## v0.5.0 — 2026-11-14

Server for the **shrew-ocr-preview v0.5** model. The server now speaks two
model contracts, `v0.5` and `v0.4`, named after the model's release line.

**Compatibility: v0.5.0 defaults to the v0.5 model contract; set
`SHREW_CONTRACT=v0.4` for v0.4/v0.4.1 weights.**

**Upgrade notes — read before upgrading.**
- **The default contract is `v0.5`.** If you serve a v0.4 or v0.4.1 model,
  set `SHREW_CONTRACT=v0.4` (or send the form field `contract=v0.4`). Under
  `v0.4` the requests sent to the model, the response shape and `/health` are
  byte-identical to 0.4.1.
- **`bbox` is four corners under `v0.5`** (breaking for clients that read
  boxes): `tables[].bbox` and `images[].bbox` are
  `[[x,y],[x,y],[x,y],[x,y]]` (top-left, top-right, bottom-right, bottom-left
  of the element's own reading frame, 0-1000 page coordinates) in place of
  `[x0, y0, x1, y1]`. There is no second box field.
- Serve the v0.5 model with `--max-model-len 32768`; `/health` is `degraded`
  below 30,000.
- Under `v0.5` the summary modes `model` and `summary_of_summaries` run as
  `contract_sos`.
- Docker Compose: `SHREW_CONTRACT`, `SHREW_FIGURE_CHECK`, `SHREW_HEADING_FIX`
  and `SHREW_LATEX_WRAP` are now passed from `docker/.env` to the server.
- The bundled llama.cpp stack (`docker-compose.ocr.yml`) now downloads the
  v0.5 GGUF files (revision `v0.5`) and runs the `v0.5` contract. Rebuild it
  (`--build`) to get the new model. llama.cpp reports no token usage while
  streaming, so there the runaway cut is the 60,000-character cap.
- A malformed `SHREW_CONTRACT`, `SHREW_LATEX_WRAP` or
  `SHREW_LATEX_WRAP_MIXED` stops the server at startup.
- New dependency: `tokenizers`.
- The README has a minimal (no PyTorch) install and an Alpine / musl recipe;
  the default pipeline does not need the `figures` extra.

**v0.5 contract.**
- Prompts are commands: the system turn is the bare command
  (`structured_extraction` for pages and text, `classify_figure` for figure
  crops, `fix_headings` for the heading outline). The user turn is the image
  alone, or one text part. Image preprocessing is unchanged.
- Decode is the 0.4.0 recipe from the same knobs (first pass 0.4 /
  presence penalty 0.3, un-enforced retry, fixed seed policy).
- New output: `tables[].description`; `images[].chart_data`
  (`{header, rows, values?, precision?}`) when the model emits one;
  `processing_log.pages[]` with one record per page (status, bucket, model
  calls, loop flags, and a per-call `trail`).
- Document passes, on by default: **figure check** (crops the model classifies
  as page artifacts are dropped), **heading fix** (heading levels repaired
  over the document outline, accepted only when no heading text changes) and
  **LaTeX wrap** (bare LaTeX gets math delimiters; nothing else is edited).
  Switches: `SHREW_FIGURE_CHECK`, `SHREW_HEADING_FIX`, `SHREW_LATEX_WRAP`, and
  the form fields `figure_check` / `heading_fix`.
- `app.compose.compose_markdown(page_json)` is the server's page markdown as a
  pure function, for external scorers (spec: `app/compose.md`).

**Token budget (v0.5).** Every request satisfies prompt + image + `max_tokens`
<= 30,000, the budget the model was trained under.
- Image tokens are computed exactly from the size of the image that is sent;
  an unreadable size is charged the maximum (7,152).
- Prompt tokens are counted with the tokenizer the serving stack uses
  (vendored and hash-checked), including the chat template's handling of its
  task tags and of plain-string text content. If the tokenizer cannot be
  loaded the count falls back to UTF-8 bytes, an upper bound.
- Text pages are bounded at 3,500 tokens as well as 11,000 characters, so
  dense scripts (CJK) no longer overrun. A line over the bound is split at
  sentence ends, then whitespace, then token boundaries; it is never refused
  or truncated. Embedded base64 `data:` payloads are replaced by
  `[payload removed]` (`SHREW_V05_STRIP_DATA_URIS=0` keeps them).
  `processing_log.text_pagination` reports pages, split lines and removed
  payloads.
- `/health` adds `budget_v05`.

**Loop guard (v0.5).** The 0.4 guard (2,000-character window) stays; v0.5 adds
content-aware guards and takes the runaway cut in output tokens.
- Guards: a repeating block (`periodic`; numbered rows that only count up
  get a longer allowance), repeated lines (`repeated`), and short numeric
  cycles (`numeric`). A dense table that is correct is no longer cut: the guards
  were accepted by replaying 41,441 saved model outputs, with 0 of 152
  hand-checked correct long pages aborted.
- Runaway cut in output tokens per tile bucket, from the longest outputs the
  model was trained on plus 10% (B0 6,500 / B1 25,500 / B2 29,500 / B3 26,500 /
  text 7,000), read from the token usage the model server reports on the
  stream. Without usage, a 60,000-character cap applies.
- Every abort records what fired, the score, the threshold and the last 2,000
  characters in `processing_log.pages[].trail[].abort_evidence`.
- A page is judged degenerate only when both the raw output and the output
  with markup stripped compress above the gate, so HTML tables do not trip it.
- `/health` adds `loop_guard_v05`, including which cut the streams actually
  got. `scripts/loopguard_replay.py` replays the guard over saved outputs.

**Known limits.**
- A model server that rejects `stream_options` with HTTP 400 fails pages until
  `SHREW_V05_STREAM_USAGE=0` is set. vLLM accepts the option.
- A text page that literally contains one of the model chat template's task
  tags (for example `<tables_otsl>`) is replaced by that task's prompt inside
  the template. The server counts it correctly and logs a warning.
- A loop that changes a number on every repeat can run to the output cut.

## v0.4.1 — 2026-10-25

Security release: dependency and base-image updates only. No code, API or
decode-recipe changes; `/v1/convert` output is unchanged (verified on PNG with
alpha + EXIF, LZW TIFF with alpha + EXIF, EXIF-rotated JPEG, palette PNG with
transparency, a 2-page PDF and a DOCX — byte-identical page images and
responses against v0.4.0).

**Python packages (image and `pip install`):**
- `pillow` 12.3.0 → **13.0.0** (41 advisories, 5 critical, 23 high). Pillow
  13.0.0 is the only release carrying these fixes. The package floor is now
  `Pillow>=13.0.0`; nothing shrew uses was removed or changed in 13.0.
- `certifi` 2026.7.22 → 2026.10.19 (1 critical)
- `filelock` 4.0.4 → 4.2.1 (1 high, 2 low)
- `setuptools` 84.0.0 → 84.1.0 (1 low)
- `pip` (image tooling) 25.0.1 → 26.2.2 (5 medium, 1 low; 26.2.2 also
  re-vendors a fixed certifi)

**Base image:** `python:3.12-slim` (Debian 13.6, Python 3.12.13) →
`python:3.12.15-slim-trixie` (Debian 13.7) plus `apt-get upgrade` at build
time. Clears 491 OS-package CVEs (4 critical, 94 high, 356 medium, 34 low,
3 unknown); every remaining Debian finding has no fix published yet.

**Reproducible image deps:** `docker/requirements.lock` pins every Python
package in the server image (installed as a pip constraints file), so a
rebuild no longer floats to whatever is newest on PyPI. Only the packages
above changed versus the v0.4.0 image.

**Known, not fixable by an upgrade yet:** pip 26.2.2 vendors `urllib3` 2.7.0
(2 high, 1 medium), `msgpack` 1.1.2 (1 high) and a `setuptools` 70.3.0 copy
(1 high, 1 medium, 1 low). These are pip's private copies, used only when
running `pip` inside the container; no pip release ships fixed versions yet.

| Image scan (trivy) | v0.4.0 | v0.4.1 |
|---|---|---|
| Python packages | 52 (6 critical, 24 high) | 7 pip-vendored (0 critical, 4 high) |
| Debian packages | 1002 (5 critical, 166 high), 491 fixable | 511 (1 critical, 72 high), 0 fixable |
| pip-audit / osv-scanner on installed packages | 41 / 46 advisory records | 0 / 0 |

### Python advisories fixed

| Package | v0.4.0 | v0.4.1 | CVE | GHSA | Severity |
|---|---|---|---|---|---|
| certifi | 2026.7.22 | 2026.10.19 | — | GHSA-6q48-6c7w-9gw2 | CRITICAL |
| filelock | 4.0.4 | 4.2.1 | CVE-2026-112940 | GHSA-v4x8-3wr5-3wq8 | HIGH |
| filelock | 4.0.4 | 4.2.1 | CVE-2026-112907 | GHSA-567q-5qvw-pf92 | LOW |
| filelock | 4.0.4 | 4.2.1 | CVE-2026-112913 | GHSA-9vhv-jxww-xfm5 | LOW |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112393 | GHSA-332p-7hrc-92wf | CRITICAL |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112398 | GHSA-5x8w-8jj5-q52h | CRITICAL |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112403 | GHSA-8g5g-8vmr-9782 | CRITICAL |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112900 | GHSA-w889-685w-c8fh | CRITICAL |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96639 | GHSA-v9pv-5gc8-w2m6 | CRITICAL |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112390 | GHSA-22hg-r758-rrw9 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112392 | GHSA-27h8-gf7j-v5hc | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112397 | GHSA-58r4-6rv3-gfh8 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112399 | GHSA-5xm2-6676-qcp9 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112400 | GHSA-65qm-jvmh-6pqg | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112402 | GHSA-7mwm-ghpf-9vrm | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112404 | GHSA-c6hw-5grf-ff96 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112405 | GHSA-cp9v-662q-vvx9 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112406 | GHSA-f2g6-542v-9j58 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112892 | GHSA-gg2r-wvfm-q9rx | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112893 | GHSA-j4m9-6w36-pw47 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112894 | GHSA-jg86-v3jv-hrpm | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112895 | GHSA-p4wg-p73j-c528 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112896 | GHSA-pchr-8j38-3x4p | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112897 | GHSA-q45j-58j2-45rg | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112902 | GHSA-xvh5-g6gv-559v | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96637 | GHSA-g842-9rc4-69pw | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96640 | GHSA-mrfp-p49p-3hf2 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96642 | GHSA-3j72-p6qg-6xp7 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96643 | GHSA-g4qv-2jxx-769h | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96644 | GHSA-9v9v-9qvf-h2hj | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96778 | GHSA-g7cr-3p24-7wf7 | HIGH |
| pillow | 12.3.0 | 13.0.0 | — | GHSA-78g4-xqwv-q3g8 | HIGH |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112391 | GHSA-254f-3936-288h | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112394 | GHSA-4558-3h8r-v8xp | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112395 | GHSA-462x-pgg6-55fp | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112396 | GHSA-4rr5-89j8-mwwm | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112401 | GHSA-74r2-7ff9-mjj2 | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112891 | GHSA-fc9g-5qg7-mf45 | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112898 | GHSA-rfgq-q3w2-pmv2 | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112899 | GHSA-vx6w-vv9f-6gvg | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-112901 | GHSA-xgh8-jp6q-p5x8 | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-114501 | GHSA-654x-cwwv-5j9c | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96636 | GHSA-p8hx-v2p6-rp84 | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96638 | GHSA-24rc-8hj2-v98c | MEDIUM |
| pillow | 12.3.0 | 13.0.0 | CVE-2026-96641 | GHSA-4rm8-3hmm-39fg | MEDIUM |
| pip | 25.0.1 | 26.2.0 | CVE-2025-8869 | — | MEDIUM |
| pip | 25.0.1 | 26.2.0 | CVE-2026-13346 | — | MEDIUM |
| pip | 25.0.1 | 26.2.0 | CVE-2026-3219 | — | MEDIUM |
| pip | 25.0.1 | 26.2.0 | CVE-2026-6357 | — | MEDIUM |
| pip | 25.0.1 | 26.2.0 | CVE-2026-8643 | — | MEDIUM |
| pip | 25.0.1 | 26.2.0 | CVE-2026-1703 | — | LOW |
| setuptools | 84.0.0 | 84.1.0 | — | GHSA-xmj3-9gf7-h52w | LOW |



## v0.4.0 — 2026-09-27

Ships alongside **shrew-ocr-preview v0.4**. shrew-server ≥ 0.4.0 is required to
serve v0.4 with its tested decode recipe; v0.3 models keep working (see
"Restoring the v0.3.x ladder" below).

**New default decode recipe (structured mode).** Two model attempts per page,
then the fallback client when one is configured:
- try 1 samples at temperature 0.4, presence penalty 0.3, with a seed derived
  from the page input (`SHREW_SEED_POLICY=fixed`, so reruns reproduce);
- on any failure (parse, schema, repetition abort) try 2 runs the same
  un-enforced contract at 0.4 / 0.3 with a fresh seed;
- no same-config re-roll; a page that fails both goes to the fallback model if
  `FALLBACK_VLM_URL` is set, and otherwise ships its page image as before.

New defaults: `SHREW_TRY1_TEMP=0.4`, `SHREW_TRY2_TEMP=0.4`,
`SHREW_RETRY_PP=0.3`, `SHREW_TRY2_ENFORCE=0`, `SHREW_ABORT_REROLL=0`
(`SHREW_TRY1_PP=0.3` and `SHREW_SEED_POLICY=fixed` unchanged). Every
environment variable and every per-request override (`temperature`,
`retry_temperature`, `seed_policy`, `retry_enforce`) still works.

Measured on a document-retrieval benchmark, same server build for every arm:
shrew-ocr-preview v0.4 with this recipe reaches hit@5 0.908 vs 0.884 for v0.3
at its shipped defaults, with no regression on any query type and failed
pages 0.32 % vs 1.00 %.

**Restoring the v0.3.x ladder** (greedy first pass, one re-roll on a loop,
greedy enforced retry at presence penalty 0.6):

    SHREW_TRY1_TEMP=0 SHREW_TRY2_TEMP=0 SHREW_TRY2_ENFORCE=1 SHREW_ABORT_REROLL=1 SHREW_RETRY_PP=0.6

**Decode fields in the response.** `processing_log.decode` records the whole
recipe (both temperatures, seed policy, both presence penalties, whether the
retry was enforced) plus per-document `rescued_by` counts, an `attempts`
histogram and total `model_calls`. Each page records `rescued_by` (`none`,
`reroll`, `retry_enforced`, `retry_unenforced`, `fallback`) and `model_calls`.

**Also in this release.**
- Stop-at-JSON-close guard: if the model keeps generating after the top-level
  JSON object has closed, the stream is stopped, the text is trimmed at the
  closing brace and the page records `finish_reason` `json_close`; the gates
  report `json_close_stops` (normally 0). `SHREW_JSON_CLOSE_GUARD=0` disables it.
- Bounded document summary (`SHREW_DOC_SUMMARY_MAX_CHARS`, default 4000) with
  selectable modes (`SHREW_DOC_SUMMARY_MODE`: `deterministic`, `contract_sos`,
  …) and per-request `summary_mode` / `summary_max_chars`.
- Guard warnings are prefixed with the document id and page.
