# Changelog

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
