# Changelog

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
