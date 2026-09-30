# Scientific benchmark contract

This directory records what GraphFLA is being checked against and how research
can resume. It is separate from the runtime benchmarks. The scientific plan is
in [tests/VALIDATION.md](../tests/VALIDATION.md).

Three checks remain distinct: synthetic correctness, agreement with a published
definition, and reproduction of an empirical result. A passing case certifies
only its stated input, preprocessing and statistic. It does not validate a whole
paper or every metric in the package.

## Layout

| Location | Purpose |
| --- | --- |
| `catalog.json` | Stable study IDs, aliases, discovery provenance, sizes and initial queue state |
| `metrics.json` | Public-function coverage, synthetic plans and metric-specific literature gaps |
| `cases/*.json` | Reviewed, versioned definitions with independently sourced expected values |
| External `papers/<study>/` | Papers, source data, acquisition logs, method audits and exploratory runs |
| External `results/sha256/` | Immutable snapshots of results and supporting artifacts |
| External `events/` | Append-only trial and checkpoint history |

Only small, attributed, reviewed fixtures belong in the repository. Store large
downloads and exploratory code outside it. A result filename is a content hash;
rerunning a working `results.json` cannot overwrite its archived predecessor.
Keep the external store backed up with the research workspace. Content hashes
detect edited records; they do not recover deleted files or replace backups.

## Resume before researching

The current research store is
`/Users/arwen/Documents/GraphFLA-validation/2026-09-30`. Its
`SESSION_HANDOFF.txt` describes the historical import and unfinished work.
From the repository root:

```sh
python -m validation --store /Users/arwen/Documents/GraphFLA-validation/2026-09-30 check --artifacts
python -m validation --store /Users/arwen/Documents/GraphFLA-validation/2026-09-30 queue
python -m validation --store /Users/arwen/Documents/GraphFLA-validation/2026-09-30 resume Papkou2023
python -m validation --store /Users/arwen/Documents/GraphFLA-validation/2026-09-30 queue --scope metric_search
```

Global options precede the subcommand. `resume` accepts a stable ID or a unique
alias and prints the dossier, cases, checkpoints and trials. Read these and the
dossier's source index before downloading or calculating anything. Reuse verified
sources. Never regenerate the catalog to clear unresolved work.

The default queue prioritizes supplied studies with measured or theoretical
size greater than 1,024. Use `--min-variants 0` to include smaller and unknown-size
entries, or `--include-closed` to inspect completed triage. A `triaged` study may
have unresolved metrics; the checkpoint explains what would justify further work.
Metric-driven discovery is a separate queue and is not limited to supplied papers.

Without `--store`, `check` verifies repository definitions and fixtures only;
its output explicitly reports `store: null`. It does not inspect research history.

## Add or extend a study

1. Reuse the existing study ID if the DOI or alias is already present. Correct
   citation metadata without changing its ID. Add a new catalog entry only for
   a new publication; conditions from one paper share its ID.
2. Create the external dossier. Preserve `metadata.json`, `sources/index.json`,
   `evidence.json` and a readable `report.txt`. Source entries identify URL,
   version, retrieval date, license, local path and SHA-256. Log failed acquisition
   attempts so future sessions do not repeat them without a reason.
3. Extract a candidate claim with an exact page, figure, equation or artifact
   locator. Record fitness scale, neighborhood, filters, missing-data treatment,
   imputation, tie handling, denominator and uncertainty. The supplied old
   GraphFLA table and new GraphFLA output are never expected values.
4. Freeze a case only when the numerical target and comparison are meaningful.
   Otherwise append a checkpoint with the missing prerequisite. A triage-only
   dossier needs no dummy `reproduction.py`.
5. Run a bounded reproduction in the external workspace using an identified
   source snapshot. Save the command, environment, seed, source/input hashes,
   observed output, errors and elapsed time. Preserve failed attempts.
6. Have the lead reviewer assess the definition, provenance and result before
   adding a deterministic fixture/test. Tests use the same case definitions;
   expected values and tolerances are not duplicated in test code.

## Case schema, version 1

Every record has integer `schema_version: 1` and a `kind`. A case requires:

| Fields | Contract |
| --- | --- |
| `id`, `revision`, `study_id`, `metric_ids` | Stable identity, positive revision and known study/metric IDs |
| `evidence_tier` | `published_numeric`, `author_artifact_numeric` or `independent_equation` |
| `definition_match` | `yes`, `no` or `unresolved` |
| `scope`, `preprocessing` | Exact population, statistic and input transformations |
| `sources` | Nonempty list of HTTP(S) URLs and precise locators for the target |
| `expected` | Frozen scalar, list or mapping; finite JSON, no missing target |
| `comparison` | `kind` (`exact` or `absolute`), nonnegative `tolerance`, and scientific `justification` |
| `inputs` | Nonempty list of `root` (`repo` or `store`), relative `path` and `sha256` |
| `pytest_node` | Test entry point when promoted; optional during research |

Use zero tolerance for exact counts. For rounded paper values, justify tolerance
from publication precision. For floating-point comparisons, justify it from the
numerical method. Do not tune tolerances, transformations or filters to obtain a
match. Conditions with different definitions need separate cases.

Once a trial references a case, preserve that case file. A correction creates a
new ID/revision alongside it; old events must still load against their original
fingerprint. Record the correction and superseded case in a new checkpoint.
Unknown schema versions require an explicit migration, never a reset.

## Record trials and checkpoints

All events require `schema_version`, `kind: event`, `type`, `study_id`, UTC
`recorded_at`, `actor` and `reason`. The timestamp is when the event was recorded;
historical imports must say so and must not invent an earlier execution time.

A checkpoint requires `state`, `last_completed_step` and `next_action`. Closed
states also require `reopen_when`. It may include a relative `dossier` path and
hashed `artifacts`. Use `blocked_access`, `blocked_artifact` or `blocked_method`
for missing prerequisites; `closed_low_yield` for a deliberate stopping decision;
`closed_no_overlap` only after checking relevant definitions. `triaged` records
completed review without claiming that every metric passed.

A trial additionally requires `case_id`, `case_fingerprint`, `implementation`
(`graphfla`, `independent_equation` or `author_code`),
`implementation_fingerprint`, `environment_fingerprint`, `outcome`, `observed`
and `result_artifact`. Compute fingerprints with `validation.contract.fingerprint`;
document the source/environment records that were hashed in the result.

The archived result is JSON with `schema_version: 1`, `kind: result`, the same
`case_fingerprint` and `observed`, plus a nonempty `artifacts` list of supporting
snapshots. Include the raw run output, evidence, driver and environment manifest.
Additional provenance and interpretation fields are allowed. The recorder checks
the result observation against the event and verifies all referenced bytes.
It cannot determine whether an author or analyst made a scientifically sound
measurement; that still requires review.

Use `archive-result PATH` to snapshot any supporting file, then the result JSON.
It returns a `{path, sha256}` reference. `record EVENT.json` validates and appends
the event. Both commands require `--store`. The Python equivalents are
`store_result(store, path)` and `append_event(store, event, studies, cases)`.
They never execute stored commands. Repeating an identical event is idempotent.

Successful outcomes are `reproduced_exact` and `reproduced_with_precision`; both
require matching definitions and observations. Exact means zero numerical error.
Other outcomes include `independent_crosscheck`, `definition_mismatch`,
`mismatch_unresolved`, `suspected_graphfla_defect`, `blocked_missing_artifact`,
`blocked_missing_method`, `access_blocked`, `no_overlapping_metric` and
`not_attempted`. A successful independent implementation is not a GraphFLA pass;
an author-artifact match is not a printed-paper target.

## Review and verification

```sh
python -m pytest -q tests/test_validation_contract.py tests/test_literature.py
python -m ruff check validation tests/test_validation_contract.py tests/test_literature.py
```

Contract tests cover revision continuity, immutable snapshots, concurrent
idempotent writes, observation/result mismatches, malformed records, edited
history, artifact hashes and path containment. Scientific tests cover the
separate published cases and synthetic oracles. No test requires network access.
Run the complete suite after integrating a reviewed tranche. Keep known failures
explicit; they are not passing correctness evidence.
