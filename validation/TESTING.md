# Literature test contract

This is the maintenance contract for all promoted literature tests. It extends
[the immutable case and event format](README.md#case-schema-version-1); it does
not replace research history or turn regression snapshots into scientific truth.
Use the [templates](templates/) when adding a study. A passing test certifies
only the claim identified by its case, data, preprocessing and evidence role.

## Separate execution

```sh
python -m pytest                                      # basic tests only
python -m validation.tests                            # all promoted literature tests
python -m validation.tests --literature-study Wagner2023 -q
python -m validation.tests --literature-case wagner.rna.ee.general.v1 -q
python -m validation.tests --collect-only -q
python -m validation.tests --junitxml=literature.xml --durations=10
```

Install `requirements-dev.txt` and the package first. `tox -e literature` is
an equivalent full-suite entry point. No test downloads data or executes a
command obtained from a case record. Repeated study/case options form a union
within each option; when both options occur, both filters must match. Unknown
IDs are errors; a selection containing no executable tests returns pytest exit
code 5. A registered research case need not yet have an executable test.

The module entry point supplies an absolute suite path and its own pytest
configuration. Do not combine basic and literature directories in one pytest
invocation: the literature hook deliberately rejects uncited tests. Basic tests
may import small, pure equation oracles from `validation/oracles/`; they must
not run empirical reproductions or import a driver that loads data at import time.
Lightweight fixture/manifest integrity checks may remain in the basic suite.

## Evidence claims

Each test, including each parametrized case, needs
`@pytest.mark.literature_case("case.id.v1", role="...")`.

| Role | What a pass means | Required evidence |
| --- | --- | --- |
| `paper_result` | The stated published number is reproduced under a matching definition | `published_numeric`, `definition_match: yes`, precise paper/SI locator |
| `author_result` | A pinned author output or procedure is reproduced | Published/author artifact target, pinned author version; disclose differences from the paper or public API |
| `independent_check` | An independently implemented equation or aligned computation agrees | Explicit independent implementation and conventions; not a claim to reproduce a printed number |
| `input_check` | Input integrity, population or graph alignment holds | Pinned data/graph and explicit structural assertions; not metric validation |

Collection rejects missing/unknown cases, unrecognized roles, incomplete paper
citations, paper-result claims with mismatched definitions, and independent
calculations labelled as author results. These checks cannot establish that a
source supports a claim: a reviewer must read the cited passage and algorithm.
A DOI alone is insufficient scientific evidence. Cite the exact equation,
figure, table, paragraph, code cell or file and immutable commit/archive version.

For EE, author replay and the corrected general estimator are separate tests
and cases. Replaying the author's duplicated endpoint variance or RNA measurement
error procedure does not validate the public estimator's neighborhood variance.
Likewise, a matching summary count alone can hide incorrect individual labels;
compare detailed results with an independent oracle when those results exist.

Lyons illustrates a different distinction: its tRNA notebook reseeds by mutation
background count, while the public global function uses one seeded stream.
The paper mean/SEM case therefore tests the production kernel with author seeds;
a separate independent-equation case tests the public function under seed 0.
Mutation labels, counts, SDs and all 828 ratios are compared explicitly. The
Figure 1a code/text discrepancy remains an independent procedural check, not a
successful paper-number claim. See [the detailed-test template](templates/test_metric_details.py.template)
for sharing verified input while keeping these claims separate.

Gamma illustrates pooled statistics: compare every ordered-position numerator
and denominator in addition to the public ratio. Its printed-result and
independent-equation cases remain distinct even when they share a dataset.
The user accepted the remaining printed-value differences on 2026-10-02;
those unresolved cases stay recorded with their original tolerances and are
not reclassified as passing paper reproductions. The
[pooled-statistic template](templates/test_pooled_metric.py.template) generalizes
this pattern without introducing a second test framework.

## Required record before promotion

1. **Reference and claim.** Reuse a study ID from `catalog.json`, or add its
   title, year, DOI, aliases and provenance. Declare metric IDs, scope, evidence
   tier and whether definitions match. Record unresolved differences explicitly.
2. **Inputs and provenance.** Pin every test input by SHA-256 in `inputs`.
   Keep source/archive and derived hashes, upstream version, retrieval method,
   license or redistribution terms, attribution, and a deterministic conversion
   recipe in the fixture's provenance document. If redistribution is unclear,
   use an external store rather than committing the data. Acquisition is a
   separate, explicit step; the tests never download or regenerate expectations.
3. **Calculation contract.** State the input population, fitness units and
   transformations, direction of optimization, neighborhood/distance rules,
   missing and duplicate handling, filters, ties and neutral edges, orientation,
   aggregation weights and denominator. Record inference choices (variance,
   test, multiple-testing family, FDR), random generator/seed and sampling count
   when applicable. Say “not applicable” for an unused choice when ambiguous.
4. **Independent target.** Freeze `expected` and `comparison` before checking
   GraphFLA. Use exact comparison for counts/labels; derive tolerances from
   reported precision or a documented numerical/stochastic error model. Do not
   generate expected values with the production helper being tested. An oracle
   must implement the mathematics independently, not call or copy its helpers.
5. **Resources.** Record population size, input bytes, typical duration, expected
   peak memory and the command/environment in the review or fixture README.
   Use the smallest *scientifically valid* input. Subsampling a full-population
   paper target creates a different claim and requires its own case. Expensive
   cases remain opt-in; CI has a separate job with a ten-minute wall limit.
6. **Executable comparison.** Put the test in `validation/tests/test_<metric>.py`
   or a clearly scoped study module. Mark it and use `assert_case_matches` for
   the frozen summary. Check informative per-item outputs where possible.
   Use `literature_inputs` for verified paths, and share expensive computations
   with a module/session fixture or bounded cache. Keep assertions in tests,
   not only in a reproduction script or a saved report.

`literature_inputs` automatically verifies all selected case inputs before a
test executes, even for legacy tests with their own readers. Verification is
cached within one invocation. Missing, modified or escaping paths fail; they
are never silently skipped. Store-backed cases require `--literature-store`;
no developer's absolute home path is a default. Treat fixtures as read-only.
Collection and filtering do not load datasets. The entire catalog's schemas
are still checked, so a malformed record cannot hide behind a test filter.

## Failures, revisions and reports

Do not fix a failure by copying GraphFLA output into `expected`, widening a
tolerance, changing a filter, skipping missing data, or silently catching an
exception. First identify an implementation defect, definition mismatch,
source error, numerical limit or missing artifact. Record the evidence and
preserve failed runs. An unresolved known failure, if temporarily `xfail`,
requires `strict=True`, an explicit reason and a tracked follow-up; it is never
counted as a scientific pass.

Once referenced by a research event, a case is immutable. A scientific change
creates a new case ID/revision and a superseding checkpoint; keep old cases and
fingerprints loadable. Moving a test is not a scientific revision: the marker
is the current executable mapping, while old `pytest_node` values remain
historical. Do not rewrite archived cases merely to update a Python path.

JUnit test properties contain the role, full citation, locators, case
fingerprint, input hashes and comparison rule. Save the command, commit plus
uncommitted diff when relevant, Python/dependency versions, raw output and
resource observations with the report. CI uploads JUnit even on failure.
JUnit is execution evidence, not the append-only research ledger. Archive
reviewed trials through `python -m validation` using the existing event contract.

## Acceptance checklist for a new test

- The cited source supports the exact target and all definition differences
  are visible; expected values do not originate from GraphFLA.
- Raw/derived provenance, license, transformations and all hashes are recorded.
- The case passes `python -m validation check --artifacts` (add an explicit
  store for external inputs); no existing case fingerprint changes.
- Collection shows the new test under the intended study/case filter.
- The targeted scientific test and basic contract tests pass offline; an
  intentionally modified input or wrong target fails, as covered by the shared
  harness tests. The test does not silently skip or download missing inputs.
- Runtime and memory fit the reviewed workload; the literature suite is still
  absent from default `python -m pytest` collection.
