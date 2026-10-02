# Idiosyncrasy test and resource audit

2026-10-02; follow-up to the scientific correction and the shared testing
infrastructure introduced by EE commit `9014395`. This round changes tests,
validation drivers and bounded benchmarks, not metric code or public APIs.
Reference: Lyons et al. (2020), *Nature Ecology & Evolution* 4:1685–1693,
[doi:10.1038/s41559-020-01286-y](https://doi.org/10.1038/s41559-020-01286-y).
The scientific definition and four-paper comparison remain in
[IDIOSYNCRASY_REVIEW.md](IDIOSYNCRASY_REVIEW.md).

## Literature validation

EE had already moved the three initial Lyons checks into `validation/tests/`.
The follow-up adopts its [contract](TESTING.md), verified input fixture and
`assert_case_matches`, and separates the evidence into five tests:

| Check | Case and role | What is established |
| --- | --- | --- |
| Published mean and SEM | `lyons.trna.iid.v1`, paper result | Independent replay and production kernel with author seeds both reproduce 0.612 ± 0.005 at printed precision |
| All directed mutations | Same case, independent check | All 828 identities, background counts, observed SDs and ratios agree; a matching mean alone is insufficient |
| Complete population | `lyons.trna.population.v1`, input check | 28,530 viable rows, 3,903 isolates, 828 directed mutations, unique 72-character sequences, positive finite fitness and preserved imported values |
| Public global function | `lyons.trna.iid.seed0.v1`, independent check | Public serial/parallel calls and a separate same-stream oracle agree at 0.6141679589301012 within 1e-12 |
| Released Fig. 1a procedure | `lyons.trna.fig1a.control.v1`, independent check | Seed 4033 gives index 0.5080668755924872 and control SD 0.24903011647303494; the conflict with printed 0.49/0.26 remains explicit |

The original published case and its fingerprint were not edited. Three new
cases freeze previously obtained independent results with precise paper/code
locators and existing input hashes. Neither the public seed-0 result nor the
Fig. 1a reconstruction is relabelled as a printed paper result. The latter is
a procedural check, not resolution of the source discrepancy.

The driver now returns detailed observations so that scientific assertions live
in the dedicated tests. Pure tuple/sequence-substitution oracles reside in
`validation/oracles/idiosyncrasy.py`, import no GraphFLA helper and load no files.
The extra `test_metric_details.py.template` shows how to share verified input
and keep paper replay, per-item agreement and a different public convention
separate. Default basic-test collection excludes these empirical tests.

## Basic-test audit

The dedicated module now contains 37 bounded synthetic checks. Coverage includes:

| Contract | Discriminating checks |
| --- | --- |
| Definition | Literal directed-substitution and finite-control oracle, equal mutation weighting, explicit sample size/replacement/self-pairs, exact integer-additive zero, values above one |
| Backgrounds | Original position/allele labels, reverse substitutions, missing/no shared backgrounds, threshold boundary including a defined n=2 case, invariant features and long-sequence fallback |
| Population and scale | An isolated genotype changes only the control population; positive/negative affine changes preserve the index on an unchanged population |
| Randomness | Effective seed, same-seed serial/parallel equality, global RNG state unchanged; deterministic single-mutation control through the external RNG factory |
| Undefined/error behavior | Flat, empty, one-genotype and one-position inputs; zero control SD propagates NaN; invalid threshold/labels, missing or duplicate configurations, nonfinite fitness, absent configuration metadata and unbuilt input |
| Interpretation | A nonlinear map of an additive trait can have positive I_id |

Removed three weak/redundant tests: an unseeded additive check with tolerance
0.02, a broad `0.7 < HoC index < 1.3` check that cannot distinguish the finite
control from its analytic replacement, and the duplicate same-seed-only API
test. The stronger dedicated checks replace these claims; existing golden
values were not refreshed. The fallback test now uses an independent oracle,
rather than only comparing two production branches. The single-mutation test
controls the RNG factory instead of replacing the production ratio helper.

For the six idiosyncrasy functions/helpers, basic tests cover **82/83 statements
and 45/46 branches**. The remaining branch is the final defensive return after
the single-mutation search: validated distinct observed alleles are always
enumerated by the worker, so no valid input reaches it. No impossible helper
result was manufactured merely to report 100%. DRI/ICI coverage is excluded
from these metric-specific counts.

## Execution and limits

```sh
python -m pytest tests/test_idiosyncrasy.py -q
python -m validation.tests --literature-study Lyons2020 -q --junitxml=lyons.xml
python -m validation.tests --literature-case lyons.trna.iid.seed0.v1 -q
python -m validation check --artifacts
```

The five literature tests passed in **10.68 s**, sharing one full-population
reproduction. The fixture is 450,155 compressed bytes; no downsampling or input
regeneration occurs. A 50 ms watchdog observed **1,186,660,352 bytes** of combined
worker/child RSS, below its 1,536 MiB budget; the invocation had a 60 s wall limit.
This includes GraphML adaptation, imports and two parallel workers. Shared pages
can be counted twice and brief RSS peaks can be missed. It is a resource review
of correctness work, not a timing measurement of the metric alone.

The full basic suite passed **2,048 tests**, with the same two strict xfails for
large-integer construction precision, in 36.52 s. It completed under an explicit
90 s / 3,072 MiB watchdog (sampled process-tree peak 2,781,364,224 bytes). Existing
construction/data checks still contribute to this full-suite memory footprint.
The independent research store check passed 126 artifact references and all
25 historical events. Five JUnit test records contain the Lyons DOI, evidence
role and four frozen case identities; default collection contains none of them.

An initial coverage invocation targeting the deep metric module caused an
igraph reinitialization error before collection. Using `--cov=graphfla` and
extracting the relevant function ranges resolved the instrumentation setup;
the failed logs are retained locally. No numerical tolerance was widened.
Only Python 3.13 was executed locally; changed Python files also parse as 3.9.

Performance uses separate, much smaller synthetic workloads and runs after
correctness checks have finished. See [the results](../benchmarks/IDIOSYNCRASY_RESULTS.md).
