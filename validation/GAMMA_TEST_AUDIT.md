# Gamma test and resource audit

2026-10-02. Builds on the scientific review in [GAMMA_REVIEW.md](GAMMA_REVIEW.md)
and the EE testing infrastructure from `9014395`. The user accepts the existing
empirical discrepancies; this does not change frozen targets or their provenance.
This round improves tests and implementation performance without changing API
signatures, sign tolerance, input population or scientific pooling.

## Literature suite and reusable contract

`python -m validation.tests --literature-study Ferretti2016` now runs **nine**
dedicated tests, separate from default `python -m pytest`. The existing
[contract](TESTING.md), citation markers, hash-verified inputs, case filters,
JUnit metadata and independent CI job are reused. All 26 existing case files
are byte-identical to the baseline; no new or enlarged empirical fixture is needed.

| Tests | Evidence role and assertion |
| --- | --- |
| csI printed gamma | Paper result: Figure 4c gamma=0.33 at printed precision |
| Three complete-data equations | Independent checks: public gamma/gamma-star, all 20 ordered position-pair numerators/denominators for each input, and the complete-cube distance-correlation identity |
| csI and TEM zero-sign denominators | Independent checks: exact sign sums 168/624 and 324/528; parallel TEM sign result agrees |
| Three population checks | Input checks: full 32×5 binary cubes, finite fitness, no genotype loss, 80 unordered neighbor pairs |

Each input is calculated once even when multiple cases share it. Six literature
checks already existed; this follow-up adds population assertions and the audit,
rather than creating another framework. The new
[pooled-statistic template](templates/test_pooled_metric.py.template) generalizes
group-contribution checks and shared inputs. It requires an independent source
for expectations and explicit orientation/multiplicity conversion; matching a
pooled scalar alone can hide compensating group errors.

Primary reference: Ferretti et al. (2016), *Journal of Theoretical Biology*
396:132–143, [doi:10.1016/j.jtbi.2016.01.037](https://doi.org/10.1016/j.jtbi.2016.01.037),
Eqs. (1), (2), (10)–(12), Figure 4c. Source/fixture provenance and other supplied
papers remain in the scientific review. The two unresolved Figure 4 case files
are still research records, not passing paper-result tests.

## Basic-test audit

The dedicated gamma module contains **42** bounded synthetic tests; the existing
three canonical gamma-star square anchors remain in `test_metrics.py`. Coverage:

| Contract | Discriminating evidence |
| --- | --- |
| Equations and weights | Independent directed substitutions on complete/sparse bi- and multiallelic inputs; heterogeneous alphabet sizes; exact pure-Walsh-order anchors; all 81 three-level squares |
| Ties and scope | Exact zero signs and denominators; no-square and flat NaNs; four observations that still form no square; graph epsilon and ordinal graph steps do not redefine allele comparisons |
| Numerical behavior | Positive/negative extreme units, smallest positive float64, overflowing finite-endpoint differences, multiallelic overflow, tiny numerators, unrelated outliers and large neutral squares |
| Indexing and population | Row/variable/allele relabeling invariance; affine fitness transforms; maximize/minimize; packed and dictionary paths; sparse high-cardinality groups and forced dense-grid cap |
| Public contract | Unbuilt/malformed inputs, one-variable warning/NaN, invalid worker count, Python float return, profile alignment and serial/parallel agreement |

Removed the broad random `abs(gamma(HoC)) < 0.5` assertion, which can pass an
incorrect zero implementation, and a redundant additive test superseded by the
exact spectral anchors. Consolidated the old high-dimensional test with its
extreme-scale counterpart. Moved eight existing oracle checks into the dedicated
module and removed their duplicate oracle implementation, retaining the independent
directed-enumeration oracle under `validation/oracles/`. No golden output was
refreshed to accommodate the optimized implementation.

Basic checks cover **145/145 implementation statements and 55/56 branches**.
The remaining ordered-only private-worker branch is exercised by the independent
literature contribution checks. Combined coverage from the separately executed
suites is **145/145 statements and 56/56 branches**. These percentages are
supporting evidence, not a substitute for the equation and boundary assertions.

Benchmark harness tests also verify gamma workload bounds/ASV registration and
that trusted baseline modules produce the scalar snapshots actually compared.
The existing shared tests protect altered outputs, NaNs, shapes/dtypes, resource
timeouts and process-tree termination.

## Resources and execution

The three literature inputs total less than 3 KiB. All observations are used;
there is no downsampling. A separate **30 s / 1,024 MiB** watchdog run passed all
nine tests in 3.16 s including interpreter startup (pytest: 1.83 s). Sampled
worker-plus-child RSS peaked at **749.83 MiB**, including two parallel workers.
RSS sampling can miss short spikes or count shared pages repeatedly; it is not
an OS allocation reservation. Dedicated performance runs use smaller synthetic
workloads and are documented in [GAMMA_RESULTS.md](../benchmarks/GAMMA_RESULTS.md).

```sh
python -m pytest tests/test_gamma.py -q
python -m validation.tests --literature-study Ferretti2016 -q
python -m validation.tests --literature-case ferretti.csi.equations.v1 -q
python -m validation --store /path/to/research-store check --artifacts
```

Full basic regression: **2,087 passed, 2 existing strict xfails**, 21.75 s in
pytest, under a 90 s / 3 GiB watchdog (sampled tree peak about 2.90 GiB). Existing
construction/data tests account for much of that memory; no construction code
changed. A first full literature run hit an explicitly imposed 2 GiB watchdog;
the failure was retained and the complete suite was rerun with a 3 GiB cap.
This does not change the 1 GiB budgets for gamma-specific validation/performance.
The capped rerun passed **30 literature tests** in 48.29 s (49.75 s including
startup), with sampled tree RSS about **2.11 GiB**. All 171 artifact references
and 32 existing research events passed integrity checks. Nine gamma JUnit
records retain the paper DOI, case identity and distinct evidence roles.

An initial coverage invocation targeting the dotted module imported NumPy twice
before collection; using a source directory resolved the instrumentation setup.
No metric failure or tolerance adjustment was hidden by that correction. Source
docstrings retain short Notes and separately describe the actual public contract;
the tutorial and API proposals stay in the shared local proposals directory.
Eight doctest statements pass; SciPy's NumPy-docstring parser verifies parameter,
return and exception sections. The changed core/driver files parse as Python
3.9; local execution used Python 3.13. The gamma/gamma-star Notes are 84/87 words
in two paragraphs each. The separate tutorial is 100 words/two paragraphs for
gamma and 39 words/one paragraph for gamma-star, versus 222/41 words in the
corresponding existing descriptions (including the old interpretation list/note).
