# Roughness-to-slope performance and test audit

Run begun 2026-10-02, final validation 2026-10-03. Calculation remains OLS with
an intercept and the existing reference/rank conventions. Correctness fixes
and the limits of published-result reproduction are in
[the scientific review](../validation/R_S_RATIO_REVIEW.md).

## Allocation and runtime controls

The implementation stores a code vector per variable and fills one design block
at a time, rather than assembling and concatenating all one-hot intermediates.
It reads only required vertex attributes, avoiding degree and basin metadata.
Prediction uses einsum without an additional N-by-p product temporary. Objective
normalization prevents squaring extreme original units.

The design-block target is 16 MiB, with at least p+2 rows for QR compression.
The augmented quadratic QR size must fit 64 MiB before allocation; oversized
or underdetermined designs return NaN with a warning. These are workspace
controls, not total RSS guarantees: input tables, arrays, LAPACK workspace and
QR temporaries also use memory. The QR path reduces tall inputs without forming
normal equations; both paths preserve the exact OLS problem. Time remains
O(N p²), with an O(p³) reduced solve; extremely wide models are not cost-free.
No sampling, configuration truncation or regularization was introduced.

## Controlled performance comparison

Existing `tools/benchmark_analysis.py` and ASV metric module were extended for
r/s. Each workload uses three fresh processes with one warmup and three timed
calls. Construction is outside the timer. Native thread counts and hash seed
are fixed; each worker is guarded at 30 seconds and 1,024 MiB process-tree RSS.
Baseline, final and repeated old-code control ran sequentially without tests or
competing benchmarks. Input hashes and full scalar outputs match the old code
on these regular, identifiable inputs. Semantic fixes have separate tests.

The old time below is the faster of baseline and repeated-control medians.
Each improvement exceeds both 5% and three combined process-median MADs against
both old rounds. These are measurements on this machine, not general speed
promises. Reports, raw samples, NPZ outputs, implementation/protocol hashes and
kernel snapshots are in [results/r-s-ratio-2026-10-02](results/r-s-ratio-2026-10-02/).

| Workload | Retained rows | Old → final, ms | Speedup |
| --- | ---: | ---: | ---: |
| Boolean 6 variables | 64 | 1.059 → 0.509 | 2.08× |
| Boolean 14 variables | 16,384 | 42.188 → 37.921 | 1.11× |
| Categorical 4 states × 5 variables | 1,024 | 2.533 → 1.816 | 1.40× |
| Categorical 64 states × 2 variables | 4,096 | 14.962 → 13.274 | 1.13× |
| Incomplete Boolean 12 variables | 1,002 | 3.313 → 2.661 | 1.25× |
| Boolean 72 variables, selected backgrounds | 560 | 9.736 → 8.283 | 1.18× |
| Ordinal 8 levels × 3 variables | 512 | 1.143 → 0.644 | 1.78× |
| Mixed categorical/ordinal/Boolean | 32 | 1.035 → 0.527 | 1.96× |

Largest final worker high-water RSS was 284.69 MiB; the process-tree monitor
observed at most 295.14 MiB, including construction. The incomplete benchmark
starts with 1,024 inputs; existing construction retains 1,002. Literature Kuo
validation retains all 197,890 observed rows and exercises the tall QR path.

## Basic-test audit

`tests/test_r_s_ratio.py` provides analytic nonzero residual/slope checks,
extreme units and opposite finite extremes, constant versus pure interaction,
minimization/maximization, profile/list_metrics integration, binary relabelling,
multistate reference dependence, ordinal single-variable curvature, incomplete
mixed fits, constants/unused states, rank deficiency and saturation, empty
inputs, invalid/nonfinite values, numerical failure and workspace guards. The
blocked and direct solvers are compared to an independently constructed design.

Removed the loose random-HoC `>0.5` check and duplicate additive check in
`test_metrics.py`. Consolidated repeated unit-invariance and constant tests from
`test_analysis_oracles.py`; its higher-order R² assertions remain. In particular,
the old large-unit checks tested only exact additivity or a zero-slope pattern,
which masked overflow/underflow in ordinary finite ratios. The new tests use
an analytic r/s=3/8 with nonzero numerator and denominator. Existing golden
landscape snapshots are retained as integration/regression coverage, not
presented as independent scientific evidence. No snapshot expectation changed.

The independent literature suite adds nine cited tests and three populations.
It reuses the EE contract, including input hashes, offline execution, citation
roles and separate CI. Default basic-test collection does not run these data
reproductions. Full-suite output, coverage, doctest and resource records are in
the shared local proposal directory.

## Final verification

Full basic suite: 2,111 passed, two existing strict xfails (construction integer
precision). The 36 focused r/s tests cover 107/107 core statements and 50/50
branches. All 39 literature tests pass, including nine new r/s tests. The
isolated r/s literature run took 3.22 s including interpreter startup and
peaked at 752.03 MiB under 30 s / 1 GiB. This includes independent dense
oracles and author replay, not just the production metric. The full suite
uses its preexisting 120 s / 3 GiB budget: 49.98 s, 2.12 GiB. An initial
full-suite attempt at 1 GiB was terminated by the watchdog and recorded;
the prior gamma review already measured about 2.11 GiB for that suite.
All 16 ASV parameter/method combinations and four docstring examples pass.
The 26 prior case files remain byte-identical, and 181 artifacts verify.
