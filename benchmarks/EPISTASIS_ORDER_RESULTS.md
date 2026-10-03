# Integrated epistasis analysis: performance and verification

2026-10-03. `walsh_hadamard` now returns coefficients, nested-order scores and
the final model's order spectrum together. `higher_order_epistasis(result)`
extracts the existing summary without fitting. See the
[scientific and API review](../validation/EPISTASIS_ORDER_REVIEW.md).

## Comparison scope

The baseline is the preceding corrected coefficient implementation (`e37c79a`)
plus separate calls to the old higher-order function at orders one and two.
The candidate is one integrated order-two call. This compares the work needed
to obtain both coefficients and the whole score curve; it does **not** claim
that adding a summary makes a coefficient-only call universally faster.

Seven bounded workloads ran sequentially, each with three fresh processes,
an untimed warmup and three timed calls per process. Construction was outside
the timer. Each worker had one native numerical thread and a 30-second /
1,024 MiB process-tree RSS watchdog. A second old-code round controls for drift.
The old time below is the faster of the two old-round medians. An improvement
is marked clear only if it exceeds 5% and three combined median absolute
deviations against both old rounds. These are local measurements.

| Workload | Rows | Separate → integrated OLS, ms | Speedup | Fixed Lasso, ms | Lasso CV, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boolean, 6 variables | 64 | 2.204 → 2.228 | within noise | 2.513 | 8.078 |
| Boolean, 10 variables | 1,024 | 7.370 → 5.386 | 1.37x | 5.321 | 34.498 |
| Categorical, 3 states x 3 variables | 27 | 2.524 → 1.685 | 1.50x | 2.169 | 7.915 |
| Categorical, 4 states x 4 variables | 256 | 4.578 → 2.787 | 1.64x | 2.926 | 18.721 |
| Incomplete Boolean, 12 variables | 1,002 | 9.356 → 6.827 | 1.37x | 6.243 | 47.687 |
| Ordinal, 4 levels x 3 variables | 64 | 2.856 → 2.006 | 1.42x | 2.437 | 10.977 |
| Mixed categorical/ordinal/Boolean | 32 | 2.595 → 1.866 | inconclusive | 2.242 | 7.923 |

The mixed case has a 1.39x point estimate but fails the combined-noise gate;
it is not counted among the five clear improvements. Fixed Lasso uses alpha
0.01; CV uses three seeded folds and selects alpha separately within each order.
These regularized timings have no legacy-equivalence speedup claim.

Largest worker high-water/monitored RSS was 219.03 MiB for integrated OLS,
211.83 MiB for fixed Lasso and 216.88 MiB for CV. The two old rounds reached
213.30 and 215.03 MiB. This small workload range demonstrates bounded execution,
not a reduction in process-wide memory. The design is built once; tall OLS uses
one augmented QR, and the highest-order solution is retained. Each lower-order
fit is still required for the defined nested-score curve.

Every OLS coefficient, order, original position and label, plus every cumulative
R-squared value, agrees with the separate-call baseline. There is no categorical
label remapping in this comparison. Raw samples, snapshots, source/protocol
hashes and equivalence results are in
[results/epistasis-order-summary-2026-10-03](results/epistasis-order-summary-2026-10-03/).

## Verification scope

The 20 new basic tests check analytic binary and multistate spectra, explicit
full-grid reconstruction, incomplete-data refitting, Lasso shrinkage and negative
gains, common CV folds, reference invariance, QR residual preservation, constant
fitness, extreme units, rank semantics, invariant columns and API compatibility.
They also prohibit encoding or solving when extracting an existing result,
check exactly one design construction, and demonstrate why tree depth fails as
an interaction-order proxy on a completely additive function.

Together with the 53 existing coefficient tests, focused coverage reaches
216/219 statements and 89/92 branches in the shared kernel. Remaining paths are
the zero-covariance CV shortcut, the defensive selected-alpha overflow error,
and an empty-support guard made unreachable by active-site enumeration.
The extraction wrapper's only uncovered statement is optional logging.
Coverage is diagnostic; no exclusions or fabricated coverage-only inputs were
introduced.

All 70 ASV workload/estimator/measurement combinations passed. Quick-run
summary extraction took 223–276 microseconds; this includes a defensive table
copy and is a smoke measurement, not a repeated performance estimate. The
73 focused tests and 14 executable docstring examples passed. Six actual
tutorial domains also passed bounded order-one OLS/CV checks (up to 512 input
rows and 1e6 design cells, 30 seconds / 1 GiB each). All six bounded inputs
admitted the requested OLS fit as well as the explicit Lasso CV fit.

The complete basic suite passes 2,183 tests with the two existing construction
precision xfails. The first final watchdog run crossed its 3 GiB process-tree
limit; a diagnostic rerun with durable verbose output passed unchanged under
the same limit (2,904.39 MiB sampled peak). Focused tests peaked at 313.70 MiB.
The initial failure is retained in local verification records; it is not used
as evidence of a metric-specific memory regression or improvement.

New independent-equation literature cases compare the printed Faure Table 1
landscape and a bounded multistate Lasso fixture against separately constructed
scores and explicit-grid model variances. Existing case targets and golden
snapshots remain unchanged. These checks do not claim full Figure 2 or Papkou
pipeline reproduction, held-out prediction, or statistical significance.
The full literature suite passes 46 tests; all 34 prior case files are byte
unchanged. The validation store checks 211 artifacts and 41 immutable events
after appending the two new observations and their checkpoint.
