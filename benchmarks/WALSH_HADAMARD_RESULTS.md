# Walsh-Hadamard performance and test audit

This records the initial coefficient-only implementation at `050776d`.
The later integrated coefficient/order-summary measurements are documented in
[EPISTASIS_ORDER_RESULTS.md](EPISTASIS_ORDER_RESULTS.md).

2026-10-03. The optimized inverse design preserves Faure's background-average
normalization, using direct products of centered state indicators. OLS and
explicit Lasso/CV have separate scientific contracts; see
[the calculation review](../validation/WALSH_HADAMARD_REVIEW.md).

## Controlled workloads

`tools/benchmark_analysis.py --metric walsh_hadamard` runs bounded OLS workloads;
`--walsh-method lasso` and `--walsh-method lasso_cv` select the regularized modes.
Each mode uses seven cases, three fresh processes per case, an untimed warmup
and three timed calls. Construction is outside the timer. Every worker has a
30-second / 1,024 MiB process-tree RSS watchdog and one native numerical thread.
All workloads use order two and at most 1e6 design cells. Fixed-alpha Lasso
uses 0.01; CV uses three seeded folds. Full ASV discovery independently checks
all 42 workload/estimator/measurement combinations.

Baseline, candidate, repeated old-code control and the two Lasso modes ran
sequentially without competing tests. The conservative old time below is the
faster of the two old-round medians. Each improvement exceeds both 5% and three
combined median absolute deviations against both old rounds. Values are local
measurements, not general performance guarantees.

| Workload | Retained rows | Old OLS → new OLS, ms | Speedup | Lasso, ms | Lasso-CV, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boolean, 6 variables | 64 | 7.033 → 1.594 | 4.41x | 1.897 | 5.467 |
| Boolean, 10 variables | 1,024 | 37.985 → 4.953 | 7.67x | 4.444 | 26.626 |
| Categorical, 3 states x 3 variables | 27 | 5.161 → 1.470 | 3.51x | 1.742 | 5.337 |
| Categorical, 4 states x 4 variables | 256 | 14.633 → 2.392 | 6.12x | 2.389 | 15.049 |
| Incomplete Boolean, 12 variables | 1,002 | 46.432 → 6.215 | 7.47x | 5.319 | 37.947 |
| Ordinal, 4 levels x 3 variables | 64 | 7.999 → 1.556 | 5.14x | 2.140 | 8.511 |
| Mixed categorical/ordinal/Boolean | 32 | 5.488 → 1.605 | 3.42x | 1.620 | 5.541 |

Largest new worker high-water/monitored RSS: OLS 213.02 MiB, Lasso 213.16 MiB,
Lasso-CV 218.67 MiB, versus 349.77 MiB in the initial old-code round. The
incomplete case starts with 1,024 rows; construction removes 22 isolated nodes.
No sampling or order reduction occurs inside the measured functions.

All OLS coefficients, orders, positions and labels are compared. Legacy
categorical labels use artificial one-character factor codes; the comparison
runner decodes those labels from the exact input before comparing corrected
labels. This explicit label correction does not hide coefficient differences.
Rank-deficient, high-cardinality delimiter failures and position corrections
have separate regression tests rather than being mislabeled as equivalent
legacy behavior. Raw samples, hashes, input/output checks and snapshots are in
[results/walsh-hadamard-2026-10-03](results/walsh-hadamard-2026-10-03/).
Final source changes after timing only clarified docstrings and made the example
portable across NumPy versions; the executable AST was verified unchanged.

## Basic-test audit

53 focused tests check finite-difference coefficients on complete 2x2, 3x3 and
2x3x4 spaces; original sequence/generic positions; actual and delimiter-containing
alleles at 64 states; mixed types with equal textual labels; incomplete fits;
underdetermined and overdetermined rank failures; exact Lasso soft-thresholding;
CV versus an explicit independent-design fold search; reproducible splits;
constant fitness; extreme units/penalties; missing/duplicate inputs; convergence
warnings; the matrix guard before term enumeration; graph round trips; and a
72-variable low-order model without exponential state-count products.

The old weak additive test was replaced with explicit multistate main effects,
a known intercept and zero interaction coefficients, plus nonzero interaction
oracles. Existing golden snapshots were not edited. Five targeted defects fail
against the saved pre-change implementation (position labels, original category
labels, large alphabet, underdetermination and rank deficiency).

Focused coverage reaches 149/150 statements and 67/68 branches in the production
module. The only uncovered statement is the defensive nonfinite selected-alpha
error: with finite fitness, this basis and the adaptive grid cannot normally
reach it. No artificial coverage-only test or exclusion was added.

## Domain and suite checks

Actual data from all six existing tutorials was checked at order one with at
most 512 input configurations, at most 1e6 design cells, and a separate
30-second / 1 GiB guard per domain. Pharmacology first uses the tutorial's
replicate aggregation; input missing rows and duplicates are removed explicitly.
Both OLS and three-fold Lasso-CV completed with finite coefficients:

| Domain | Retained rows | Observed states per variable |
| --- | ---: | --- |
| Protein | 512 | 20, 20, 20, 20 |
| Chemical biology | 512 | 21, 25 |
| Chemistry | 36 | 4, 3, 3 |
| Materials | 496 | 31, 31 |
| Microbiome | 512 | 26, 20 |
| Pharmacology | 512 | 2, 469 |

These are compatibility checks on bounded subsets, not validation of the full
empirical landscapes or claims about sparse-recovery accuracy. In constrained
spaces such as compositions, the full Cartesian background includes unobserved
or infeasible combinations; fitted coefficients describe that model extension.
Source hashes, retained populations, fit information and resource measurements
are saved in the local proposal directory. Peak domain-check RSS was 251.16 MiB.

Full basic suite: 2,163 passed / two existing strict xfails. Full literature
suite: 44 passed, including five Faure tests, under 120 s / 3 GiB. The isolated
Faure suite used 30 s / 1 GiB. All 42 ASV combinations and seven doctest examples
passed. All 30 prior case files remain byte-identical. Full validation and
resource logs are in `.codex-local/proposals/walsh-hadamard/`.
