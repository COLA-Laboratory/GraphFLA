# Integrated epistasis order analysis

2026-10-03. This extends the coefficient implementation reviewed in
[WALSH_HADAMARD_REVIEW.md](WALSH_HADAMARD_REVIEW.md). The user authorized a
single canonical call returning coefficients and order summaries, with shared
computation and plain dictionary results. No website content is changed.

## One result and one fitting pipeline

`walsh_hadamard` returns a dictionary: `coefficients` is the final
coefficient table; `order_summary` contains the nested-fit curve and final-model
spectrum; `fit_info` states the estimator, references and evaluation conventions.
Access the coefficient table as `result["coefficients"]`.

The internal pipeline encodes retained observations once, counts and allocates
one ordered design, then uses column views for each order. It fits the highest
order once, checks coefficient identifiability, and retains its solution instead
of recomputing it at the end of the profile. Tall OLS problems use one augmented
QR of [design, centered fitness], preserving every prefix least-squares problem
and residual norm; no normal equations or full Q are retained. Numerical rank
uses the original sample count in its relative SVD tolerance. Smaller designs
use direct least squares. Lasso fits each order separately with the same CV
splits, including when a mutable RNG or random_state=None is supplied. Selecting
alpha separately per order is intentional, as is preserving negative fit gains.
Invariant columns are excluded before term or spectrum-support enumeration;
counting a tiny model must not permit combinatorial loops over constant sites.

Read `result["order_summary"]` directly; it is already computed and stored.
Tests check one design build and exactly one fit for each requested nonconstant
order in the main pipeline. At the user's request, `higher_order_epistasis`
has been removed entirely, including its module, public exports, benchmark
and `profile()` / `list_metrics()` entry. There is no compatibility alias or
separate score-only fitting branch. Migrate calls to
`walsh_hadamard(landscape, max_order=k)["order_summary"]`; the former scalar score
is the last row's `r2`. A removal test checks public imports and registry behavior.

## What each quantity measures

For nested order k, R2_k = 1 - SSE_k/SST on the retained training observations,
and delta_R2_k = R2_k - R2_(k-1). The constant baseline has R2=0 when fitness
varies. Constant fitness gives NaN R-squared/increments, not an apparent 100%
interaction explanation. RMSE is sqrt(SSE_k/N) in input units. CV tunes alpha;
it does not turn these values into held-out performance. Gains are conditional
on lower-order terms and the selected fitting procedure, not causal effects or
significance tests. OLS training gains are nonnegative apart from numerical
roundoff; regularized gains can be negative because shrinkage is retuned.

The `model_variance_fraction` column answers a different, explicit question:
how much of the highest-order fitted model's uniform-product variance belongs
to each order? Write that model as epsilon_0 + sum_k g_k. Distinct supports are
orthogonal under uniform independent states, while allele columns within a
support can be correlated. At site i their covariance is I/s_i - 11^T/s_i^2.
Apply its square root along each coefficient-tensor axis and sum squares, then
normalize by the total model variance. This is an exact analytic contraction,
not genotype sampling, an exponential-grid reconstruction, or bare squaring of
multistate coefficients. Coefficients are rescaled together before contraction
to preserve fractions in extreme fitness units. Constant models give NaN.

A complete full-order OLS decomposition has model variance equal to observed
uniform-grid variance, so its order fractions equal incremental R-squared.
If truncated, the fractions instead sum to one within the fitted model; they
need not sum to the observed-data R-squared. Incomplete populations break order
orthogonality under empirical sampling, and Lasso shrinkage also separates
model variance from explained-data variance. The table therefore keeps these
columns distinct. Nonzero counts concern exact orders in the highest-order
Lasso solution; they are not significance claims and are NA for OLS.

Rank-deficient OLS may still have unique predictions on observed rows, but its
coefficients do not define a unique decomposition. The unified API therefore
raises, as before, and never silently changes to a regularized estimator.
Users can explicitly select Lasso to estimate such a model.

## Why tree depth is not interaction order

On a complete four-bit cube, y=x1+x2+x3+x4 is entirely additive. Nevertheless,
single regression trees of depths 1, 2, 3 and 4 have training R-squared values
0.25, 0.50, 0.75 and 1.0: their successive gains are needed just to represent
all four additive effects. Interpreting those gains as higher-order interaction
creates false pairwise, three-way and four-way contributions. A regression tree
can also split the same numeric variable repeatedly, increasing depth without
increasing the number of interacting variables. Thus trees can be predictive
approximations, but tree-depth profiles are not this metric's cheap estimator.
The additive counterexample is an executable regression test. See also
[sklearn's tree documentation](https://scikit-learn.org/stable/modules/tree.html)
on piecewise-constant approximations and greedy fitting.

## Independent evidence and limits

Existing Faure and Phillips targets remain unchanged, with test consumers
adapted to the new result/table interfaces. New cited independent-equation cases
freeze nested scores and model spectra before calling GraphFLA: the printed
3x3 Table 1 phenotype values and a 2x3x4 centered multistate Lasso fixture.
An independent forward finite-difference operator is inverted for those checks;
order variances are calculated by explicit full-grid predictions rather than
the production covariance contraction. These are equation checks, not newly
claimed published numerical results. All previous case fingerprints remain.

Basic tests additionally check analytic spectra, incomplete-data refitting,
common CV splits, negative regularized gains against an independent design,
multistate reference invariance, QR residual preservation, constant/zero-order
models, scale changes, rank semantics, API removal, memory guards and reuse.
The paper's large Figure 2/Papkou full pipelines and external predictive R2
remain outside this validation claim.
