# Walsh-Hadamard calculation and evidence review

Scope approved 2026-10-03: correct labels, general discrete-state computation,
rank handling and resource use; add explicit OLS/Lasso selection and optional
cross-validation, without changing the surrounding analysis architecture.
Reference: [Faure et al. (2024)](https://doi.org/10.1371/journal.pcbi.1012132).

## Mathematical convention

The coefficients are background-averaged differences, epsilon = V H y, not
unweighted Fourier-Walsh amplitudes. Backgrounds are uniform on the Cartesian
product of states observed in the retained population. For a focal state a at
site j, its inverse-design factor is `1[x_j=a] - 1/s_j`; multiply these factors
for interactions. The constant column is one. This is H^-1 V^-1 from Eqs.
(12), (23)-(25), after cancellation of full-space state-count products.
It remains valid for different numbers of states at different sites.

All variable types use their discrete states; ordinal spacing is not used.
Boolean variables use zero as reference when observed, and other variables
use the first retained row. Original labels survive computation. Sequence
positions refer to original one-based sites; generic frame positions retain
input feature order, including invariant columns. A reference change can
change coefficients and regularized fits. Labels are metadata, not numerical
encodings or parser inputs. Reserved delimiters in labels are escaped.

Full-order OLS on the complete product equals the direct transform. Incomplete
or truncated fits estimate coefficients of the chosen model; even a unique fit
does not prove recovery of a true full-space coefficient. Under the uniform
complete Cartesian population, different orders of this centered product basis
are orthogonal as groups, although alternate-allele columns within a group need
not be orthogonal. Full-data OLS truncation therefore agrees with refitting
through that order. This equivalence does not generally hold on an incomplete
or differently weighted population, or when regularization is retuned. No
unobserved states are guessed and no construction-filtered genotypes are
recovered by this metric.

## Estimator behavior

Default OLS uses least squares with an intercept and checks column rank.
Too few observations are detected before allocating the design; other rank
failures use the solver's numerical rank. A concise ValueError gives counts
and an action; no minimum-norm coefficient table is silently returned.

Explicit Lasso minimizes mean squared residual / 2 plus alpha times the sum
of absolute nonconstant coefficients. The constant is unpenalized, and columns
are not standardized, preserving the stated coefficient units. This differs
from the author script, which penalizes its constant column too. Positive
numeric alpha uses these supplied fitness units. Internal fitness scaling and
corresponding alpha scaling preserve the objective while avoiding overflow.
Alpha='cv' uses 100 logarithmically spaced penalties (alpha_max through
0.001*alpha_max), shuffled K-fold CV (five folds, seed 0 by default), and refits
on all retained observations. It does not return held-out performance or
claim unique unregularized coefficients. ConvergenceWarning remains visible.
A constant solution with no selection needed records alpha=0 for the CV mode.

This general API does not reproduce the authors' repeated 10-fold CV, fixed
lambda grid, random train/test subsampling, imputation pipeline, measurement
error adjustment or biological contact enrichment. Those are distinct
scientific procedures and must not be inferred from a passing transform test.

## Verification evidence

- Table 1: all nine 3x3 coefficients compared to the printed values with the
  justified half-last-decimal tolerance (0.005); a separate finite-difference
  calculation verifies coefficients from the printed phenotypes numerically.
- An independent 2x3x4 general combinatorial landscape tests all 24 coefficients
  through third order, including reference labels and population alignment.
- Pinned author recursive/elementwise matrices are checked against the forward
  finite-difference operator and used for a full-rank incomplete second-order
  fit. The author commit is daabe62d0a8256e2333be8818324413daf723486.
- A centered complete toy landscape permits a meaningful fixed-alpha comparison
  with the authors' penalized-constant objective. Every coefficient and the
  Lasso optimality conditions are checked. This is an author-procedure check,
  not a paper Figure 2 reproduction or a promise about sparse recovery.

Inputs, source hashes, licenses, extraction instructions and limitations are in
[fixture provenance](../tests/fixtures/literature/faure2024/PROVENANCE.md).
Four new immutable cases and five dedicated tests reuse the existing literature
contract, filters, verified inputs, JUnit metadata and separate CI job. Tests
never download data. All historical cases remain unchanged.

## Allocation and API migration

The design is filled directly in row blocks; no one-hot interaction table,
quadratic diagonal weighting matrix, or samples-by-terms-by-sites arrays are
built. Model size is counted before enumerating terms. The default max_cells
is reduced from 1e9 to 1e7 (80 MB for the design alone), now counting all terms,
including the constant and first-order columns. It is not a process RSS bound.
The existing bounded benchmark runner supplies independent time/RSS watchdogs.

The public name and four output columns remain. New method/alpha/CV/solver
parameters are keyword-only. max_order=0 now genuinely fits just a constant.
WT remains the historical label for that constant, not reference fitness.
DataFrame attrs carry fit information, original column labels and reference
alleles; save them separately when exporting to formats such as CSV.
Corrected categorical labels and original positions can change downstream joins.
Default OLS errors replace silent nonidentifiable estimates. These are intended
behavior changes, documented in the changelog, not relaxed regression targets.
