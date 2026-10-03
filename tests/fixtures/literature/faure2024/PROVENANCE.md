# Faure et al. (2024) Walsh-Hadamard fixtures

Reference: Andre J. Faure, Ben Lehner, Verónica Miró Pina, Claudia Serrano Colome,
and Donate Weghorn. *An extension of the Walsh-Hadamard transform to calculate
and model epistasis in genetic landscapes of arbitrary shape and complexity.*
PLOS Computational Biology 20(5), e1012132.
https://doi.org/10.1371/journal.pcbi.1012132

Retrieved and inspected 2026-10-03. `manifest.json` pins original and derived
SHA-256 values. Original PDF and notebook/script downloads remain in the
external validation store under `papers/faure2024/sources`; offline tests use
only the small files here. The supplied PDF hash is recorded in the manifest.

## Table 1

`table1.csv` transcribes the nine sequences, phenotypic effects and epistatic
coefficients printed on PDF page 6, Table 1. The paper is CC BY 4.0; attribution
is above. Values are printed to two decimal places. The published-result case
uses an absolute tolerance of 0.005, half a printed decimal unit. It does not
claim equality to unrounded experimental estimates. Measurement-error columns
are not transcribed or modeled, and no uncertainty-propagation claim is made.

The test embeds the two variable sites at positions 6 and 66 in an otherwise
constant artificial DNA string, solely to test original-position reporting.
It does not claim that this padding is the original tRNA sequence. All nine
combinations and fitness values are retained; reference alleles are G and C.
An independent forward transform checks all nine coefficients at float64
precision separately from the printed-number comparison.

## Hand-specified general combinatorial landscape

`synthetic.json` contains the complete Cartesian product of 2, 3 and 4 states
(24 configurations in lexicographic order). For row i, the initial score is
`(17*i*i + 3*i) % 31 - 10`; subtract the population mean. This fixed toy input
is not the paper's 4^6 simulation or either large empirical population.
It exercises unequal alphabet sizes, ordinal labels treated discretely,
nonbiological column names and nonzero interactions through third order.

Independent expectations come from `validation/oracles/walsh.py`: enumerate
backgrounds, take signed differences at focal sites, average over other sites,
and assemble the forward operator. Its numerical inverse supplies a small
independent regression design. No production helper generates an expectation.
The incomplete test removes the final row and retains terms through order two.

## Pinned author source

Source: https://github.com/lehner-lab/whmatrixextms/tree/daabe62d0a8256e2333be8818324413daf723486

`author_matrices.py` contains exactly the source text of `H_matrix_recursive`,
`H_matrix`, and `V_matrix` extracted from code cells in
`whmatrixextms-benchmarking.ipynb`, with only a provenance header and NumPy import
added. AST source extraction excludes notebook execution and plotting code.
The MIT license is retained as `LICENSE.author`. Tests import only these
reviewed, hash-verified pure functions, with at most 24 rows/columns.

The author script `whmatrixextms.py::fit_lasso` fits `LassoCV` with the constant
column included, `fit_intercept=False`, 10 folds repeated three times, and a
fixed alpha grid. The production API leaves the constant unpenalized, uses
shuffled K-fold CV (default five folds, seed 0), and an adaptive 100-value grid.
There is no feature standardization in either approach. The author script's
main grid is 0.005 to below 0.25 in steps of 0.005; its method default extends
to below 0.5. These choices are not silently treated as the generic API default.

The fixed-alpha replay uses the same Lasso objective as the author, but with
alpha=0.05, full third order, this centered complete toy input, tol=1e-12 and
max_iter=100000. Here the constant is orthogonal to all other columns and zero,
so penalizing it does not change the answer. Frozen coefficients are obtained
from the independent inverse matrix and sklearn Lasso without importing
GraphFLA. Tests separately replay the pinned author matrix, compare every
coefficient, and check the Lasso optimality conditions. This is a bounded
procedural cross-check, not a reproduction of Figure 2, its cross-validation,
its generalization scores, or its biological enrichment statistics.

To regenerate the derived data without GraphFLA: transcribe the three Table 1
columns; enumerate the stated 24 rows and score formula; extract the three
named functions with `ast.get_source_segment`; evaluate the independent
forward operator and the specified fixed-alpha objective. Regenerated research
expectations require new case revisions once archived; do not overwrite old
case files. The original preparation script is preserved in the local W-H
proposal directory as an acquisition record, never run by tests.
