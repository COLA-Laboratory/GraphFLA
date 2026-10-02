# Gamma and gamma-star calculation review

Reviewed 2026-10-02 against the nine PDFs supplied for this task. Public
signatures are unchanged. The mathematical pooling in the previous version
already implements Ferretti's complete-square equations, including exact ties.
The correction is numerical: squared effects could overflow or underflow and
return NaN after a change of fitness units. No threshold was fitted to a paper
target, and `classify_epistasis` behavior was not changed.

## Definition and implementation decision

For each directed focal substitution, compare its effect `s` with the same
substitution's effect `t` after changing a different variable. Pool
`sum(s*t)/sum(s*s)` over all such directed comparisons. Equivalently, enumerate
each complete two-variable square and both pairs of parallel sides, pooling
`a*b` and `(a*a+b*b)/2`. These are raw second moments, not a Pearson correlation
of arbitrarily oriented undirected edges, and not an average of square ratios.

[Ferretti et al. (2016)](https://doi.org/10.1016/j.jtbi.2016.01.037), published
Eqs. (1), (3), and (11), provide the equations. Gamma-star applies the same
pooling to signs. An exact zero has sign zero, so its product and squared-effect
terms are zero; the rest of its square still counts. The optional tolerance in
Eq. (10) makes the interval `[-epsilon, epsilon]` neutral. Current GraphFLA uses
zero tolerance. Construction epsilon is a different graph setting.

**Correction to the previous audit:** Eq. (12) and Appendix C.3 explicitly
require no neutral mutations for `gamma_star = 1 - phi_s - 2*phi_rs`. The
previous scout called a 0.0449 discrepancy on tied TEM data a defect. That
inference was incorrect: it compared an effect-weighted sign statistic with
motif frequencies on a different population, outside the theorem's assumptions.
The published equation number is (12); (15) in the older combined preprint is
not the published numbering. A tied square `[0,0,1,2]` gives gamma-star `2/3`
under Eq. (11), without satisfying the tie-free motif identity.

For missing genotypes, GraphFLA pools only complete observed squares in both
sums. This is an explicit conditional statistic, not Ferretti's separate
missing-data estimator `(rho_1-rho_2)/(1-rho_1)` in Section 2.2. All allele pairs
are enumerated regardless of ordinal adjacency or graph epsilon. Multiallelic
weights are per substitution/background comparison, not equal per locus. These
existing choices are now documented; choosing another estimator is an API
discussion rather than a silent calculation fix.

## What the nine supplied papers can validate

| Supplied PDF | Relevant evidence and disposition |
| --- | --- |
| `1-s2.0-S0022519316000771-main.pdf` — Ferretti et al., 2016, JTB | Primary definition, Eqs. (1)–(3), (10)–(12), (26), Appendices C.1/C.3. Figure 4c prints TEM gamma/gamma-star 0.85/0.59 and csI 0.33/0.25. Formula and Figure 4 pages visually checked. Main empirical targets below. |
| `s41437-018-0110-1.pdf` — [Ferretti et al., 2018, Heredity](https://doi.org/10.1038/s41437-018-0110-1) | Figure 5 repeats the TEM/csI examples and identifies MAGELLAN as the source of statistics. Useful confirmation, but not an independent experimental replication or a resolution of the input discrepancy. |
| `file.pdf` — [Franke et al., 2011](https://doi.org/10.1371/journal.pcbi.1002134) | Supplies the 8-locus A. niger data in Table S1. Extract the 32-row csI subset specified by arg/pyr/leu/oli/crn, with fwn/phe/lys absent. Does not itself report gamma. Supplement **s009** is Table S1; s001 is Figure S1. |
| `0807.3002v1.pdf` — [de Visser, Park & Krug, 2008 preprint](https://arxiv.org/abs/0807.3002) | Table 1, pp.32–33, provides an earlier CS I table and separate ranks. Its printed fitness precision yields different near-ties from the 2011 table. Keep this input separate; do not use its ranks to break ties silently. No gamma target (it predates the definition). |
| `nihms-1795281.pdf` — [Song & Zhang, 2021](https://doi.org/10.1111/evo.14363) | Table 1 prints raw and extrapolated `1-gamma` for three datasets. The empirical code pools products over complete squares but variance over **all neighbor effects**. With incomplete data or unequal alphabet sizes, this is a different denominator weighting from GraphFLA. The extrapolated results also require their noise model. These are not interchangeable scalar test targets. |
| `iyag140.pdf` — [Ogbunugafor, 2026](https://doi.org/10.1093/genetics/iyag140) | Perspective, no new matching gamma target identified. Its Data availability points to a useful freshly released 32-row TEM MIC table. This independently confirms the all-mutant MIC of 4100 and enables a new source-pinned input check. |
| `2604.22611v1.pdf` — [Ribeca et al., 2026 preprint](https://arxiv.org/abs/2604.22611) | Reuses the correlation-of-effects definition and derives sign-epistasis fractions for unstructured Gaussian random fields, Eqs. (3)–(4). These ensemble/model relations are not universal sample identities. Data statement still says Zenodo `[ADD]` in the supplied version. No new exact empirical scalar target promoted. |
| `2605.03046v1.pdf` — [Ghafari et al., 2026 preprint](https://arxiv.org/abs/2605.03046) | Eqs. (1) and (3) connect gamma with the spectrum. Pure order-k landscapes give `1-2*(k-1)/(L-1)`; orders 1–5 are checked independently in basic tests. Expected peak counts depend on model assumptions, so are not used as deterministic gamma oracles. |
| `nihpp-2026.06.25.734428v1.pdf` — [Martí-Gómez & McCandlish, 2026 preprint](https://doi.org/10.64898/2026.06.25.734428) | Discusses gamma_i→j in relation to averaged squared local epistasis (p.4), then generalizes local epistasis statistics and inference. The gpmap-tools/deltaU code is relevant to that inference task, not a second published global gamma/gamma-star target. |

Song & Zhang code was checked at commit
[`a9d15169`](https://github.com/song88180/fitness-landscape-error/blob/a9d15169d0488b5e615467132710d6847dc2f73e/5_Empirical_Extrapolation/trna_Domingo/Generate_raw_data.ipynb),
code cell 10 (`cal_gamma`). Its simulation utility also uses `np.cov` with the
default sample normalization divided by `np.var` with population normalization;
that is separate from the empirical implementation.

## Empirical results and unresolved reproductions

All three inputs contain 32 genotypes, 80 squares and 640 directed comparisons.
Use natural logarithms of positive W/MIC. Log base does not change these ratios;
raw versus log fitness can change gamma. Every production result agrees with
independent directed enumeration and the complete-cube distance-correlation
identity. All 20 ordered position-pair contributions per input are checked too.

| Input | Gamma: independent / GraphFLA | Figure 4c | Gamma-star: independent / GraphFLA | Figure 4c |
| --- | ---: | ---: | ---: | ---: |
| csI, Franke 2011 Table S1 | 0.327295054464430 | **0.33: matches precision** | 168/624 = 0.269230769231 | **0.25: unresolved** |
| CS I, de Visser 2008 Table 1 | 0.325198831796701 | rounds to 0.33 | 152/624 = 0.243589743590 | **0.25: unresolved** |
| TEM, 2026 released MIC table | 0.835347932044410 | **0.85: unresolved** | 324/528 = 0.613636363636 | **0.59: unresolved** |

The first two rows are different printed representations of the same
experiment, not two independent empirical confirmations. The 2018 paper repeats
the targets, also not a new experiment. The two unresolved case records retain
the printed targets and `definition_match: unresolved`; they are **not** among
the six passing scientific tests. `python -m validation.gamma` prints each
comparison, including the failures. Those unresolved cases must not be replaced
with GraphFLA's outputs or relabelled as successful reproductions.

The original MAGELLAN `.fl` input files remain unavailable (HTTP requests for
both named files timed out on 2026-10-02). Different input precision or
preprocessing could explain differences, but this has not been established.
Neither a tolerance sweep nor arbitrary tie-breaking establishes which input
the authors used. The package's old TEM CSV issue is recorded separately; this
review does not overwrite it or claim that correcting 41000 to 4100 resolves
Ferretti's numbers.

## Reference-code discrepancy

The [MAGELLAN gamma.c mirror](https://github.com/rdiaz02/OncoSimul/blob/26d96c797b8f2f1a8cdad7ac53b3d85c09fd8ca3/OncoSimulR/src/FitnessLandscape/gamma.c)
was pinned and its bytes matched the previously downloaded archive. In
`deltaf`/`deltafMultiAllele`, `df > -tolerance` is strict. At tolerance zero,
`df == 0` falls through to -1, contrary to Eq. (10). `GammaDistance` uses this
function with zero tolerance for gamma-star. Compiling the unchanged library
and calling its routines gave:

| Input | MAGELLAN gamma | MAGELLAN gamma-star |
| --- | ---: | ---: |
| csI 2011 | 0.327295043898 | 0.2625 |
| CS I 2008 | 0.325198852450 | 0.2375 |
| TEM 2026 | 0.835347929209 | 0.5625 |
| All fitness values equal | undefined | **1** |

Small numeric gamma differences arise from the mirror's float32 fitness
storage. The sign discrepancy is substantive. This does **not** prove that the
2016 paper ran this exact mirror version, nor does this mirror reproduce its
printed gamma-star values. GraphFLA keeps the published zero-sign rule.

## Numerical correction and verification

The four-point input `[0,1,2,4]` has gamma `8/9`. Before this change, multiplying
it by `1e200` or `1e-200` returned NaN. Numeric contributions now carry a shared
power-of-two scale when needed, and are combined with their original
effect-size weights. Finite endpoints whose difference overflows are handled
by first halving the endpoints. Signs are taken before scaling. Unrelated
outliers and large neutral squares cannot erase small informative squares.
The ordinary-magnitude path keeps the prior arithmetic.

Basic tests cover extreme scales, minimum-range subnormals, flat and incomplete
inputs, every three-level two-locus assignment, heterogeneous allele counts,
parallel execution, the high-dimensional dictionary fallback, construction
epsilon independence, motif-identity assumptions, and exact spectral anchors.
Source docstrings state actual behavior and contain eight passing doctest
statements. No construction code, fitness data, API signature or website was
changed. Further API choices (sign tolerance, estimator selection, diagnostics
or position-resolved outputs) remain for discussion.

Run independently:

```sh
python -m pytest tests/test_gamma.py tests/test_analysis_oracles.py
python -m validation.tests --literature-study Ferretti2016 -q
python -m validation.gamma
python -m validation --store /path/to/research-store check --artifacts
```

Fixture provenance and conversion rules are in
[`tests/fixtures/literature/ferretti2016`](../tests/fixtures/literature/ferretti2016/).
The dedicated suite uses less than 3 KiB of numerical input, runs in about two
seconds locally and includes all observations. It does not download data or
run author code. Local exploratory source/compile logs are kept outside Git.

**Acceptance boundary:** formula implementation and the numerical correction
are verified on the stated inputs; one empirical gamma target is reproduced.
The remaining Figure 4 values are unresolved. This is stronger than matching
one number, but is not a claim that every published result has been reproduced
or that every possible landscape is validated.
