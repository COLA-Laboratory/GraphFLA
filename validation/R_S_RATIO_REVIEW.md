# Roughness-to-slope review

2026-10-02/03. Scope: calculation, basic-test audit, separate literature tests,
bounded performance measurements and source API documentation. Public signature
remains `r_s_ratio(landscape) -> float`. General combinatorial optimization is
part of the supported use case; RNA conventions are confined to validation.

## Definition and correction

For an explicit encoded design, fit an intercept and additive coefficients by
ordinary least squares. Use residual RMSE with denominator N, divided by the mean
absolute non-intercept coefficient. This is Szendro et al. (2013), Eqs. (3)-(5),
for binary 0/1 variables. RMSE is not residual variance or a degrees-of-freedom
estimate. The metric describes the supplied objective scale, not a compulsory
logarithm, and does not depend on maximization versus minimization or graph edges.

The old binary formula was correct. Confirmed defects were numerical overflow
and underflow in residual squares, invalid finite slopes from rank-deficient
designs, and unnecessary dense one-hot/prediction allocations. Earlier extreme
unit tests used a zero residual or zero slope, so they did not expose failure of
an ordinary nonzero finite ratio. The new analytic test has r/s=3/8 across
scales from 1e-308 to 1e306, negative scale and large additive offsets.

Fitness is power-of-two scaled, shifted and normalized before fitting. Small
inputs use SVD; tall inputs are reduced by blocked QR followed by SVD. No normal
equations, regularization, sampling or imputation are introduced. Singular rank
uses a relative cutoff `max(N, p+1)*eps` on the augmented design. A deficient or
underdetermined design returns NaN with a warning, since its coefficients and
slope are not identifiable. A full-rank saturated fit is allowed with a warning.
Constant/empty populations return NaN; a slope at most 1e-12 of the objective
range returns infinity. Invariant variables and unobserved category levels are
excluded, avoiding artificial dilution of the slope.

A categorical variable still uses drop-first coding, in pandas category order
(normally sorted observed labels). A multi-state reference change preserves the
fitted function and r but can change s. That is a convention dependence, not an
OLS error. No reviewed paper establishes a universal extension that justifies
silently replacing it with a reference-free formula. Ordinal variables retain
construction ranks; a nonlinear single-variable response contributes to r even
without interactions. Thus this scalar must not be advertised as a pure measure
of epistasis for every optimization problem. An explicit-reference API and an
optional categorical treatment of ordinal states are proposed locally, not
implemented in this calculation repair.

## Supplied literature inventory

All nine PDFs were read for relevant definitions and numerical targets. The two
Szendro files are versions of one paper, so there are eight distinct studies.
Local PDF SHA-256 values, extracted text and rendered formula/table checks are
in the shared local proposal directory, not redistributed in the repository.

| Supplied file | Definition and usable evidence | Decision |
| --- | --- | --- |
| `Szendro_2013_J._Stat._Mech._2013_P01005.pdf` | [Szendro et al. 2013](https://doi.org/10.1088/1742-5468/2013/01/P01005), §3.1 Eqs.(3)-(5), Table 2 p.15. Binary OLS/RMSE; Table 2 uses four-variable faces for larger landscapes. | Promote row A, 0.122 on 16 Chou configurations. |
| `1202.4378v2.pdf` | [arXiv v2](https://arxiv.org/abs/1202.4378), same study and equations. | Corroboration, not a second independent replication. |
| `042010v1.full.pdf` | [Ferretti et al. preprint](https://doi.org/10.1101/042010), §4 p.19 and Fig.4c p.20: Aspergillus 0.89, TEM 0.43 on log fitness. | Promote Aspergillus; TEM input mismatch remains unresolved. |
| `Song_2021_Evolution.pdf` | [Song & Zhang 2021](https://doi.org/10.1111/evo.14363), Methods p.2669 and Table 1. Explicit numeric targets and author notebooks; important formula/version and estimator differences below. | Promote Kuo independent OLS checks and separate author Ridge replay. |
| `140633.pdf` | [Aita et al. 2001](https://doi.org/10.1093/protein/14.9.633), Eqs.(6)-(10), Table II p.635. A weighted slope relative to a fitted pseudo-optimum, and slope/roughness theta=2.75 (abstract rounds to 2.8). | Different slope and reciprocal statistic; do not use 1/2.75 as a GraphFLA expectation. |
| `nrg3744.pdf` | [de Visser & Krug 2014](https://doi.org/10.1038/nrg3744), Box 1: residual SD relative to the linear fit divided by mean absolute coefficients. | Definition corroboration; no additional exact input/target promoted. |
| `pnas.200906192.pdf` | [Carneiro & Hartl 2010](https://doi.org/10.1073/pnas.0906192106), Roughness section and Table 1. Residual roughness with a specified reference constraint, without the slope denominator. | Related but different statistic. Printed roughness values are not r/s targets. |
| `pnas.201612676.pdf` | [Bank et al. 2016](https://doi.org/10.1073/pnas.1612676113), Methods p.14090 and Fig.4 inset p.14088. Methods calls roughness residual variance; cited definition uses RMSE. Plotted sublandscape distributions and guide lines lack exact scalar labels. | Do not read approximate plot coordinates as precise test targets. Existing input/code leads remain available for future author-artifact work. |
| `1405.3504v1.pdf` | [Blanquart et al. 2014](https://doi.org/10.1111/evo.12545), supplied preprint section on genotypic landscapes, Fig.7. Least-squares binary model on log fitness; simulations compare distributions. | Definition/model evidence; no fixed realization, seed and scalar target promoted. |

## Reproductions and limits

| Case | Target | Observed | Claim |
| --- | ---: | ---: | --- |
| Szendro Table 2 A | 0.122 | 0.121797044256 | Published precision, on the pinned existing Chou transcription. |
| Ferretti Fig.4c Aspergillus | 0.89 | 0.892830718043 | Published precision, on the source-pinned csi table. |
| Song Table 1 Kuo, author estimator | 4.063 | about 4.063031 | Author Ridge procedure matches printed precision; not the public OLS estimator. |
| Kuo OLS, reference A | Independent equation | 3.732443899068 | Public function agrees with independent explicit coding. |
| Kuo OLS, reference U | Independent equation | 4.062897962893 | Same residuals as A, different slope; agreement does not prove the published estimator is OLS. |

Paper tolerances are half the last printed decimal, fixed before execution.
The Kuo independent numerical budget is 1e-10 for 197,890-row fits. Both reference
fits and all residuals are compared, not just a quotient. The two dense reference
SVD residual vectors differ by about 3e-13 from reduction/solve roundoff, below
that predeclared budget. Small binary tables additionally use orthogonal 0/1
contrasts to check every coefficient and residual independently of either solver.

### Song paper, manuscript and code are not interchangeable

The supplied typeset PDF p.2669 shows a **4n** denominator for four-state slope.
The locally archived PMC author manuscript XML, Methods paragraph P34, shows
**3n**. The old scout's 3n transcription was from that manuscript; it was not a
faithful description of the supplied PDF. Preserve this version discrepancy.
Both four-indicator formulas require a coefficient constraint to be identifiable.

Author code commit `a9d15169d0488b5e615467132710d6847dc2f73e`,
`5_Empirical_Extrapolation/SD_seq/Generate_raw_data.ipynb`, uses ATCG indicators
for RNA strings containing U, then removes constant columns. On Kuo this leaves
27 columns, implicitly reference U. Its `cal_r_s` uses `Ridge(alpha=1)` and the
mean of all 27 absolute coefficients. The generic `utils/utils.py` uses OLS.
The independent OLS result and author Ridge result both round to 4.063, despite
being different estimators. The contract deliberately permits only
`author_result`, not `paper_result`, for the Ridge replay case with this version
and methods discrepancy. The public function has no Ridge penalty.

Domingo and Li Table 1 targets require their exact preprocessing and input
artifacts. Existing Domingo data do not close the 2.102 target; absence of U at
some sites also makes a naive OLS replay of all author indicator columns rank
deficient. The original notebooks are now pinned, but the linked Deep Blue data
record returned HTTP 403 in this session. No alternative transformation,
reference or tolerance was chosen to make those targets pass.

### Other unresolved targets

Ferretti's existing pinned TEM source gives 0.444084122858 versus printed 0.43.
It is not a passing reproduction. The known source/input issue and the accepted
gamma discrepancies remain unchanged; no fitness entry was repaired in this
session. Szendro rows B-J have existing method/input discrepancies in
`harvest/Szendro2013/`; they were not relabelled as passing evidence. The new
Aspergillus result does not settle gamma-star or any other metric.

## Input provenance and execution contract

The four new cases pin existing inputs, with no new large data copy:

- `data/BioSequence/Chou2011.csv`: 424 bytes, 16 rows. Historical GraphFLA
  log-fitness transcription, SHA-256 in the case. No further transform. The
  original raw-to-log conversion file and a source-specific license are not
  available in the prior dossier. This test certifies the pinned derived table
  and its printed-number agreement, not raw-data acquisition or conversion.
- `tests/fixtures/literature/ferretti2016/csi.csv`: 32 rows. Natural log of W.
  The adjacent provenance manifest pins the Franke Table S1 PDF, extraction,
  population and CC BY source terms; reuse that provenance without alteration.
- `data/BioSequence/Kuo2020.csv`: 7,101,196 bytes, 197,890 rows, all 9 states
  consistent with their sequences. Existing derived FL_arti data with complete
  triplicates, stored log(GFP) mean unchanged. The prior Kuo source dossier pins
  Supplemental File S3 and documents its source acquisition; no data are
  downloaded by tests. The local validation view preserves all observations,
  avoiding construction and storage of unused neighborhood edges. This is a
  test adapter for the existing get_data/data_types protocol, not a new public
  constructor or an assertion about connectivity filtering.

Input SHA-256, paper locators, exact preprocessing and evidence roles are in
`validation/cases/*.rs.*.json`. Existing cases remain byte-for-byte unchanged.
Use the established EE `validation/tests` harness and its SHA verification,
role enforcement, study/case filters and separate CI job. Nine new tests are
separate from default basic tests. The additive-model template extends this
contract with explicit coding, rank, coefficients, residuals and separate
numerator/denominator checks; it does not introduce another framework.

```sh
python -m validation.tests --literature-study Szendro2013
python -m validation.tests --literature-study SongZhang2021
python -m validation.tests --literature-case ferretti.csi.rs.v1
python -m validation.r_s_ratio
```

The public API docstring is in `graphfla/analysis/ruggedness.py`. Narrative and
API proposals remain in the shared local `.codex-local/proposals/r-s-ratio/`
directory for later incorporation into doc-epistasis by the website session.
Runtime/memory measurements are in `benchmarks/R_S_RATIO_RESULTS.md`.

The nine r/s tests take about 3.22 s with 752.03 MiB process-tree RSS on the
review machine (including independent dense fits and author replay), under
a 30 s / 1 GiB watchdog. Full suite resources are reported separately.
The archived PMC XML SHA-256 is
`360c450b09023078c896386e9e51540e31d6dddbec793016718be1379596802c`;
the pinned SD_seq notebook SHA-256 is
`416bce1df8db1bcbfa00923abdd4e504908b0460f9fed1c344ca424dd81d9efa`.
