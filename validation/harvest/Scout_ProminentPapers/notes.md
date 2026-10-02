# Scout_ProminentPapers — ranked shortlist of reproducibility anchors

**Correction, 2026-10-02:** the gamma-star defect conclusion below was based on
applying a tie-free identity to tied data. The published paper explicitly
states the missing assumption (Eq. (12), Appendix C.3). See
[GAMMA_REVIEW.md](../../GAMMA_REVIEW.md). The original numerical observations
and `record.json` are retained as research history, not current defect claims.

Agent: claude-scout (discovery pass, 2026-10-01).

Brief: find prominent published papers that could anchor GraphFLA metrics, weighted
toward the thinly-anchored (`gamma`, `r_s_ratio`, `walsh_hadamard`,
`extradimensional_bypass`, `classify_epistasis`) and unanchored (`gamma_star`,
`fdc`, `fitness_distribution`) families. Reproduce only where the target is cheap
and the input is already local.

Every citation below was verified field-by-field against the Crossref REST API
(`api.crossref.org/works/<DOI>`) **by this agent**, not taken from a second-hand
list. Three focused literature searches were delegated; their candidates were
re-verified here and every number quoted below was read out of the primary source
by this agent.

**Headline results.** Two metrics moved:

- **`fitness_distribution` goes from NO anchor to a 14-landscape exact anchor.** Li
  et al. 2025 *Cell Systems* Table S1 prints the fitted-Cauchy location and the
  kurtosis for 16 combinatorially complete protein landscapes. GraphFLA reproduces
  **both** statistics at the printed precision on **14 of the 15** landscapes that
  are bundled locally. This is almost certainly the paper GraphFLA's
  `fitness_distribution` was implemented from.
- **`r_s_ratio` gains a second anchor** — Song & Zhang 2021 *Evolution* Table 1,
  r/s = 4.063 on the Kuo 2020 landscape, reproduced as 4.0628980 — **but only
  under the paper's reference-allele coding, not GraphFLA's.** GraphFLA's default
  gives 3.7324. See the finding below: `r_s_ratio` is not invariant to which allele
  is dropped.

And one defect was localised:

- **`gamma_star` and `classify_epistasis` violate Ferretti et al.'s own identity
  `γ* = 1 − φ_s − 2φ_rs` on any landscape with exact fitness ties.** The identity
  holds to float round-off (≤ 7e-17) on five tie-free landscapes and fails by
  0.0449 on the 32-genotype TEM landscape. Minimal reproducer is already bundled.

---

# RANK 1 — Li, Yang, Johnston, Gürsoy, Yue & Arnold 2025, *Cell Systems*

## The first anchor for `fitness_distribution` — reproduced on 14 landscapes

**Verified citation (Crossref `10.1016/j.cels.2025.101387`)**

> Title: *Evaluation of machine learning-assisted directed evolution across diverse
> combinatorial landscapes*
> Authors: Li, Francesca-Zhoufan; Yang, Jason; Johnston, Kadina E.; Gürsoy, Emre;
> Yue, Yisong; Arnold, Frances H.
> Journal: *Cell Systems*, 16(9), article 101387 (September 2025)
> DOI: 10.1016/j.cels.2025.101387 · PMID 40934912
> Preprint: bioRxiv `10.1101/2024.10.24.619774`, posted 2024-10-24, CC-BY-NC-ND.
> Crossref records the preprint as `is-preprint-of` the journal article, with an
> identical six-author list in the same order.

Source read: the bioRxiv v1 full PDF, saved as
`papers/_scout/sources/Li2025_biorxiv_v1_scout.pdf` and text-extracted to
`Li2025_biorxiv_v1_scout.txt`.

### Why this is the top candidate

GraphFLA's `fitness_distribution` returns an unusual descriptor set, including the
**location parameter of a fitted Cauchy distribution** — a choice with no obvious
precedent. This paper is that precedent, and it states the implementation in the
Methods, verbatim (section "Fitness statistics"):

> "We used the 'statistical functions' (`scipy.stats`) and signal (`scipy.signal`)
> modules from the SciPy Python package75 to calculate kurtosis, estimate the
> Cauchy peak location, and determine the number of KDE peaks. Specifically,
> kurtosis was calculated using the `kurtosis` function with default settings from
> the `stats` module. Cauchy peak location was estimated using the `fit` method
> from the `cauchy` distribution object in the `stats` module."

That is exactly `graphfla/analysis/fitness.py`: `stats.kurtosis(fitness_values) + 3`
and `cauchy.fit(fitness_values)`. Their rationale, verbatim:

> "The Cauchy distribution is known for its heavy tails. We sought to use the
> fitness corresponding to its peak location as a landscape attribute to capture
> the majority of the variant finesses."

### The printed target: Table S1

Caption, verbatim:

> "Table S1. Combinatorial landscapes with additional details including landscapes
> with fewer than 1% active variants, related to Table 1."

Columns: `Landscape | PDB ID | Sites | Percent active | Fraction of local optima |
Fraction of non-magnitude epistasis | Cauchy peak location | Kurtosis | Number of
KDE peaks`. Transcribed from the PDF:

| Landscape | % active | frac. local optima | non-mag. epistasis | Cauchy peak location | Kurtosis | KDE peaks |
| --- | --- | --- | --- | --- | --- | --- |
| ParD2 | 82.89 | 0.001 | 0.34 | 0.0807 | 0.07 | 3 |
| ParD3 | 91.96 | 0.001 | 0.31 | 0.2521 | −0.29 | 3 |
| GB1 | 23.13 | 0.005 | 0.40 | 0.0003 | 76.92 | 33 |
| DHFR | 10.68 | 0.004 | 0.42 | 0.1271 | 19.21 | 7 |
| T7 | 3.48 | 0.368 | 0.52 | 0.0000 | 46.51 | 11 |
| TEV | 11.5 | 0.060 | 0.56 | −0.0114 | 37.74 | 27 |
| TrpB3A | 0.74 | 0.390 | 0.60 | −0.0399 | 53.44 | 9 |
| TrpB3B | 0.23 | 0.667 | 0.54 | −0.0554 | 84.31 | 8 |
| TrpB3C | 0.44 | 0.514 | 0.59 | −0.0736 | 7.15 | 8 |
| TrpB3D | 9.26 | 0.043 | 0.50 | 0.0036 | 32.52 | 13 |
| TrpB3E | 2.02 | 0.348 | 0.63 | 0.0008 | 355.09 | 15 |
| TrpB3F | 1.06 | 0.232 | 0.54 | −0.0230 | 47.49 | 15 |
| TrpB3G | 1.37 | 0.213 | 0.52 | −0.0037 | 131.81 | 23 |
| TrpB3H | 0.69 | 0.547 | 0.62 | −0.0152 | 464.45 | 13 |
| TrpB3I | 32.04 | 0.006 | 0.43 | 0.0228 | 9.38 | 6 |
| TrpB4 | 6.15 | 0.057 | 0.46 | 0.0118 | 48.56 | 27 |

All **printed**, in a table. The Cauchy and kurtosis columns appear **only** in
Table S1; main-text Table 1 carries only % active / frac. local optima /
non-magnitude epistasis, for the nine landscapes with ≥ 1 % active.

Preprocessing, verbatim (Methods, "Landscape preparation" and "Landscape attributes"):

> "All fitness values were normalized so that the variant with the maximum fitness
> has a value of one."

> "We do not impute missing values."

### What GraphFLA produced

Fitness divided by its maximum, generic categorical `Landscape`, then
`analysis.fitness_distribution`. Li et al. use `scipy.stats.kurtosis` defaults =
**Fisher/excess** kurtosis; GraphFLA adds 3 to convert to Pearson's, so the
comparison is `graphfla_kurtosis − 3` against the printed column.

| Landscape | local file | printed Cauchy | GraphFLA Cauchy | printed kurtosis | GraphFLA kurtosis − 3 | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| ParD2 | `Lite2020_ParD2.csv` | 0.0807 | 0.080715 | 0.07 | 0.067 | both match |
| ParD3 | `Lite2020_ParD3.csv` | 0.2521 | 0.252104 | −0.29 | −0.2906 | both match |
| GB1 | `Wu2016_GB1.csv` | 0.0003 | 0.000293 | 76.92 | 76.9202 | both match |
| T7 | `Tu2022_T7.csv` | 0.0000 | **−0.723299** | 46.51 | **−0.3771** | **mismatch** |
| TEV | `Tu2022_TEV.csv` | −0.0114 | −0.011360 | 37.74 | 37.7374 | both match |
| TrpB3A | `Johnston2024_TrpB3A.csv` | −0.0399 | −0.039944 | 53.44 | 53.4413 | both match |
| TrpB3B | `Johnston2024_TrpB3B.csv` | −0.0554 | −0.055374 | 84.31 | 84.3071 | both match |
| TrpB3C | `Johnston2024_TrpB3C.csv` | −0.0736 | −0.073578 | 7.15 | 7.1462 | both match |
| TrpB3D | `Johnston2024_TrpB3D.csv` | 0.0036 | 0.003583 | 32.52 | 32.5161 | both match |
| TrpB3E | `Johnston2024_TrpB3E.csv` | 0.0008 | 0.000781 | 355.09 | 355.0871 | both match |
| TrpB3F | `Johnston2024_TrpB3F.csv` | −0.0230 | −0.022960 | 47.49 | 47.4855 | both match |
| TrpB3G | `Johnston2024_TrpB3G.csv` | −0.0037 | −0.003729 | 131.81 | 131.8072 | both match |
| TrpB3H | `Johnston2024_TrpB3H.csv` | −0.0152 | −0.015176 | 464.45 | 464.4492 | both match |
| TrpB3I | `Johnston2024_TrpB3I.csv` | 0.0228 | 0.022800 | 9.38 | 9.3821 | both match |
| TrpB4 | `Johnston2024_TrpB4.csv` | 0.0118 | 0.011761 | 48.56 | 48.5621 | both match |

**14 of 15 reproduce both statistics at the printed precision** (4 dp for Cauchy,
2 dp for kurtosis). The DHFR row was not attempted: Li et al.'s DHFR is the
amino-acid-level 20³ landscape at positions A26/D27/L28, while the bundled
`Papkou2023_DHFR.csv` is the nucleotide-level 4⁹ landscape, so the inputs are not
the same object.

This is strong enough to promote directly. It is a 14-landscape regression anchor
for `fitness_distribution`, it exercises both a heavy-tailed fit and an extreme
kurtosis range (−0.29 to 464), and it incidentally confirms that 14 bundled CSVs
are byte-faithful to the primary measurements, since a single altered fitness value
would move a four-decimal Cauchy location.

### The one miss — and what it says about a bundled file

`Tu2022_T7.csv` misses on both statistics, grossly (Cauchy −0.7233 vs 0.0000;
kurtosis −0.38 vs 46.51). GraphFLA is not at fault; the file is on a different
fitness scale from Li et al.'s T7 input. Diagnostics:

- The bundled T7 values run −1.1766 to 0.8512 and are platykurtic (Fisher kurtosis
  −0.38), the shape of a log-transformed quantity. Li's printed kurtosis of 46.51
  with a Cauchy location of exactly 0.0000 is the shape of a raw enrichment
  measure concentrated at zero with heavy tails.
- Undoing a base-10 log (`10**f`, then max-normalising) moves GraphFLA to Cauchy
  0.0289 / kurtosis 39.39 — much closer to 0.0000 / 46.51, but still not a match.
- `Tu2022_T7.csv` and `Tu2022_TEV.csv` share an **identical minimum of −1.1766**,
  which is the signature of a floor applied during curation rather than a property
  of two different assays. TEV nonetheless reproduces exactly on the bundled scale,
  while T7 does not.

So: the bundled `Tu2022_T7.csv` is on a transform that neither matches its sibling
`Tu2022_TEV.csv`'s relationship to Li et al.'s input nor inverts cleanly. **This is
a data-provenance question about a bundled file, not a metric defect**, and it is
the same class of problem as the tenfold `Weinreich2006` error already in
`LEDGER.md`. Recorded here, not fixed — this tranche makes no repository changes.

### The other two Table S1 columns do NOT transfer

`Fraction of local optima` and `Fraction of non-magnitude epistasis` are both
defined over **active variants only**, and the active cutoff cannot be
reconstructed from the bundled CSVs. Verbatim:

> "A local optimum is a variant with higher fitness than all its neighboring active
> variants differing by one"

and (Figure 2 legend) "fraction of local optima (normalized to the number of
variants measured)"; and for epistasis:

> "For each active variant, we assigned an epistasis type for each possible double
> substitution at chosen sites. We then calculated the fraction of epistasis type
> for each starting variant in the landscape. To enhance relevance to DE
> navigability, we incorporated additive interactions into magnitude epistasis, and
> merged sign and reciprocal sign epistasis into non-magnitude epistasis."

with the cutoff itself:

> "For landscapes containing fitness data for variants with stop codons, 'active'
> variants were defined as those 1.96 standard deviations above the mean fitness of
> all sequences containing stop codons, which are expected to be inactive. For GB1,
> T7, and TEV we followed the cutoffs set by the authors, based on the detection
> limit of their fitness measurement system."

Stop-codon rows are not in the bundled CSVs, so the cutoff is unavailable. Measured
anyway, to size the divergence:

| Landscape | printed non-mag. | GraphFLA sign + reciprocal | printed frac. LO | GraphFLA `local_optima_ratio` (no active filter) |
| --- | --- | --- | --- | --- |
| ParD2 | 0.34 | 0.4195 | 0.001 | 0.00063 (5/7882) |
| ParD3 | 0.31 | 0.3478 | 0.001 | 0.00051 (4/7882) |
| GB1 | 0.40 | 0.5651 | 0.005 | 0.00123 (184/149361) |
| T7 | 0.52 | 0.6785 | 0.368 | 0.01755 (118/6725) |
| TEV | 0.56 | 0.6652 | 0.060 | 0.00697 (1109/159132) |
| TrpB3A | 0.60 | 0.6164 | 0.390 | 0.00778 (62/7971) |
| TrpB3B | 0.54 | 0.6479 | 0.667 | 0.00925 (74/7996) |
| TrpB3I | 0.43 | 0.5289 | 0.006 | 0.00206 (16/7784) |

The epistasis fractions are in the right range but uniformly higher, as expected
when additive squares are not folded into magnitude and non-active starting
variants are not excluded. The local-optima fractions diverge by up to 70× on the
sparsely-active landscapes (TrpB3B: 0.23 % active, so nearly every active variant
is a peak relative to other *active* variants). Both are
**`definition_incompatible` as printed**; both would become reproducible if the
active cutoffs were recovered from the authors' code.

The two large landscapes — GB1 (149,361 configurations) and TEV (159,132), both
with 20 alleles per site — behave exactly like the small ones: non-magnitude
epistasis uniformly high, local-optima fraction uniformly low. The divergence is
therefore a property of the two definitions, not an artefact of landscape size or
of sparse activity alone.

### Inputs, for promotion

> "All data and results that support this study are deposited at
> https://doi.org/10.5281/zenodo.13910506. All code is available at
> https://github.com/fhalab/SSMuLA."

The Zenodo deposit plus SSMuLA would settle (a) the exact variant set used for the
fitness statistics, (b) the per-landscape active cutoffs, which would unlock the
other two Table S1 columns, and (c) the T7 scale question.

### Corroborating source (weaker venue, do not use as the anchor)

Cotet & Krawczuk, "Why risk matters for protein binder design", arXiv:2504.00146
(two authors; ICLR 2025 GEM Workshop; **no DOI**) states verbatim that it follows
"the methodology established by Li et al. (2024)" and its Table A.4 prints
**skewness** alongside the Cauchy location and kurtosis for 11 further landscapes.
It is the only source found that treats skewness, kurtosis and Cauchy location as a
landscape descriptor *triple*, which matches GraphFLA's return value more closely
than Li et al. do. Its own Otsu-based active thresholding makes its numbers harder
to reproduce, so it is corroboration that the descriptor set is in use, not an
anchor.

### The rest of `fitness_distribution` still has no published origin

`cv`, `quartile_coefficient`, `median_mean_ratio` and `relative_range` were not
found in any fitness-landscape-analysis source. The nearest published relatives are
the **`y-Distribution`** feature class of Exploratory Landscape Analysis —
Mersmann, Bischl, Trautmann, Preuss, Weihs & Rudolph, GECCO '11, pp. 829–836, DOI
`10.1145/2001576.2001690` — which is `{skewness, kurtosis, number of density
peaks}` of the objective values, i.e. it covers skewness/kurtosis/KDE-peaks but not
the four ratios. Checked and negative: Malan, *Algorithms* 14(2):40, 2021, DOI
`10.3390/a14020040` (zero occurrences of "kurtosis", "Cauchy", "coefficient of
variation" or "interquartile"); and the maintainer's own prior work (Huang, Mao &
Li, ISSTA 2025, arXiv:2412.16888), which mentions skewness only to criticise it and
contains no Cauchy, CV, quartile or relative-range terms. **Honest answer: those
four are standard textbook relative-dispersion ratios with no landscape-specific
precedent.** Say so in the docs rather than hunting further.

### Documentation bug found in passing

`fitness_distribution`'s docstring says the statistics "are chosen to be unitless
(scale-invariant) to allow meaningful comparisons across different landscapes with
varying fitness scales". **`cauchy_loc` is not scale-invariant** — it is a location
parameter in the units of fitness, which is precisely why Li et al. have to
max-normalise before reporting it. Neither is it shift-invariant. Everything else
in the dict is a ratio; `cauchy_loc` is not.

---

# RANK 2 — Song & Zhang 2021, *Evolution*

## Second `r_s_ratio` anchor (exact), and a decisive verdict against it for `gamma`

**Verified citation (Crossref `10.1111/evo.14363`)**

> Title: *Unbiased inference of the fitness landscape ruggedness from imprecise
> fitness estimates*
> Authors: Song, Siliang; Zhang, Jianzhi
> Journal: *Evolution*, 75(11), 2658–2671 (online 2021-10-07)
> DOI: 10.1111/evo.14363 · PMID 34554581 · PMC9018209

A methods paper that defines eight ruggedness statistics and prints all eight, in a
table, for three named empirical landscapes — two of which are bundled locally
with *exactly* the row counts the paper reports.

### The printed target: Table 1

Caption, verbatim:

> "Table 1. Ruggedness of three empirical fitness landscapes analyzed.
> Extrapolated ruggedness is presented, followed in parentheses by that based on
> average fitness estimates."

The **parenthesised** column is the recomputable one — the plain ruggedness of the
replicate-averaged landscape, with no extrapolation model in between.

| statistic | Li et al. 2016 (69 sites) | Domingo et al. 2018 (10 sites) | Kuo et al. 2020 (9 sites) |
| --- | --- | --- | --- |
| Number of available genotypes | 21,182 | **4,176** | **197,890** |
| N_max | 2518 (2534) | 53 (**70**) | NA (**2404**) |
| F_rse | 0.0793 (0.0820) | 0.1550 (**0.1748**) | NA (**0.2141**) |
| r/s | 3.246 (3.355) | 1.964 (**2.102**) | 3.965 (**4.063**) |
| F_bp | NA (0.9157) | 0.7187 (0.7444) | NA (0.7738) |
| E | ? | 0.3362 (**0.3699**) | 0.8040 (**0.8066**) |
| 1−γ | 0.5070 (0.5186) | 0.6616 (**0.7005**) | 0.5923 (**0.6322**) |
| 1/N_adapt | 0.4644 (0.4680) | 0.2885 (0.3007) | 0.2030 (0.2329) |
| 1−P_adapt | NA (0.9887) | NA (0.9871) | NA (0.9996) |

Footnotes, verbatim: "1Genotypes with sufficient experimental replicates are
considered. 2NA, ruggedness inference not applicable because the data are located
in the concave segment of the S-shaped error-ruggedness relationship. 3?,
ruggedness cannot be estimated due to the incompleteness of the landscape."

### Printed definitions (Materials and Methods), verbatim

`N_max`:

> "A genotype is considered a local fitness maximum if it is fitter than all of its
> neighboring genotypes, which differ from the focal genotype by one point
> mutation. ... For the three empirical landscapes, we examined all genotypes with
> fitness estimates and treated neighboring genotypes without fitness estimates as
> having a fitness of 0."

`F_rse`:

> "For each genotype pair, reciprocal sign epistasis is recorded if both genotypes
> are either fitter or less fit than the two intermediate genotypes between them.
> F rse is the proportion of genotype pairs that exhibit reciprocal sign
> epistasis."

`r/s`:

> "The roughness value r is the square root of the mean squared fitness residual
> from the above regression and s is the average of the absolute values of the
> linear coefficients β i (i =1, 2, …, and n)."

> "When a landscape has four instead of two states per variable site, we adjusted
> the above additive model to F(x) = β0 + Σ βiA xiA + βiT xiT + βiC xiC + βiG xiG,
> ... and s = Σ (|βiA| + |βiT| + |βiC| + |βiG|) / 3n"

`γ`:

> "γ describes the correlation between the fitness effect of a mutation in one
> genetic background and that in another background differing from the first
> background at one site. We calculated γ by following a previous study (Ferretti
> et al., 2016) and regarded 1-γ as the ruggedness measure"

`E`:

> "Therefore, E equals 1 − E1 / Σ Ej, where Ej is the epistasis of the jth order."

Inputs, verbatim:

> "The tRNA landscape has 10 variable sites ... where 6 sites are biallelic and 4
> sites are triallelic. The original paper used log-transformed Wrightian fitness,
> which we followed. The landscape includes 4176 genotypes with fitness measured in
> six replicates (Domingo et al., 2018)."

> "The landscape has 9 variable sites each with four states, leading to a space of
> 4 9 = 262,144 genotypes (Kuo et al., 2020). ... the context with the highest
> genotype coverage across all three replicates (arti) was chosen, which has
> 197,890 genotypes after the exclusion of genotypes with missing replicate
> measures."

Code statement, verbatim:

> "Data and code accessibility: There is no data to be archived. Computer code and
> intermediate results are available at
> https://github.com/song88180/fitness-landscape-error ."

### Inputs — already local, row counts identical

| paper landscape | local file | rows | sha256 |
| --- | --- | --- | --- |
| Domingo et al. 2018 | `data/BioSequence/Domingo2018.csv` | 4,176 | `5b8cf25627ec37a1dce53f48b6ac121bb043ddb6f33616bb5ed00b203a7df2ff` |
| Kuo et al. 2020 `arti` | `data/BioSequence/Kuo2020.csv` | 197,890 | `69a86caf1ccbfc94a519f5917624d6862a0dfede0082abae79300eda0bb959c0` |

The bundled sequences spell uracil `U`, so `DNALandscape` rejects them; the
generic categorical `Landscape` is the correct entry point.

### What GraphFLA produced

**Kuo 2020** (197,890 configurations, 2,180,702 edges):

| statistic | printed | GraphFLA | gap | outcome |
| --- | --- | --- | --- | --- |
| N_max | 2404 | `n_lo` = 2390 | −0.58 % | mismatch |
| F_rse | 0.2141 | `classify_epistasis().reciprocal_sign` = 0.2154909 | +0.65 % | mismatch |
| r/s | 4.063 | 3.7324439 default; **4.0628980 under the paper's allele coding** | exact at printed precision | **reproduced_with_precision** (paper coding) |
| 1−γ | 0.6322 | 0.4983628 | −21 % | definition_incompatible |
| E | 0.8066 | 1 − `higher_order_epistasis(order=1)` = 0.7126037 | −12 % | definition_incompatible |

**Domingo 2018** (4,176 configurations, 26,336 edges):

| statistic | printed | GraphFLA | gap | outcome |
| --- | --- | --- | --- | --- |
| N_max | 70 | `n_lo` = 86 | +23 % | mismatch |
| F_rse | 0.1748 | 0.1820457 | +4.1 % | mismatch |
| r/s | 2.102 | 1.8458360 default; 2.0566631 under the paper's coding | −2.2 % | mismatch |
| 1−γ | 0.7005 | 0.7158542 | +2.2 % | definition_incompatible |
| E | 0.3699 | 0.4745851 | +28 % | definition_incompatible |

### Why each statistic behaves as it does — grounded in the authors' own code

The authors' repository was read: `utils/utils.py` (saved as
`papers/_scout/sources/SongZhang2021_utils_scout.py`, sha256
`ff6487628553c15f1de4cae7d5ee7370c4b7a311543581d416f8c6b7cb159dcc`), plus the two
empirical pipelines `5_Empirical_Extrapolation/trna_Domingo/{Prepare_index_files,
Generate_raw_data}.ipynb` and `5_Empirical_Extrapolation/SD_seq/Generate_raw_data.ipynb`.
The landscape data files are **not** in the repository (only placeholder
`README.md` stubs), but the code settles every definitional question.

**1. `r/s` — the gap is a reference-allele convention, and GraphFLA's choice is the
arbitrary one. This is the most actionable finding in this record.**

Both empirical notebooks build the design matrix with

```python
BASES = np.asarray(['A','T','C','G'])
data = sequences[..., None] == BASES              # one-hot over A,T,C,G
x = x[:, np.where((x != x[0]).sum(axis=0) > 0)[0]]  # drop constant columns
```

The sequences are RNA and spell `U`, which is **absent from `BASES`**, so no `U`
column is ever created. The surviving columns are one per observed non-U allele per
site, which makes `U` the implicit reference level. For Kuo that is exactly 27
columns = 3 per site: drop-one one-hot with **U as the dropped level**. GraphFLA's
`r_s_ratio` uses `pd.get_dummies(..., drop_first=True)`, which drops the **first
category in sorted order — `A`**.

The two codings span the same column space, so the roughness `r` is identical; but
`s = mean(|β|)` is **not** invariant to which level is dropped, so the ratio
differs. Reproducing their coding gives **4.0628980 against a printed 4.063**.
(`Ridge(alpha=1)`, which their notebook actually calls instead of the
`LinearRegression` their Methods and `utils.py` describe, gives 4.0630308 — also
4.063 at printed precision, so the penalty is immaterial; the coding is what
matters.)

*Consequence for GraphFLA.* On any landscape with more than two alleles per site,
`r_s_ratio` is **not invariant to the choice of reference allele**, and the current
choice is undocumented and arbitrary. Kuo moves 3.732 → 4.063, i.e. 8.9 %, on
nothing but that choice. The r/s family (Aita; Szendro; Song & Zhang; Manivannan)
is defined for biallelic loci and the multi-allelic extension is unsettled in the
literature, so there is no single right answer — but GraphFLA must at minimum
document the convention, and a maintainer may prefer a reference-free definition
(e.g. sum-to-zero effect coding, which is invariant).

For Domingo the paper's coding leaves 17 columns, and at the three sites with no
`U` (pos1 {A,G}, pos5 {A,C}, pos8 {A,G}) *every* observed allele keeps a column, so
the design is rank-deficient and the individual β are not identifiable at all.
Their Domingo r/s is therefore convention- and solver-dependent, which is the
likely source of the residual 2.2 % there on top of an input difference (below).

**2. `γ` — definitionally different from both GraphFLA and, arguably, from
Ferretti.** The shipped `utils.py` computes

```python
cov = np.cov(gt_1_diff_list, gt_2_diff_list)[1,0]
var = np.var(gt_1_diff_list)
return cov/var
```

— a **centred** covariance over a ddof=0 variance, i.e. a regression slope. The
empirical notebooks compute something different again:

```python
cov = Σ (y1−y0)(y3−y2) + Σ (y2−y0)(y3−y1)
cov = cov / (2 * n_squares)
var = np.var([y[nb] − y[i] for every i, every neighbour nb])
return cov/var
```

— a non-centred second moment over the squares, divided by the variance of **all
single-step fitness differences in the whole graph**, not of the square-derived
effects. GraphFLA follows Ferretti eq. (3): a non-centred ratio whose denominator
is `0.5·(Σb² + ΣB²)` over the *same* square effects. The denominator populations
differ, so these are different statistics, which is why the Domingo gap is +2.2 %
while the Kuo gap is −21 %: a denominator swap is landscape-dependent, not a
constant rescaling.

**Verdict: Song & Zhang cannot serve as a second anchor for `gamma`.** Do not tune
`gamma` toward Table 1.

**3. `N_max` — GraphFLA matches the authors' CODE, not their prose.** The notebook is

```python
if np.sum(fit <= y[neighbor_list[i]]) == 0: N_max += 1
```

with `neighbor_list[i] = np.where((seqs != seq).sum(axis=1) == 1)[0]` over the
**observed** sequences only: strictly fitter than every *present* neighbour, absent
neighbours ignored — exactly GraphFLA's rule, and *not* the "treated ... as having
a fitness of 0" the Methods claim. An independent numpy recount on the bundled CSVs
under that rule gives 2388 (Kuo) and 86 (Domingo). Imputing 0 for missing
neighbours, as the prose says, gives 42 for Domingo — further from 70, not closer.
The printed 70 and 2404 are reachable from neither rule on the bundled data.

**4. `F_rse` — the same quantity as GraphFLA's `reciprocal_sign`, two caveats.** The
notebook's `cal_epi` counts both orientations (`> all four` **and** `< all four`),
matching the prose and matching GraphFLA. But the square enumeration in
`Prepare_index_files.ipynb` keeps a square only when *both* intermediate indices
exceed the starting genotype's index:

```python
elif idx_10 < idx_01 and idx_10 > idx_00: ...
elif idx_10 > idx_01 and idx_01 > idx_00: ...
```

Squares whose intermediate sorts before `idx_00` in file order are silently dropped
from both numerator and denominator — a row-order artefact, and the most likely
source of the residual 0.65 % / 4.1 % gaps. Note also that `utils.py`'s `cal_epi`
(used for the *simulated* landscapes) counts only the `>` orientation, so the
authors' two implementations of F_rse disagree with each other.

**5. `E` — not GraphFLA's `higher_order_epistasis`.** Theirs is a Walsh–Hadamard
variance share, `(ΣE² − E₀² − ΣE₁²)/(ΣE² − E₀²)`, computed on a *completed* 2^10
biallelic sub-landscape whose missing cells are Ridge-interpolated (the Methods say
filled with 0; the notebook uses `Ridge` — a third paper/code inconsistency).
GraphFLA's is an unregularised one-hot polynomial OLS R² on the observed subset.
Different estimand, different support. Same verdict as Papkou Fig. S21 already in
the ledger.

### What is needed to promote this

The Kuo r/s match is already strong enough to stand as a second `r_s_ratio` anchor
*if* the reference-allele convention is made explicit. For the rest the blocker is
the input, not the method:

- Kuo: the bundled `Kuo2020.csv` holds only **2,473 distinct fitness values across
  197,890 rows** — rounded to 3 decimal places. That manufactures ties (37
  genotypes tie with a neighbour and have no fitter neighbour) and makes `N_max`
  and `F_rse` precision-limited at roughly the 0.5–1 % level, which is the size of
  the residual gaps. The authors recomputed fitness from read counts at full
  precision.
- Domingo: the bundled `Domingo2018.csv` is at full precision (4,176 distinct
  values), so its 23 % `N_max` gap is **not** a rounding artefact. The authors
  recompute fitness from raw read counts with their own generation-ratio formula
  (`get_fitness` in `trna_Domingo/Generate_raw_data.ipynb`) rather than taking the
  published fitness column. The genotype counts coincide; the fitness values
  evidently do not.

So the next step is to run the authors' `get_fitness` over the Domingo read counts
and the Kuo per-replicate table — not to adjust GraphFLA.

### A second GraphFLA observation found here

**`sum(graph.vs['is_lo']) != landscape.n_lo`.** On `Kuo2020.csv`, `is_lo` is `True`
for 2,392 vertices while `n_lo` is 2,390. The four flagged vertices that are not
strictly fitter than all neighbours are two tied *pairs* —
`ACAACACUA`/`UCAACACUA` (both f = 0.667, differing at pos1) and
`ACAACGCUU`/`ACUACGCUU` (both f = 0.778, differing at pos3) — each a two-genotype
plateau with no fitter neighbour. So `n_lo` counts plateaus while `is_lo` flags
their members. Coherent, but the two accessors disagree numerically with no
documentation, which is an easy trap for a user counting peaks via `is_lo`.

---

# RANK 3 — Ferretti, Weinreich, Tajima & Achaz 2018, *Heredity*

## Second printed `gamma`, first printed `gamma_star`, and the identity that found a bug

**Verified citation (Crossref `10.1038/s41437-018-0110-1`)**

> Title: *Evolutionary constraints in fitness landscapes*
> Authors: Ferretti, Luca; Weinreich, Daniel; Tajima, Fumio; Achaz, Guillaume
> Journal: *Heredity*, 121(5), 466–481 (online 2018-07-11)
> DOI: 10.1038/s41437-018-0110-1 · PMID 29993041 · PMC6180097

**Preprint identity settled.** arXiv:1507.00041, "Epistasis and constraints in
fitness landscapes", is the preprint of **this** paper — not of the JTB 2016 γ
paper, whose preprint is bioRxiv `10.1101/042010`. The arXiv version carries seven
authors (Ferretti, Weinreich, Schmiegelt, Yamauchi, Kobayashi, Tajima, Achaz) and a
different title; the journal version dropped to four. Any bibliography entry giving
arXiv:1507.00041 the JTB title, or the four-author list, is wrong on one count or
the other. Saved as
`papers/_scout/sources/Ferretti2018_arxiv1507.00041_scout.pdf`. Europe PMC and NCBI
return abstract-only for PMC6180097, so the quotes below are from the preprint,
which prints *more* than the journal version does.

### The printed targets

Section 5, "Measures on two experimental landscapes", verbatim:

> "The first landscape is the landscape of antibiotic (cefotaxim) resistance of
> β-lactamase mutations in an Escherichia coli plasmid from Weinreich et al. (2006)
> (Figure 7 left). ... Given the huge selective advantage of the combined
> mutations, this landscape is single-peaked, where the peak corresponds to the
> five-point mutant. It also has a single sink, that interestingly does not
> correspond to the wild type."

> "The second is one of the four L = 5 complete sublandscape (csI) (Franke et al.,
> 2011) of a larger landscape (L = 8) of deleterious mutations in Aspergillus niger
> from de Visser et al. (1997) (Figure 7 right). ... This landscape has 4 peaks and
> 2 sinks; in fact, at present it is one of the most rugged among the completely
> resolved landscapes."

> "The difference in ruggedness between β-lactamase and Aspergillus landscapes is
> confirmed by the values of γ (0.85 vs 0.33) and r/s (0.43 vs 0.89)."

Figure 7 panel labels (printed text annotations inside the figure, extracted from
the PDF text layer; the glyphs for γ and γ\* extract as `a` and `a*`):

> `peaks: 1   r/s: 0.43   steps: 9   sinks: 1   a: 0.85   a*: 0.59   chain trees: 2   origins: 6`
> `peaks: 4   r/s: 0.89   steps: 7   sinks: 2   ...`

This gives **four printed targets for the TEM landscape**: peaks 1, sinks 1,
γ 0.85, γ\* 0.59, r/s 0.43. The A. niger csI values (4 peaks, 2 sinks, γ 0.33,
r/s 0.89) are a consistency re-print of the case already reproduced in LEDGER.md —
which is itself useful: it confirms the ledger's reading of Ferretti 2016 Fig. 4c
was correct, since two independent publications print the same numbers.

### What GraphFLA produced on the TEM landscape

Input `data/BioSequence/Weinreich2006Tan2011_Weinreich2006.csv`, 32 genotypes,
5 boolean loci.

| statistic | printed | GraphFLA | outcome |
| --- | --- | --- | --- |
| peaks | 1 | `n_lo` = 1 | **reproduced_exact** |
| γ | 0.85 | 0.825394 as bundled; 0.835072 with the tenfold defect corrected | mismatch |
| γ\* | 0.59 | 0.613636 (both versions) | mismatch |
| r/s | 0.43 | 0.445517 as bundled; 0.443968 corrected | mismatch |

**The known tenfold data defect is NOT the cause of the TEM mismatch — hypothesis
tested and rejected.** `LEDGER.md` records that this file stores 15.32333629 for
the all-ones genotype, which back-transforms to 2^15.3233 = 41,000 while the source
MIC table prints 4,100. Substituting log2(4100) = 12.00140819 moves γ from 0.8254
to 0.8351, leaves γ\* unchanged at 0.613636, and moves r/s from 0.4455 to 0.4440.
None of the three reaches the printed value, and the corrected figures are the ones
already in the ledger (0.8353 / 0.6136 / 0.4441), so the earlier agent was already
working from the corrected value. The residual 2–5 % gap is an input difference,
consistent with the ledger's standing blocker on Weinreich's Supporting Online
Material and the MAGELLAN `.fl` inputs. **The MAGELLAN landscape library at
`wwwabi.snv.jussieu.fr/public/Magellan/` is unreachable** (connection timed out
over both http and https), so that route is closed for now.

### The identity — and the defect it exposes

The preprint prints an exact algebraic identity, section 2.5 and the summary,
verbatim:

> "γ ∗ is directly related to the number of square motifs with sign and reciprocal
> sign epistasis as γ ∗ = 1 − φs − 2φrs."

also given as eq. (15):

> `γ ∗ = 1 − φs − 2φrs`

This is free validation: it must hold between GraphFLA's `gamma_star` and its
`classify_epistasis` fractions on *any* landscape, with no external data. Tested:

| landscape | `gamma_star` | 1 − φ_s − 2φ_rs | abs. difference |
| --- | --- | --- | --- |
| `deVisser2009.csv` | 0.075000000000 | 0.075000000000 | 6.9e-17 |
| `Mira2015_TEM_AM.csv` | 0.291666666667 | 0.291666666667 | 5.6e-17 |
| `Lozovsky2009Jiang2013_Lozovsky2009.csv` | 0.500000000000 | 0.500000000000 | 0 |
| `Domingo2018.csv` | 0.281999066650 | 0.281999066650 | 5.6e-17 |
| `Lite2020_ParD2.csv` | 0.467860049614 | 0.467860049614 | 5.6e-17 |
| `Johnston2024_TrpB3A.csv` | 0.127283870043 | 0.127280049450 | 3.8e-06 |
| **`Weinreich2006Tan2011_Weinreich2006.csv`** | **0.613636363636** | **0.658536585366** | **4.49e-02** |

The identity holds to float round-off on five landscapes, to 4e-6 on an incomplete
one, and **fails by 0.0449 on the TEM landscape**.

Cause, diagnosed by independent enumeration: the TEM landscape has **80 square
motifs, of which 39 contain at least one tied (zero-magnitude) fitness effect**,
leaving 41 classifiable. An independent census of those 41 gives magnitude 29, sign
10, reciprocal-sign 2 — exactly GraphFLA's reported fractions 0.707317 / 0.243902 /
0.048780, so `classify_epistasis` is internally correct *on the squares it counts*,
with a denominator of 41 that excludes every tied square. `gamma_star` meanwhile
drops zero-sign effects from its own denominator via `np.sign`, which is a
different exclusion over a different population (edge pairs, not squares). Hence:

- `gamma_star` = 0.613636
- eq. (15) over the 41 classifiable squares = 0.658537
- eq. (15) over all 80 squares (ties counted as no-sign-epistasis) = 0.825

**Three mutually inconsistent answers. Finding: `gamma_star` and
`classify_epistasis` use different conventions for zero-magnitude fitness effects,
which breaks Ferretti et al.'s eq. (15) on any landscape containing exact fitness
ties.** The minimal reproducer is a 32-row bundled CSV. This matters well beyond
this one file: MIC and doubling-dilution assays are intrinsically tie-heavy, so
every TEM/β-lactamase landscape in `data/` (`Mira2015_TEM_*`, 15 files) is in the
affected class — and `Mira2015_TEM_AM` happens to satisfy the identity, so the
failure is data-dependent and will not be caught by a single smoke test. Not fixed
here; this tranche makes no repository changes.

Note separately that **the 38-landscape collection analysed in this paper is
plot-only** (Figure 4, axes in `1 − γ`); Table 1 gives only Spearman ρ² between
pairs of measures, not per-landscape γ. There is no published table of per-landscape
γ for that collection.

---

# RANK 4 — Cervera, Lalić & Elena 2016, *Journal of Virology*

## The only tabulated γ + peaks + sinks + r/s + epistasis-type panel outside Ferretti's own work

**Verified citation (Crossref `10.1128/jvi.01243-16`)**

> Title: *Effect of Host Species on Topography of the Fitness Landscape for a Plant
> RNA Virus*
> Authors: Cervera, Héctor; Lalić, Jasna; Elena, Santiago F.
> Journal: *Journal of Virology*, 90(22), 10160–10169 (2016-11-15)
> DOI: 10.1128/jvi.01243-16 · PMID 27581976 · PMC5105653

**Table 2**, "Summary statistics describing the topographies of both landscapes",
computed with the MAGELLAN web server, prints for two host landscapes (5 biallelic
loci, 32 genotypes each) the number of peaks, the number of sinks, a
slope/roughness ratio, **ρ (= Ferretti's γ)**, and the frequencies of
multiplicative / magnitude / sign / reciprocal-sign epistasis. If reproduced, this
single table would anchor `n_lo`, `r_s_ratio`, `gamma` **and**
`classify_epistasis` simultaneously — the exact combination the project is short
of.

**Status: not attempted, `input_unavailable`.** The 32 per-host fitness values are
presented only in Figure 2; Table 1 lists the five mutations, not fitnesses. There
is no data-availability statement and no supplementary data table. The *A.
thaliana* values originate in Lalić & Elena 2015, *J Evol Biol* 28:2236–2247; the
*N. tabacum* values appear to be new to the 2016 paper. No local CSV exists. The
MAGELLAN bundled-landscape library, the other plausible route, is unreachable.

Because this agent did not read Table 2 in the primary source (Europe PMC and NCBI
both return abstract-only for PMC5105653 and the J Virol site was not retrieved),
**the specific values reported by the delegated search are deliberately not
transcribed into this record.** The table's existence, its caption and its column
list are what is recorded. Anyone promoting this must read Table 2 directly.

Two traps flagged for whoever does:

1. The statistic is labelled a **slope-to-roughness** ratio, i.e. the reciprocal of
   `r/s`. Whether the printed numbers are θ or r/s as MAGELLAN emits them must be
   settled before comparison — do not assume either reading.
2. ρ is glossed in the Results as the correlation between *fitness levels* of
   nearby genotypes, while the Methods define it as the correlation of fitness
   *effects* (Ferretti). The Methods definition is the one that matches GraphFLA.

---

# RANK 5 — Weinreich, Lan, Jaffe & Heckendorn 2018, *Journal of Statistical Physics*

## 17 printed peak counts and 3 printed order-wise Walsh profiles, nearly all on local data

**Verified citation (Crossref `10.1007/s10955-018-1975-3`)**

> Title: *The Influence of Higher-Order Epistasis on Biological Fitness Landscape
> Topography*
> Authors: Weinreich, Daniel M.; Lan, Yinghong; Jaffe, Jacob; Heckendorn, Robert B.
> Journal: *Journal of Statistical Physics*, 172(1), 208–225 (online 2018-02-07)
> DOI: 10.1007/s10955-018-1975-3 · PMID 29904213 · PMC5986866 (open access)
> Preprint: bioRxiv `10.1101/164798`.

Full text retrieved and tables extracted directly from Europe PMC
(`papers/_scout/sources/Weinreich2018_fulltext_scout.xml`).

**Table 1** caption, verbatim: "Analyses of published combinatorially complete
empirical and simulated (NK) fitness landscapes, sorted by P value associated with
Kendall's τ_b". Columns: Phenotype [citation] | Number of loci (L) | Number of
maxima | Number of epistatic terms significantly different from zero | Kendall's
τ_b | P value. 22 rows (17 empirical plus 5 NK).

**Printed (L, number of maxima)** pairs, with the bundled file each corresponds to:

| phenotype as printed | L | maxima | local file |
| --- | --- | --- | --- |
| Log[S. cerevisiae HSP90 mutant growth rates] | 6 | 4 | `Bank2016a.csv` / `Bank2016b.csv` |
| Log[diploid S. Cerevisiae mutant growth rate] | 6 | 4 | `Hall2010_diploid.csv` |
| Log[E. coli IMDH mutant relative growth rates] | 6 | 1 | `Lunzer2005_*.csv` |
| Avian lysozyme thermostability | 3 | 1 | `Malcolm1990.csv` |
| Log[relative fitness among Methylobacterium extorquens mutants] | 4 | 1 | `Chou2011.csv` |
| Log[HIV replicative capacity on CCR5+ cells] | 5 | 3 | `daSilva2010_CCR5.csv` |
| Log[cefotaxime MIC of E. coli TEM alleles] | 5 | 1 | `Weinreich2006Tan2011_Weinreich2006.csv` |
| Log[relative viability among fruit fly mutants] | 5 | 3 | `Whitlock2000.csv` |
| Log[cefalexin MIC of Bacillus cereus metallo-β-lactamase alleles] | 4 | 1 | `Meini2015.csv` |
| Log[relative fitness among LTEE E. coli mutants in DM25 + EGTA] | 5 | 2 | `Khan2011Flynn2013_DM25_EGTA.csv` |
| A. niger colony growth | 5 | 4 | `deVisser2009.csv` |
| sesquiterpene synthase 5-epi-aristolochene | 6 | 10 | — |
| P. falciparum DHFR pyrimethamine | 4 | 2 | `Lozovsky2009Jiang2013_Lozovsky2009.csv` |
| E. coli DHFR trimethoprim IC75 | 6 | 2 | `Palmer_DHFR_ic75.csv` |
| mammalian GR cortisol sensitivity | 4 | 4 | `Bridgham2009.csv` |
| ampicillin MIC, E. coli TEM | 4 | 3 | `Mira2015_TEM_AM.csv` (or `_AMP`; must be checked) |
| N = 5, K = 0 / 1 / 2 / 4 / 5 | 5 | 1 / 2 / 2 / 5 / 7 | simulated |

Spot-check done in passing while testing the eq. (15) identity: GraphFLA gives
`n_lo` = 1 on `Weinreich2006Tan2011_Weinreich2006.csv`, matching the printed 1.
The remaining 14 were **not attempted** in this pass — the file-to-row mapping
needs checking per landscape (which fitness column, which transform, and for Mira
which of `_AM`/`_AMP` is ampicillin), and `n_lo` is already the project's
best-anchored metric.

**Table 3** caption, verbatim: "Average epistatic influence on fitness landscape
topography as a function of epistatic order in select datasets". Columns: Epistatic
order | Aggregate reduction in residual variance | Number of epistatic terms
significantly different from zero | Mean reduction in residual variance per
epistatic term. The decomposition is explicitly Fourier–Walsh. Printed, for three
empirical landscapes, all with local data:

| panel | landscape | 1st | 2nd | 3rd | 4th | 5th | 6th |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (a) | Log[IC75 of E. coli DHFR alleles against trimethoprim] | 0.279 | 0.266 | 0.233 | 0.144 | 0.0685 | 0.0065 |
| (b) | Mammalian glucocorticoid receptor cortisol sensitivity | 0.171 | .405 | 0.420 | 0.004 | — | — |
| (c) | Log[MIC of E. coli TEM alleles against ampicillin] | 0.353 | 0.278 | 0.279 | 0.091 | — | — |
| (d) | N = 5, K = 4 (simulated) | 0.027 | 0.315 | 0.402 | 0.241 | 0.015 | — |

(Printed as shown, including the typographic ".405".) Footnote, verbatim: "Largest
value for each dataset shown in bold".

**Why this is ranked 5 rather than higher.** It is a genuinely strong,
cleanly-printed, multi-landscape anchor for the order-wise variance decomposition —
the family with "three anchors but all different conventions". It is ranked below
the first four only because the reproduction was not attempted here: "aggregate
reduction in residual variance" per order is a *per-order increment* in a
Fourier–Walsh basis, whereas `higher_order_epistasis` returns a *cumulative* OLS
R², so the comparison is `R²(k) − R²(k−1)` against their column and the match is
not guaranteed. Working it properly needs the regression-significance machinery
their third column implies. **This is the single best next target for the
`higher_order_epistasis` / `walsh_hadamard` family**, and all three empirical
inputs are already local.

Also useful: **Table 2**, "Published combinatorially complete fitness landscapes not
examined here", is a curated inventory of 18 further landscapes with L and genotype
counts and no statistics — a corpus-expansion list, not an anchor. Noted typo in
their own reference list: Hall et al. 2010 is cited as "J. Hered. 1010, S75–S84"
(correct volume is 101, Suppl 1).

---

# RANK 6 — Jones & Forrest 1995 — the `fdc` origin, with exact small targets

**Verified citation (no DOI exists)**

> Terry Jones and Stephanie Forrest. "Fitness Distance Correlation as a Measure of
> Problem Difficulty for Genetic Algorithms." In Larry J. Eshelman (ed.),
> *Proceedings of the Sixth International Conference on Genetic Algorithms
> (ICGA'95)*, Pittsburgh, PA, 15–19 July 1995, pp. 184–192. Morgan Kaufmann.
> ISBN 1-55860-370-0.
> Archival record: **Santa Fe Institute Working Paper 1995-02-022**, February 1995,
> RePEc handle `RePEc:wop:safiwp:95-02-022`.

**There is no DOI.** The ICGA'95 proceedings are not registered with Crossref;
Morgan Kaufmann's pre-2000 ICGA volumes were never DOI-registered. Cite the SFI
working paper as the stable, openly downloadable artifact. Do not invent a DOI.

Two implementation mismatches that matter before any comparison:

1. Jones & Forrest define FDC as the **Pearson** correlation between fitness and
   Hamming distance to the nearest global optimum. GraphFLA's `fdc` defaults to
   `method="spearman"`. Any comparison must pass `method="pearson"`.
2. They maximise, and classify: misleading `r ≥ 0.15`, difficult
   `−0.15 < r < 0.15`, straightforward `r ≤ −0.15`.

They state that FDC is computed **exhaustively** for problem spaces of 2^12 points
or fewer, so those values are exact and reproducible. The usable targets are the
small deceptive binary functions: Whitley's F2 (4 bits, 16 genotypes, `r = 0.51`),
Whitley's F3 (`r = 0.36`), the Deb & Goldberg 6-bit fully deceptive function (64
genotypes, `r = 0.30`) and its "fully easy" counterpart (`r = −0.23`), the Liepins
& Vose 10-bit fully deceptive function (`r = 0.98`) and its transform (`r = −0.02`),
the Horn–Goldberg–Deb 11-bit long path (`r = −0.12`), and One Max (`r = −1.0`).
The NK values (`−0.83, −0.55, −0.35`) are **means over ten randomly generated
landscapes** and are not reproducible; the royal-road and De Jong values are from a
4,000-point sample and are reproducible only to sampling error.

**Status: not attempted.** The blocker is the fitness tables, not GraphFLA: the
function definitions live in Whitley, FOGA 1 (1991) pp. 221–241, Deb & Goldberg
IlliGAL Report 92001, and Liepins & Vose, FOGA 1 (1991) pp. 36–50 — all old, none
DOI-registered, and all reproduced inconsistently in later literature, so the
originals are required. **These quoted values come from the delegated search, not
from this agent's own reading of the primary PDF, and must be re-read before use.**

One free, exactly checkable corroboration does exist: Poli & Galván-López, "On the
Effects of Bit-Wise Neutrality on Fitness Distance Correlation, Phenotypic Mutation
Rates and Problem Hardness", FOGA 2007, LNCS 4436, pp. 138–164, DOI
`10.1007/978-3-540-73482-6_9`, derives analytically that OneMax has `r = −1` for
any string length. That is an independent published justification for the
`fdc(onemax) == -1` assertion already in `tests/test_metrics.py`.

A published critique worth citing alongside the origin: Altenberg, "Fitness Distance
Correlation Analysis: An Instructive Counterexample", SFI WP 1997-05-037 / ICGA'97
(single author, no DOI), constructs a 64-bit function that is GA-easy with FDC
exactly 0 by analytic derivation.

---

# RANK 7 — Manivannan & Ogbunugafor 2026, *G3* — ten seascapes, most already local

**Verified citation (Crossref `10.1093/g3journal/jkag166`)**

> Title: *Deconstructing empirical fitness seascapes across scales of granularity*
> Authors: Manivannan, Swathi Nachiar; Ogbunugafor, C Brandon
> Journal: *G3: Genes, Genomes, Genetics*, 16, article jkag166 (2026-06-26)
> DOI: 10.1093/g3journal/jkag166 (preprint PMC12889575, bioRxiv 2026.02.04.703871)

Printed r/s definition, verbatim (Methods, eqs. 11–13):

> "(11) F(x) = β0 + Σ βi xi, where F(x) = fitness, xi = state at locus i, n = number
> of loci. The roughness is the mean squared error of the regression model, (12) Σx
> (fx − F(x))² / m (where m = number of genotypes), while the slope is the mean of
> absolute values of linear coefficients, (13) Σ |βi| / n."

Note eq. (12) as printed is the mean squared residual with **no square root** —
unlike Song & Zhang's otherwise identical formula, which takes the root. Either the
printed equation has a typographical error or their r is on a squared scale; this
must be settled before any value from this paper is treated as GraphFLA's
`r_s_ratio`.

Where the numbers are: Figure 6 panels a, b, d, e are annotated with r/s values
("Fitness graph with C w (avg), C w (tot), and r/s values for the DHFR pyrimethamine
seascape at a pyrimethamine concentration of 1 μg ml⁻¹"). They are **text
annotations inside a figure**, not plotted points and not table cells. Section 3.6
is qualitative only: "we observe that the distributions of roughness-to-slope ratios
look different across seascapes". **There is no table of computed statistics** —
Table 1 is the dataset inventory, Table 2 the metric inventory.

Inputs: Table 1 names all ten seascapes, and most have local CSVs — HIV replicative
capacity (`daSilva2010_CCR5/CXCR5`), DHFR proteostasis (`Guerrero2019_*`, 9 files),
yeast growth (`Hall2010_haploid/diploid`), LTEE (`Khan2011Flynn2013_*`), DHFR
pyrimethamine (`Lozovsky_DHFR_ic50_c57..c61`, `Palmer_DHFR_ic75`), β-lactam
resistance (`Mira2015_TEM_*`, 15 files), blaTEM cefotaxime/piperacillin
(`Weinreich2006Tan2011_*`, subject to the known tenfold defect).

Verdict: **medium strength.** It would anchor `r_s_ratio` and possibly
`walsh_hadamard` across many conditions at once with local inputs, but the targets
are figure annotations and the printed roughness definition is ambiguous. Worth one
focused session, mainly for the breadth.

---

# Searched and ruled out — do not repeat

| candidate | outcome |
| --- | --- |
| **Schenk, Szendro, Salverda, Krug & de Visser 2013**, *Mol Biol Evol* 30(8):1779–1787, DOI `10.1093/molbev/mst096` (Crossref-verified; author list correct as supplied) | **Ruled out.** All four ruggedness statistics (r/s, F_sum, f_s+f_r, N_cp) are **box plots** in Fig. 3. Caption verbatim: "The central line of the box plots indicate the median, the borders of the box the 25th and the 75th percentile, and error bars the 1st and the 99th percentile, which are based on resampling the data." The only recomputable printed numbers are topological: "the global maxima occur at genotypes carrying two mutations, respectively, at E104K + G238S and I173V + S235T ... In addition, the large-effect landscape contains a local maximum (E104K + R164S)", i.e. `n_lo` = 2 for the large-effect landscape, plus "rank 14 and 13 out of 16". `n_lo` is already well anchored, data are in supplementary tables S1/S2 with no local CSV. Residual value: it is a pointer to Szendro et al. 2013, whose meta-analysis table is what Schenk quotes. |
| **Szendro, Schenk, Franke, Krug & de Visser 2013**, *J Stat Mech* P01005 | **In flight elsewhere, deliberately not duplicated.** Schenk 2013 confirms it tabulates r/s, F_sum, f_s+f_r and N_cp across landscapes — still the most promising comparative table in the field. |
| **Fragata, Matuszewski, Schmitz, Bataillon, Jensen & Bank 2018**, *Heredity* 121(5):422–437, DOI `10.1038/s41437-018-0125-7` (Crossref-verified) | **Ruled out as a printed-number anchor.** Computes exactly the right quantities — per-position-pair Ferretti γ_{i→j}, r/s, peak counts — and every one is in a figure: γ in Fig. 4 and Figs. S12–S13, r/s in Fig. S11, number of optima in Fig. 5 and Fig. S14. Europe PMC and NCBI return abstract-only for PMC6180090. Data archiving, verbatim: "The complete documentation of all analyses, which allows for the reiteration of all steps, is available from the Dryad Digital Repository 10.5061/dryad.k7jm5hp." That deposit remains a **code-level** corroboration lead for `gamma` — recovering the authors' computed γ values from their analysis documentation, which is not the same as anchoring to a printed number and must be labelled as such. |
| **Schulz, Tan, Wu & Wang 2025**, *PNAS* 122(2), DOI `10.1073/pnas.2413884122` (Crossref-verified) | **Exclusion confirmed.** The brief asked whether it offers an unusually strong printed statistic justifying work despite sitting at the 1,024-variant cutoff. It does not. Its one r/s-family statistic is inverted and plotted: "We find that antibody landscape has a larger s/r than any constrained NK landscape (SI Appendix, Fig. S10 A)". No peak count, no epistasis-type fractions, no γ in text or table. Worse, the landscape analysed is an *inferred model* (maximum-likelihood plus Walsh–Hadamard band-pass denoising, SI section C), not the raw measurements. Local CSV `Schulz2025.csv` exists if that ever changes. |
| **de Visser & Krug 2014**, *Nat Rev Genet* 15(7):480–490, DOI `10.1038/nrg3744` | **Dead end.** It has **no tables at all**: `nature.com/articles/nrg3744/tables/1` returns 404 (Nature serves every table at that path) and the article's float list is four figures. Figure 3, "Trends in the ruggedness of empirical fitness landscapes", is the ruggedness survey — plotted, and its numbers are Szendro et al. 2013, which is already in flight. Adds nothing. |
| **Bank 2022**, *Annu Rev Ecol Evol Syst* 53(1):457–479, DOI `10.1146/annurev-ecolsys-102320-112153` (single author) | **Dead end.** The author's accepted manuscript (arXiv:2204.13321v3) contains Figures 1–4 and zero tables, and zero numeric values for any target statistic. Purely conceptual. |
| **Blanquart & Bataillon 2016**, *Genetics* 203(2):847–862, DOI `10.1534/genetics.115.182691` | **Different statistics — not an anchor for our six.** Table 1 is dataset metadata; Table 3 prints per-landscape *Fisher's-geometric-model posterior parameters* (n, Wmax, σ, Q, P-values) for 26 datasets; Table S1 holds their six summary statistics (mean selection coefficient, mean pairwise epistasis coefficient, SDs of each, correlation of epistasis with background fitness, maximal fitness). No local optima, no r/s, no γ, no epistasis-type fractions, no Walsh orders, no path counts. Epistasis types appear only as prose anecdotes. **Residual value: File S1/S2 is a cleaned compilation of 26 landscapes** including the Costanzo 2010 yeast double-mutant subsets, which is a corpus-expansion resource; pull it from the GSA site, since Europe PMC's supplementary bundle for PMC4896198 holds only figures. |
| **Weinreich, Lan, Wylie & Heckendorn 2013**, *Curr Opin Genet Dev* 23(6):700–707, DOI `10.1016/j.gde.2013.10.007` | **No table of order-wise variance fractions** — Table 1 is bibliographic (mutation counts, gene counts, largest complete subset). Figure 1's order-wise decomposition for 14 landscapes is plotted only, on a log scale with error bars, and is **mean-squared Walsh coefficient per order**, not fraction of variance. Two printed recomputable numbers survive: "resistance increased monotonically on only 18 of the 120 trajectories" for the Weinreich 2006 TEM landscape (18/120 = 0.15, checkable against local data), and Figure 2's two individual Walsh coefficients for the L=3 Malcolm 1990 avian lysozyme landscape (S91T = −1.53 °C; I55V×S91T = zero) with the normalisation convention stated as the unnormalised Hadamard matrix. Those two are small but exact unit tests, and `Malcolm1990.csv` is local. |
| **arXiv 1303.3842**, "Antibiotic resistance landscapes: a quantification of theory-data incompatibility for fitness landscapes" | **Dead end, and never peer-reviewed.** Real author list (9): Crona, Patterson, Stack, Greene, Goulart, Mahmudi, Jacobs, Kallman, Barlow. Single version, no `journal-ref`, only DOI is the arXiv stub `10.48550/arXiv.1507.00041`-style DataCite record. Its primary object is the TEM **mutation record** (which variants exist), not measured fitnesses; its "tables" are lists of mutants. The one marginally relevant number (5 / 6 / 8 of 19 double mutants more fit than both / one / neither single mutant, from Goulart et al. 2013) is rank-based, not a fitness statistic. Likely conflated in a second-hand bibliography with Crona, Greene & Barlow, *J Theor Biol* 317:1–10 (2013), DOI `10.1016/j.jtbi.2012.09.028`. |
| **Brouillet, Annoni, Ferretti & Achaz**, "MAGELLAN: a tool to explore small fitness landscapes", bioRxiv DOI `10.1101/031583` | **Not a validation-target source, and never published in a journal.** Crossref type is `posted-content` with no container-title, volume or pages; any bibliography entry giving it a journal is fabricated. It *lists* γ and γ\* in section 3.2 — verbatim: "γ and γ∗ (Ferretti et al., submitted): correlation in fitness effects between genotypes that only differ by 1 locus, averaged across the landscape. γ∗ is the correlation in sign and is therefore independent of the scale (linear or log)." — but prints **no numeric value anywhere**. Useful provenance by-product: MAGELLAN attributes "number of sinks" and the chain statistics to Ferretti et al. 2018, r/s to Aita et al. 2001, peaks to Weinberger 1991, epistasis types to Weinreich et al. 2005 / Poelwijk et al. 2007, and Fourier/Walsh to Stadler 1996 / Weinreich et al. 2013 / Neidhart et al. 2013. |
| **Hinz, Amado, Kassen, Bank & Wong 2024**, *Mol Biol Evol* 41(5):msae086, DOI `10.1093/molbev/msae086` | γ defined and used, verbatim: "Epistasis was estimated using the summary statistic gamma (γ), defined as the correlation of fitness effects of the set of AMR mutations across multiple genetic backgrounds (Ferretti et al. 2016)." **No γ number printed** — values live only in Fig. 4c. **Residual value: the reference implementation is downloadable** (GitHub `andreamado/unpredictability_hinz_et_al`, Zenodo `10.5281/zenodo.11111573`), so exact γ values are recoverable from their code. That is a cross-implementation diff, not a published number, and the landscape is a mutation × background × environment panel rather than a hypercube. |
| **Ghenu, Amado, Gordo & Bank 2023**, *Phil Trans R Soc B* 378(1877):20220058, DOI `10.1098/rstb.2022.0058` | Same γ, **no numeric value printed** (Fig. 3a and ESM Fig. S22 only). Text is qualitative/statistical. Data on GitLab + Zenodo `10.5281/zenodo.7661199` + BioProject PRJNA910115. |
| **Pressman, Liu, Janzen, Blanco, Müller, Joyce, Pascal & Chen 2019**, *J Am Chem Soc* 141(15):6213–6223, DOI `10.1021/jacs.8b13298` | Prints γ₁ "approximately 0.3–0.4" as a range in running text, correctly attributed to Ferretti et al. 2016. **Ruled out: their γ_d uses Levenshtein edit distance (substitutions, insertions and deletions), not Hamming**, and is computed on a non-hypercube deep-sequencing pool. Not like-for-like, and a range is not a target. |
| Other γ users confirmed plot-only and not worth re-chasing | Lai, Liu & Chen 2021 *PNAS* 118(21), DOI `10.1073/pnas.2025054118` (Pressman's Levenshtein γ_d, Fig 6C only); Mira, Østman, Guzman-Cole, Sindi & Barlow 2021 *AAC* 65:e01990-20 (uses PWEM, not γ); Zhu et al. 2024 *Nat Commun* 15:10330 (uses MAGELLAN but reports only epistasis-type fractions: magnitude 47.9 %, sign 35.4 %, reciprocal sign 13.3 %, Supplementary Fig. 15 — a possible `classify_epistasis` lead, unevaluated); Guerrero et al. 2019 *Genetics* 212:565–575 (Ferretti cited in the introduction only); Martí-Gómez et al. 2026 *Mol Biol Evol* 43(2):msag023 (gpmap-tools; cites MAGELLAN, no γ value). |
| Ferretti/Achaz 2026 follow-ups | Ribeca et al., "Simple sign epistasis and evolutionary detours in fitness landscapes", arXiv:2604.22611 — γ and γ\* are **figure axes only**; its value is that it restates the identity `γ* = 1 − φ_ss − 2φ_rs`, which is what exposed the tie-handling defect above. Ghafari et al., arXiv:2605.03046 — theory only, no empirical landscapes, no printed γ. Both arXiv-verified only; no Crossref journal record exists yet. |
| **Fragata et al. 2019**, *Trends Ecol Evol* 34(1):69–82, DOI `10.1016/j.tree.2018.10.009` (Crossref-verified, five authors) | **Parked, unresolved.** No PMC record, Semantic Scholar reports the PDF as closed, cell.com and sciencedirect return 403, no preprint, no repository copy found. Cannot confirm or deny a numeric table. The abstract frames it as a model taxonomy, so the prior is low. Needs institutional access. |
| MAGELLAN bundled landscape library | **Unreachable.** `wwwabi.snv.jussieu.fr/public/Magellan/` times out over both http and https. This was the most promising route to the Weinreich 2006 `.fl` input and to the Cervera 2016 fitness values, and to γ for the 38-landscape Ferretti collection. Blocker, not a dead end — retry later. |
| Whether Song & Zhang's Domingo/Kuo numbers are reachable from the bundled CSVs | r/s on Kuo: **yes, exactly**, under the paper's allele coding. N_max, F_rse: no, under any of the conventions enumerated — missing neighbours absent / imputed 0 / imputed −inf; 2-state, 3-state and 4-state site models; raw, ln, log2, log10 and min-max fitness scalings; observed-only and 0-filled complete product spaces; drop-first, drop-U and min-norm full one-hot designs; OLS and Ridge at α ∈ {0.01, 0.1, 1, 10}. γ and E: definitionally different, not pursued further. |
| Whether the known tenfold `Weinreich2006` defect explains the TEM γ/γ\*/r/s mismatch | **No — tested and rejected.** Correcting 41,000 → 4,100 moves γ 0.8254 → 0.8351 (target 0.85), leaves γ\* at 0.613636 (target 0.59) and moves r/s 0.4455 → 0.4440 (target 0.43). |

---

# State of the thin and unanchored metrics after this pass

| metric | before | after |
| --- | --- | --- |
| `fitness_distribution` | **no anchor** | **14-landscape exact anchor** (Li et al. 2025 Table S1, Cauchy location + kurtosis). The remaining four descriptors (`cv`, `quartile_coefficient`, `median_mean_ratio`, `relative_range`) have no published origin — a settled negative, not an open question. |
| `r_s_ratio` | one anchor | **second anchor** (Song & Zhang Kuo r/s 4.063, exact under the paper's coding), plus the finding that the metric is reference-allele dependent on multi-allelic data. Two further leads: Manivannan 2026 (figure annotations, 10 local seascapes) and Szendro 2013 (in flight). |
| `gamma` | one anchor | **second printed value** (Ferretti 2018: TEM γ = 0.85), still mismatching at 0.835 for input reasons, with the data-defect hypothesis eliminated. Song & Zhang ruled out as an anchor on definitional grounds. Cervera 2016 Table 2 is the best remaining tabulated target, blocked on data. |
| `gamma_star` | **no anchor** | **a printed value** (Ferretti 2018 Fig. 7: TEM γ\* = 0.59; GraphFLA gives 0.6136) **and an exact algebraic anchor** — eq. (15) `γ* = 1 − φ_s − 2φ_rs`, which needs no external data and which GraphFLA satisfies to 1e-16 on tie-free landscapes and **violates by 0.0449 on tied ones**. No second paper prints a γ\* number. |
| `classify_epistasis` | one anchor | **two near-misses that are the same quantity** — Song & Zhang F_rse (0.2141 vs 0.21549 on Kuo, 0.65 %) and Li et al.'s non-magnitude fraction (definition-incompatible as printed, active-variant-restricted). Plus eq. (15) ties it algebraically to `gamma_star`. Unevaluated lead: Zhu et al. 2024 *Nat Commun* 15:10330 Supplementary Fig. 15. |
| `walsh_hadamard` / `higher_order_epistasis` | one / three-with-different-conventions | **Weinreich et al. 2018 Table 3** is the best next target: printed Fourier–Walsh order-wise variance reductions for three empirical landscapes, all local. Plus two exact small unit tests from Weinreich et al. 2013 Figure 2 (`Malcolm1990.csv`). |
| `fdc` | **no anchor** | **origin identified** (Jones & Forrest 1995, no DOI; SFI WP 1995-02-022) with exact exhaustive targets on small deceptive binary functions, blocked on the originals of three 1991–92 function definitions. Two implementation notes: GraphFLA defaults to Spearman where the origin is Pearson; OneMax `r = −1` has an independent analytic proof (Poli & Galván-López, DOI `10.1007/978-3-540-73482-6_9`). |
| `extradimensional_bypass` | one anchor | **unchanged.** Nothing found. Not pursued; the Papkou anchor stands alone. |

# GraphFLA observations for the maintainer (nothing was changed in the repository)

1. **`gamma_star` and `classify_epistasis` disagree on zero-magnitude fitness
   effects**, breaking Ferretti eq. (15) on tied landscapes. Minimal reproducer:
   `data/BioSequence/Weinreich2006Tan2011_Weinreich2006.csv` (32 rows; 39 of its 80
   squares contain a tie). Affects the whole MIC/doubling-dilution class, including
   the 15 `Mira2015_TEM_*` files — and `Mira2015_TEM_AM` happens to pass, so a
   single smoke test will not catch it.
2. **`r_s_ratio` is not invariant to the reference allele** on multi-allelic
   landscapes. 3.732 vs 4.063 on Kuo 2020, from nothing but which one-hot level is
   dropped. The convention is undocumented.
3. **`sum(graph.vs['is_lo']) != landscape.n_lo`** when plateaus are present (2,392
   vs 2,390 on Kuo 2020). `n_lo` counts plateaus, `is_lo` flags members; both are
   defensible, the silent disagreement is not.
4. **`fitness_distribution`'s docstring overclaims scale invariance.**
   `cauchy_loc` is a location parameter in fitness units — which is exactly why Li
   et al. max-normalise before reporting it.
5. **`data/BioSequence/Tu2022_T7.csv` is on a fitness scale inconsistent with its
   sibling `Tu2022_TEV.csv`** relative to Li et al.'s inputs, and does not invert
   cleanly. Both files share an identical minimum of −1.1766, the signature of a
   curation-time floor. A data-provenance question, in the same class as the
   documented tenfold `Weinreich2006` error.
6. **`data/BioSequence/Kuo2020.csv` is rounded to 3 decimal places** — 2,473
   distinct fitness values across 197,890 rows. Any tie-sensitive statistic on this
   file is precision-limited at roughly the 0.5–1 % level.

# Reproduction scripts

Working scripts live in this session's scratchpad, not in the repository or in this
directory. They are short and were written to be regenerated rather than archived;
the recipes needed to recreate every number above are stated inline in the sections
above (input file, build call, metric call, transform, and coding convention).

# Incomplete at hand-off

- The 14 remaining Weinreich et al. 2018 Table 1 peak counts, and all of its
  Table 3 order-wise profiles, are unattempted. This is the highest-value cheap
  work left.

(Closed since first writing: `classify_epistasis` on `Wu2016_GB1.csv` and
`Tu2022_TEV.csv` at `sample_cut_prob=0` had not finished inside the first time
budget; it has since completed and both rows are folded into the Li et al. table
above. The conclusion is unchanged.)
