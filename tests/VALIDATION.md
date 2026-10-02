# Scientific validation

This is a testing and evidence-gathering phase. Passing tests do not certify every
analysis function. We keep three separate checks: hand-computable or synthetic
correctness, agreement with a published definition, and reproduction of a
published result. Existing saved GraphFLA outputs are regression snapshots, not
independent scientific evidence. The supplied landscape-statistics table is a
paper discovery index only.

## Function-by-function plan

Each row describes the checks needed for that family of functions. References
are candidates or definition sources unless the results section explicitly
records a successful reproduction. A prominent journal or an available dataset
does not by itself establish that a result can be reproduced.

| Functions | Synthetic / hand-computable check | Theory and empirical check |
| --- | --- | --- |
| Landscape construction, `n_lo`, `local_optima_ratio` | Exhaustive pairwise neighbors across Boolean, DNA, RNA, protein, categorical and ordinal encodings; missing genotypes, ties, plateaus, minimization, edit distance and filters; direct peak enumeration | Multiple independent landscapes: [Papkou 2023, Science](https://doi.org/10.1126/science.adh3860), [Westmann 2024, Nature Communications](https://doi.org/10.1038/s41467-024-54723-y), [Wu 2016, eLife](https://doi.org/10.7554/eLife.16965), [Johnston 2024, PNAS](https://doi.org/10.1073/pnas.2400439121), and [Bank 2016, PNAS](https://doi.org/10.1073/pnas.1612676113). Match thresholds, imputations and peak candidate sets before counting. |
| `classify_epistasis` | Enumerate four-corner squares independently; magnitude, sign and reciprocal sign; additive and neutral cases; multiallelic squares; permutation and monotone-transform invariance of sign classes | [Weinreich 2005](https://doi.org/10.1111/j.0014-3820.2005.tb01768.x), [Poelwijk 2007, Nature](https://doi.org/10.1038/nature05451), Bank, Westmann, Papkou and Johnston. Audit whether additive or neutral squares are included and whether significance thresholds change the denominator. Positive/negative epistasis also requires a matching fitness scale and reference orientation. |
| `gamma`, `gamma_star` | Direct directed-substitution sums, including sparse and multiallelic inputs; additive and interaction anchors; fitness offset/scale and allele-label invariance | [Ferretti 2016](https://doi.org/10.1016/j.jtbi.2016.01.037), Eq. 3 and rank-based extension; Bank's multiallelic supplement. Seek the two empirical Figure 4 inputs and additional source-backed datasets. Complete-square pooling on missing data must be distinguished from the paper's alternative correlation estimator. |
| `r_s_ratio` | Independent least-squares design; analytic additive-plus-interaction cubes; zero slope, constant fitness and very small fitness units | Ferretti Appendix A and [Skwara 2023, Nature Ecology & Evolution](https://doi.org/10.1038/s41559-023-02197-4). Record genotype coordinates, fitness transformation, residual normalization and any shuffle normalization. Values in 0/1 and −1/+1 coordinates differ by a factor of two. |
| `higher_order_epistasis` | Known orthogonal interaction components and cumulative explained variance; incomplete data and rank-deficient designs; first order equals the additive fit | [Phillips 2021, eLife](https://doi.org/10.7554/eLife.71393) full-data author regression outputs. [Kuo 2020, Genome Research](https://doi.org/10.1101/gr.260182.119) is a useful definition contrast: conditional-mean effects differ from OLS on incomplete landscapes. Do not compare full-data R² with held-out scores. |
| `walsh_hadamard` | Direct small transform matrix, known pure interaction orders, reconstruction, reference changes and incomplete-input behavior | [Poelwijk 2016, PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1004771), [Poelwijk 2019, Nature Communications](https://doi.org/10.1038/s41467-019-12130-8), and Phillips coefficients. Confirm coefficient normalization before using numerical targets. |
| `idiosyncratic_index`, `global_idiosyncratic_index` | Enumerate matched backgrounds; verify SD conventions and random-pair baseline; additive, missing-background and minimum-pair cases; global aggregation excludes undefined mutations | [Lyons 2020, Nature Ecology & Evolution](https://doi.org/10.1038/s41559-020-01286-y), original definition and analysis artifacts. Check whether the global summary is an extension of the original per-mutation statistic. |
| `diminishing_returns_index`, `increasing_costs_index` | Explicit per-mutation background/effect correlations, beneficial/deleterious selection and constant-effect cases | Lyons provides the biological context. The exact aggregate index and any ceiling normalization need separate provenance; do not imply that the paper defines every package summary. |
| `evolvability_enhancing_fraction` / `evolvability_effects` | Additivity, full focal-site exclusion, directions, ties, missing neighbors, zero variance, FDR, minimization, scalar/table agreement and original labels | [Wagner 2023](https://doi.org/10.1038/s41467-023-39321-8): all four published beneficial/deleterious counts replayed; 456,448 signed author decisions matched. The corrected estimator is independently checked. See [EE review](../validation/EE_MUTATIONS_REVIEW.md); the general public API deliberately uses neighborhood variation, with the RNA measurement-error procedure confined to validation. |
| `neutrality` | Count all undirected neighbor pairs satisfying an explicit threshold; ensure construction epsilon does not silently change this denominator | Match each experiment's neutrality convention. No universal empirical fraction is assumed. |
| `fitness_distribution`, `fitness_effect_distribution`, `single_mutation_effects`, `all_mutation_effects` | Direct arrays with known moments and paired B−A effects; reverse mutation, absent allele, one position, missing backgrounds and constant distributions | Elementary statistical definitions; paper-derived fitness scale and replicate treatment for empirical inputs. A prestigious reference is not required for basic arithmetic. |
| `fdc` | Hand-computed distances and Pearson/Spearman correlations; ties, multiple optima, maximization/minimization and custom distances | Jones & Forrest (1995), *Fitness Distance Correlation as a Measure of Problem Difficulty for Genetic Algorithms*. Match global-optimum and distance conventions. |
| `autocorrelation` | Exact small-graph random-walk transition calculation versus seeded sampling; stationary weighting, centering, lag and neutral edges | [Weinberger 1990](https://doi.org/10.1007/BF00202749). Sampling uncertainty needs a justified interval, not exact equality to one random run. |
| `neighbor_fitness_correlation` | Compute neighbor means independently, then the requested correlation; isolated vertices and ties | Confirm whether the intended statistic is correlation with neighbor mean or correlation across edges. No matching original numerical target is yet established. |
| `gradient_intensity`, `fitness_flattening_index` | Direct computation from the documented definition; unit scaling, sign and degenerate inputs | Provenance unconfirmed; potentially author-defined metrics. Do not force a literature origin. |
| `local_optima_accessibility`, `global_optima_accessibility` | Exhaustive reachable sets on small directed graphs; include/exclude the endpoint explicitly; disconnected components and tied peaks | Bank, Papkou and Wu. Distinguish existence of a path from a walk's probability of reaching a peak. |
| `mean_path_length_to_local_optima`, `mean_path_length_to_global_optimum` | Enumerate tiny paths or solve a small absorbing Markov chain under an explicit transition rule | Bank's absorbing-chain methods. Shortest-path averages, all-path averages and expected adaptive-walk lengths are different quantities. |
| `mean_distance_to_local_optima`, `mean_distance_to_global_optimum` | Exact Hamming/ordinal/custom distances; multiple peaks, endpoint and unreachable cases | Use paper results only when they specify the same genotype distance and averaging population. |
| `basin_fitness_correlation` | Hand-built greedy basins versus full accessible basins; tied choices and transformed basin sizes | Westmann source workbook and code. Its bounded accessible-path calculation and log-basin correlation do not equal the package's default greedy-basin statistic. |
| `extradimensional_bypass` | Direct enumeration of small substitution/detour motifs with known totals | Wu's indirect-path analysis provides context; reconcile a local motif fraction with any reported whole-path accessibility statistic before comparing. |
| `profile`, `list_metrics` | Check selection, argument propagation, output schema, errors and equality with direct function calls | Integration wrappers; inherit the scientific evidence of the metrics they call. No separate paper is required. |

## Current promoted empirical tests

`validation/tests/test_empirical.py` fixes preprocessing and input SHA-256 hashes. Expected
results come from the publication or its archived author outputs, not from
GraphFLA. The local data directory is part of the checkout; no network request
is made during tests.

- **Papkou:** 135,178 vertices, 324,044 directed edges, 514 peaks and 18,019
  functional variants with the specified component and threshold edge filter.
  Exact epistasis enumeration also matches 740,211 motifs and 85,203 reciprocal-
  sign cases from SI Figure S20. The archived author notebook gives all three
  counts (408,065 magnitude/additive, 246,943 sign, 85,203 reciprocal), which
  supply the exact expected fractions for the public classification function.
  The processed CSV uses recoded nucleotide labels; the test reverses the
  verified A→A, C→G, G→T, T→C mapping. All 135,178 decoded sequence/fitness
  pairs agree with the source-backed raw component. The paper's full-library
  count (261,382) differs by 49 from the archived non-null input (261,333);
  that broader coverage discrepancy remains unresolved.
- **Westmann:** 17,765 vertices and 2,092 peaks; 58 above and 2,034 below the
  wild-type normalized repression level. Supplementary Table S1 and publisher
  source data agree with direct independent enumeration.
- **Phillips:** four full-data regression R² values for two orders and two
  antigens. These are **author-artifact outputs**, not four independent papers
  or reproductions of the publication's cross-validation curves. Repository
  commit and source paths are in the test.
- **Bank / Reia and Campos:** six named peaks reproduced from the downstream
  archive used by [Reia and Campos 2020](https://doi.org/10.1098/rsos.192118).
  The small, attributed CC0 input is in `fixtures/literature/bank2016_reia2020`.
  The project's existing Bank2016a.csv contains final-timepoint mean read counts
  and is not used as a growth-rate input for this test.
- **Wu:** the completed 160,000-variant input gives 30 peaks, 15 above WT.
  The fraction of genotypes able to reach every one of those 15 peaks is
  92.638125%, matching the published 93% to whole-percent precision. That
  intersection is a graph-based aggregate, not the per-peak API's output.
- **Johnston:** the completed 160,000-variant input gives 520 peaks among
  9,783 active measured candidates. The author's activity mask and 871 imputed
  neighbor values are retained. The unrestricted 797-optimum count is a
  different quantity and is not frozen as a published target.

## Limits and unresolved discrepancies

- Westmann reports 83,100 genotype squares. Independent enumeration finds 88
  squares with neutral edges; GraphFLA's strict directed motif count is 83,012.
  Similar rounded fractions do not establish identical denominator semantics.
- Bank's recovered archive yields 63.8447%, 27.7178% and 8.4375% magnitude,
  sign and reciprocal-sign epistasis, whereas the paper reports about 62%, 30%
  and 8%. Independent four-corner arithmetic agrees with GraphFLA; the original
  posterior inputs and aggregation remain unresolved.
- Kuo's reported R² ranges reproduce with its published equations, while
  GraphFLA's OLS estimator is a different calculation on missing genotypes.
- Skwara's ±1 coding and shuffle-normalized ruggedness differ from GraphFLA's
  raw r/s in 0/1 coding. This is not an arithmetic defect.
- Broad coverage and saved regression snapshots do not replace original-paper
  reproduction for gamma, r/s or path statistics. EE author-procedure replay now
  reproduces the four Wagner counts, while production intentionally corrects
  its variance defect; see `validation/EE_MUTATIONS_REVIEW.md`. For idiosyncrasy,
  the Lyons tRNA global mean/SEM now have an offline empirical test and case
  `lyons.trna.iid.v1`; see `validation/IDIOSYNCRASY_REVIEW.md` for its RNG and
  population scope and the unresolved Fig. 1a source discrepancy.

## Resolved defects

The four defects found during the testing phase are fixed; the strict
expected-failure cases that recorded them are now ordinary passing tests. Each
fix is anchored by an analytic reference, not by a snapshot of the new output.

1. `r_s_ratio` lost fitness-unit invariance near a slope of zero because the
   degeneracy test used an absolute tolerance. Both roughness and slope scale
   linearly with fitness, so the test is now relative to the fitness scale.
   `test_roughness_slope_is_invariant_to_fitness_units` checks the analytic
   value 3/8 across scale factors from 1e-10 to 1e10.
2. `single_mutation_effects` and `all_mutation_effects` labelled a mutation
   A→B while reporting mean effect and the binomial test in the reverse
   direction. Effects are now signed `mutation_to` minus `mutation_from`,
   matching `fitness_effect_distribution`. The independent reference in
   `test_metrics.py` was re-derived under the corrected convention, and
   `test_mutation_effects_have_consistent_direction` ties the summary to the
   per-background distribution. This changes the sign of `mean_effect` and the
   meaning of `test_type` relative to releases before this fix.
3. `fitness_effect_distribution` raised on a single-position landscape because
   the background column set was empty. Such a landscape has one trivial
   background shared by every genotype, which is now represented explicitly.
4. Fully neutral landscapes were rejected as edgeless. A landscape whose
   neighbouring pairs are all neutral has neighbours but no directed edges;
   construction now rejects only a graph with neither directed edges nor
   neutral pairs.

A documentation contradiction in `gamma` was also corrected: the Returns prose
stated that values near 0 mean weak epistasis, contradicting both the Notes and
Ferretti et al. (2016), under which an additive landscape gives gamma = 1 and a
House-of-Cards landscape gives gamma near 0. `gamma_star`'s Returns prose was
likewise reworded and given the same reference.

## Promotion and artifact contract

The executable record format and incremental workflow are documented in
[validation/README.md](../validation/README.md). Versioned cases in that directory
are the source of expected values and input fingerprints for empirical tests.
Research history is append-only in the external artifact store.

Before any empirical test is promoted, retain the original source, URL, version,
license and checksum; record an exact paper locator and the expected value before
running GraphFLA; specify fitness scale, filtering, imputation, neighborhood,
neutrality, denominator and uncertainty. Keep a deterministic reproduction
script, environment/source identity, results and failed attempts.

Each claim records both an **evidence tier** (published number, author-artifact
output, or independent calculation) and an **outcome** (reproduced, definition
mismatch, unresolved mismatch, missing data/method, or no overlapping metric).
A successful independent calculation of a different statistic is not a passing
GraphFLA publication reproduction. Multiple conditions from one study add
coverage but do not count as independent publications.

Large downloads and exploratory scripts remain in the external validation
workspace. Researchers work against a read-only source snapshot; only reviewed,
small and deterministic checks enter this suite. Never tune an epsilon,
transform or tolerance merely to make a published number match.
