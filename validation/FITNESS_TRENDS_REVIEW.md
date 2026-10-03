# Diminishing returns and increasing costs audit

## Decision and implemented definition

The two public scalars now implement the **pooled transition definition** in
Huang, Zhou and Li (2025), [NeurIPS, Appendix C.3.2](https://doi.org/10.52202/085713-1180).
The old implementation correlated one mean effect per vertex with its fitness.
That was a different estimand, not an optimization of edge pooling. This change
was explained during the session; public names, signatures and default Pearson
method remain unchanged. Local API proposals are not implemented.

For every retained improving edge `u -> v`, let `q=f` for maximization and
`q=-f` for minimization. DRI pairs `q(u)` with `q(v)-q(u)`; ICI pairs `q(v)` with
the same positive gap, representing the reverse deleterious move. Each edge
has unit weight. Pearson, average-tie-rank Spearman and intercept-inclusive OLS
use the same observations. Zero-effect edges are excluded. The graph's
neighborhood, epsilon, tau, incomplete sampling and component selection remain
part of the input. No edges or observations are reconstructed by these metrics.
A biological one-mutation interpretation requires a reversible one-step graph.

This is a descriptive summary. A two-locus additive landscape with fixed effects
1 and 2 already gives DRI `-1/sqrt(11)` and ICI `+1/sqrt(11)`. The available
mutation mixture changes across backgrounds despite zero interaction. Shared
background measurement error and effect-sign selection can also create trends.
Consequently a nonzero value, its sign, or an ordinary edge-wise p-value would
not establish mutation-specific epistasis. No inferential p-value is exposed.
Nonlinear transformations of fitness can change even the sign; no automatic
ratio/log conversion or ceiling normalization is applied.

## Reproduction and direct validation

| Evidence | Population and result | Boundary |
| --- | --- | --- |
| Johnson et al. 2019, [Science 366:490](https://doi.org/10.1126/science.aay4199), Results and Fig. 4 | 80 mutations with >=50 paired backgrounds; 64 significant OLS fits, 58 negative, 6 positive, 48 negative and deleterious on average | `paper_result` for the production centered-moment kernel on per-mutation data; not the public pooled index |
| Same Johnson input | All 80 slopes and Pearson coefficients agree with independent SciPy fits, including split-block accumulation | `independent_check`; significance uses the stated n-2 t test, not labels copied from the supplement |
| Papkou et al. 2023, [Science 382:eadh3860](https://doi.org/10.1126/science.adh3860), Fig. S22 B-C | Full archived graph: 135,178 vertices, 324,044 edges; all six public statistics independently checked | `independent_check` for coefficients and `input_check` for graph population; no numerical coefficient printed in the figure caption |

Johnson input was downloaded anew from [eLife 76491 Supplementary file 1 v2](https://cdn.elifesciences.org/articles/76491/elife-76491-supp1-v2.xlsx).
Its publisher metadata identifies the supplement as underlying data including
Johnson 2019 and licenses the article CC BY 4.0. A deterministic extraction
retains 10,645 paired measurements for 91 experimental insertions; the >=50
filter is applied in tests, not used to trim the source fixture. The 162-row
background table is the available population, not a claim that every one of
the paper's 163 recruited segregants has paired data. Original/derived SHA-256,
conversion and attribution are in `tests/fixtures/fitness_trends/manifest.json`.

Papkou reuses the existing unrounded fitness and original author edge list,
pinned to [Zenodo 8228920](https://doi.org/10.5281/zenodo.8228920). The adapter
aligns every vertex by exact sequence and gives the metric a read-only graph
view; it does not rebuild or subsample the genotype graph. In the clipped case,
fitness below -0.507774 is floored, matching the author graph-weight convention.
Both cases have identical topology:

| Supplied fitness | DRI Pearson | DRI Spearman | DRI slope | ICI Pearson | ICI Spearman | ICI slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw measured m | -0.4764872795 | -0.5586519704 | -0.4329207419 | 0.4410695088 | 0.3366470695 | 0.4090788350 |
| Floor-clipped m | -0.3290289901 | -0.2808729595 | -0.3140339878 | 0.5521615778 | 0.5438780331 | 0.4652846307 |

These targets were frozen from independent vector fits before calling the new
public metrics on these inputs. Cases use a 2e-12 absolute roundoff allowance.
The exact archived Fig. S22 plotting input `distDF_fitness.rds` remains absent;
neither row is claimed as a replay of that plotted coefficient. The raw row
also agrees with the earlier independently enumerated S22 reconstruction.

Run only the six new tests, offline:

```sh
python -m validation.tests --literature-case johnson2019.fitness_trends.counts.v1 --literature-case papkou.fitness_trends.raw.v1 --literature-case papkou.fitness_trends.clipped.v1
```

## Review of supplied literature leads

The supplied Claude note is a search lead, not a source or validation result.
Primary material was re-read where available, and earlier pinned local dossiers
were reused. The following distinctions determine which claims are promoted;
this table does not assert a fresh complete replication of every study.

| Study / DOI | Relevant method | Audit disposition |
| --- | --- | --- |
| Chou 2011, `10.1126/science.1203799` | Relative benefit `Wmut/Wbg - 1` vs background W; multiplicative null | A different response scale. The four rounded slopes in the note are its own calculations, not printed coefficients. Conceptual source, not a pooled-index target. |
| Khan 2011, `10.1126/science.1203801` | Fixed-mutation effects and per-mutation correlations, Fig. 4 | Existing primary-source survey records r/P per locus. Dryad landing page confirms original replicates and Fig. 4 R script; current file routes returned 403/401. Do not freeze the lead script's rounded means as raw measurements or use its mislabeled secondary table. |
| Johnson 2019, `10.1126/science.aay4199` | Signed effect on background, one OLS per mutation; mean effect classifies IC | Promoted as above. Keep positive, zero and negative effects within a mutation. |
| Johnson & Desai 2022, `10.7554/eLife.76491` | Centered mutation-specific models with additional slope threshold | Source of the 2019 data. The lead script merely counts already computed 2022 labels; not promoted as a refitted 2022 result. |
| Ardell 2024, `10.1126/science.adn0753` | Mutation-by-environment slopes and FDR | Lead gives 546 fits against 545 in the paper, despite matching 250/245 counts. Incomplete reconciliation, no passing case. |
| Lyons 2020, `10.1038/s41559-020-01286-y` | Per-mutation correlations; independent replicate groups for background and effect | Re-read the primary Methods. The 87.8% figure concerns mutation-specific signs; neither our two public indices nor the existing idiosyncrasy reproduction establishes it. No duplicate I_id study promoted here. |
| Diaz-Colunga et al. review 2023, `10.1098/rstb.2022.0053` | Global trends, random-landscape slope -1 | Primary text/Box 1 distinguish mutation effects, background fitness and regression to the mean. Supports interpretation limits; no empirical pooled-scalar target. The lead's phrase “a negative slope says nothing” is too absolute; inference depends on design and null. |
| Reddy & Desai 2021, `10.7554/eLife.64740` | Locus variance / signed-effect slopes | Primary equations use a particular orientation and variance normalization, not a pooled positive-effect index. No default conversion to `-slope/2`. |
| Bakerlee 2022, `10.1126/science.abm4774` | Mutant vs background fitness, total least squares | Pinned prior primary audit: subtracting one from its fitted slope gives an effect slope only for that fitted model. TLS is not a generic cure for correlated measurement noise. |
| Johnson, Reddy & Desai 2023, `10.1186/s12915-023-01585-3` | Mechanistic review of global epistasis | Primary text supports sign changes across backgrounds; conceptual comparator. |
| Diaz-Colunga, Sanchez & Ogbunugafor 2023, `10.1038/s41467-023-43806-x` | Environmental changes in effect-variance strength and R² globality | Primary definitions differ from DRI/ICI; no numeric scalar promoted. |
| Rokyta 2011, `10.1371/journal.pgen.1002075` | Pairwise additive epistasis on doublings/h | Primary paper studies pairwise antagonism; mean epistasis is not a fitness-effect/background correlation. The lead's -4.52 is not a DRI target. |
| Schoustra 2016, `10.1098/rspb.2016.1376` | Pairwise epistasis vs mutation effect, error correction | Primary text reports -14.94 and -19.25 for mean pairwise epistasis; these are not DRI/ICI values. |
| Berger & Postma 2014, `10.1534/genetics.114.169870` | Measurement-error coupling in epistasis regressions | Relevant caution; no correction silently added without replicate/error inputs. |
| Kryazhimskiy 2014, `10.1126/science.1250939` | Selected knockout effects vs background | Prior pinned primary survey: graphical trends, no common pooled scalar. |
| Wiser 2013, `10.1126/science.1243357` | Adaptation-trajectory model parameter g | Prior primary survey: different model/level of observation; g=6 is not a DRI/ICI target. |
| MacLean 2010, `10.1534/genetics.110.123083` | Selected mutations across rpoB backgrounds | Lead only for this audit; no new data or coefficient promotion. Three backgrounds do not supply a landscape-wide pooled-index target. |
| Wei & Zhang 2019, `10.1093/molbev/msz035` | Allele effects in high/low fitness segregant groups | Lead only; grouping/contrast estimator is not this scalar. The broad “immune to regression to the mean” claim is not adopted here. |
| Couce & Tenaillon 2015, `10.3389/fgene.2015.00099` | Review / adaptability trends | Context, not an independent numeric DRI/ICI anchor. |
| Couce et al. 2024, `10.1126/science.add1417` | Distribution of effects across evolved backgrounds | Lead's 6.8%/3.2% are beneficial fractions, not background/effect correlation. Not promoted. |

Huang's printed tables were computed with GraphFLA and cannot serve as
independent evidence for GraphFLA itself. In particular they cannot justify
retaining a node-mean implementation against the paper's pooled definition.
The chosen public definition is one documented summary, not a field-wide
standard. A separate per-mutation API remains a local proposal.

## Compatibility and resources

Existing node-mean scalar values intentionally change; minimizing an equivalent
negative fitness encoding now preserves interpretation. Missing/stale
`delta_fit` no longer changes the calculation. Invalid methods are rejected
before degenerate-data exits; constant backgrounds return NaN for OLS instead
of an arbitrary rank-deficient coefficient. Pearson/regression stream 32,768
edges at a time; Spearman sorts the full edge population. See
`FITNESS_TRENDS_TEST_AUDIT.md` and `../benchmarks/FITNESS_TRENDS_RESULTS.md`.

Research failures remain failures: no tolerances were widened to match package
output, and existing case fingerprints were not changed. Tutorial/API proposals
remain local; actual API reference is in the two source docstrings.
