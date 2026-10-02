# Papkou 2023 — Fig. S13 raw reproduction notes

The verbatim source fragments and their page/figure/section locators are kept in the matching `quote` and `definition_quote` fields in `record.json`. This note explains the interpretation and reproduction decisions without duplicating those excerpts.

## Acquisition and bibliography

- Reused the dossier's main Science article PDF and input audit. The supplied corpus entry has an incorrect author list: it names Andrei Papkou, Aviv Regev, and Joanna Masel. Crossref DOI metadata verifies the article title and the four authors Andrei Papkou, Lucia Garcia-Pastor, José Antonio Escudero, and Andreas Wagner. The retrieved metadata is saved as `papers/Papkou2023/sources/crossref_adh3860_codex.json`; both the successful metadata request and the failed supplement request are in `papers/Papkou2023/acquisition_codex.jsonl`.
- The Science supplementary-PDF endpoint returned HTTP 403; the access-challenge HTML response is logged and saved as `papers/Papkou2023/sources/Papkou2023_science_adh3860_supplement_attempt_codex.html`. I reused the existing SI text extraction at `/Users/arwen/Documents/GitHub/GraphFLA/_verify/papers/papkou_science_SM.txt` (SHA-256 `abd424ad6fee922b61e34ccb961490e28f820cdc17a4057dc123dc783b293072`), copied in the dossier as `papers/Papkou2023/sources/Papkou2023_science_adh3860_SM_codex.txt`. No new file was written under the GraphFLA repository.
- The main paper's data-availability paragraph includes the phrase “available from Zenodo” (Data and materials availability, printed p.8). The existing `papers/Papkou2023/sources/zenodo_archive_members.json` index lists author adaptive-walk and accessibility RDS outputs. I found no separate publisher Source Data attachment for Fig. S13 and did not read values from those plots.

## Methods interpretation

- For the Fig. S13 adaptive trajectories, the Methods describe mutation-fixation steps weighted by Kimura fixation probabilities: probabilities for one-mutant neighbors are rescaled to a distribution, and the next neighbor is sampled from it. The walk ends at a fitness peak. This is neither a greedy choice nor uniform random selection among improving neighbors; a direct fitness-proportional rule is NOT STATED IN PAPER. GraphFLA's existing `HillClimb` supports best-improvement and uniform random improving-neighbor choices; neither is the paper's Kimura rule, so I did not run either as a substitute.
- Exact-tie handling: **NOT STATED IN PAPER.** The Methods do not say whether exactly equal-fitness neighbors enter the transition distribution.
- The SI defines an accessible path through strictly fitness-increasing steps and describes enumerating all shortest paths. Fig. S13D summarizes path multiplicity by peak group. GraphFLA's navigability functions return reachability fractions or shortest-path lengths; they do not count all shortest paths. I marked these path-count targets `definition_incompatible` and did not substitute path length for path count.
- The paper describes uniform fixation probability for improving mutations and greedy highest-fitness moves as separate simulation types. These are not the walk rule tied to the Fig. S13A/B adaptive-trajectory result. The separate greedy-walk count is useful for interpreting the S13D denominator, as detailed below.

## GraphFLA run and input

- Input was the trusted `data/BioSequence/Papkou2023_DHFR.csv` file, SHA-256 `fca2fb47cad417a049698f43b20f90b320a4927418d502e527ad112ea46c82da`.
- I reused the dossier's recorded build settings: `DNALandscape().build_from_data(df.sequences, df.fitness, tau=-0.507774, filter_mode="both", epsilon=0, verbose=False)`. The run returned `n_configs=135178`, `n_edges=324044`, and `n_lo=514`. The SI Methods network-construction section reports edge count “324,044” (printed p.15); the run therefore used the recorded nonfunctional-pair filter and matches the paper's directed graph size.
- As a preprocessing diagnostic, a first default `build_from_data` call without the dossier's threshold/edge filter returned 1,490,486 edges, not the published graph. I did not use that graph for any comparison. The justified run uses the dossier's already-recorded `tau` and `filter_mode` settings, not a parameter search. An initial wrapper also failed while serializing the metrics inventory's DataFrame; output formatting was fixed and the landscape was rerun successfully. That wrapper error did not produce a scientific result.

## Fig. S13D denominator discrepancy

- The same curated component has 135,178 variants and GraphFLA finds 514 strict peaks, so `n_configs - n_lo = 134,664` nonpeaks. The paper's Methods separately says there were 134,664 greedy-walk starts excluding peak variants, consistent with the strict peak count.
- Fig. S13D prints 134,662 variants excluding peaks. Subtracting this from the component size implies 516 excluded variants, two more than GraphFLA's 514 peaks. The implied 516 count is my arithmetic, not a paper-stated count; its explanation is **NOT STATED IN PAPER.** The paper gives no alternate peak definition for S13D, and its separate greedy-walk count aligns with the standard strict peak set. I therefore recorded a mismatch and did not remove two additional variants or change epsilon.

## Stochastic values and plot-only panels

- No adaptive-walk trial was run, so there is no GraphFLA seed to report; the paper's random-number seed is **NOT STATED IN PAPER.** The Methods start sample is uniform with replacement (see the quote fragments in the S13A overlap), supporting a binomial approximation. For the printed 12.5% start-allele fraction, the record gives a plug-in uncertainty calculation using the caption's 961,450 walks and the rounded reported proportion: standard error about 0.0337 percentage points, with a nominal 95% margin about 0.0661 percentage points. This is an approximate calculation, not a paper-reported interval or a reproduction. The 78% first-step result has no separate denominator or interval in its statement; the conditional uncertainty calculation is labeled as such in `record.json`.
- The Fig. S13A averaged fitness curve, S13C endpoint composition, S13D Cys27-group path-count distribution, and S13E rank distribution are marked `plotted` and were not numerically read. Their exact values are not printed, and the matching GraphFLA statistics are absent or definition-incompatible. The author archive includes walk-result files, but retrieving them would not make these quantities GraphFLA metrics.

## Result

The paper's landscape size (`n_configs`) and strict peak count (`n_lo`) reproduce exactly. The high/low shortest-path multiplicities do not match a GraphFLA metric. The S13D denominator is an unresolved two-variant mismatch. Walk-based percentages and trajectories are not claimed as exact reproductions.
