# Lyons 2020 — yeast tRNA idiosyncratic-index reproduction

## Bibliography and acquisition

I located the study by title and DOI. The Crossref record for DOI `10.1038/s41559-020-01286-y` gives the title “Idiosyncratic epistasis creates universals in mutational effects and evolutionary trajectories” and authors “Daniel M. Lyons; Zhengting Zou; Haiqing Xu; Jianzhi Zhang” (Crossref `title` and `author` fields). That agrees with the assignment’s abbreviated author order; I found no supplied-citation error. The Nature page independently lists the same article citation: [Nature article page](https://www.nature.com/articles/s41559-020-01286-y).

The target paper’s reference list cites the original tRNA landscape study. Crossref DOI `10.1126/science.aae0568` gives the title “The fitness landscape of a tRNA gene” and authors “Chuan Li; Wenfeng Qian; Calum J. Maclean; Jianzhi Zhang” (Crossref `title` and `author` fields). The cited dataset paper and DOI therefore agree with the source attribution in the Lyons paper.

The author manuscript was obtained as Europe PMC JATS full text. The authors’ repository was acquired with `git clone --depth 1 https://github.com/lyonsdm/idiosyncrasy.git`; the checked-out commit is `0a6c2ce3a277d56679dae52b222106d05af93983`. The clone is retained under `papers/Lyons2020/sources/idiosyncrasy_repo_codex/`. Its logged SHA-256 is for the tracked-file stream from `git archive --format=tar HEAD`; the raw data and notebook also have individual file hashes. Codex-suffixed copies of the data and analysis notebook, the manuscript XML, and both Crossref records are in the same dossier’s `sources/` directory. Every saved acquisition and its SHA-256 is listed in `papers/Lyons2020/acquisition_codex.jsonl`.

The direct Science publisher URL `https://www.science.org/doi/10.1126/science.aae0568` returned HTTP 403 in the web lookup. Bibliography verification succeeded via Crossref instead. No reproduction input was blocked.

## Paper overlap and quoted definition

The printed tRNA result is the target stored in `record.json`, `overlaps[0].quote`; its locator is Results, “Epistasis is highly idiosyncratic,” in the author manuscript. The same sentence includes the quoted fragment “828 single mutations” and reports the standard error. It is printed in the text, not read from a plotted point.

The existing dossier audit already transcribes the full defining Results sentence and the Methods estimation paragraph verbatim in `papers/Lyons2020/definition_audit_claude.md`, §§1b and 1d. I reuse that audit instead of repeating its long quotations. The key Methods phrase is “same number of pairs” (author manuscript, METHODS → “Estimating idiosyncrasy index”). See the existing audit for the full verbatim definition and aggregation quotes.

For `I_id`, a minimum number of backgrounds is **NOT STATED IN PAPER.** Whether the paper’s standard deviation uses a sample or population divisor is **NOT STATED IN PAPER.** The released code resolves the latter for the reproduction: the notebook calls `numpy.std(...)` without a `ddof` argument, so the executed procedure uses NumPy’s default population SD. The metric implementation remains under review, so `definition_match` is recorded as `unknown` as required by the validation instructions.

## Rebuilding the authors’ input

The downloaded author input is `papers/Lyons2020/sources/ExpData_codex.txt` (SHA-256 `1a89b314dfc6fe69c60af8c50ff7d694121b3f2207b8ae961acf5e55d0fe8ddb`). The notebook loads its fitness values after the authors’ scaling line:

```python
gt_fit_list.append(fit_df['Fit'][i] / 0.5)
```

Locator: `idiosyncrasy_repo_codex/01_trna/31_fitness/get_fitness.py`, line 27.

The tRNA notebook then applies these operations (short verbatim excerpts):

```python
if x[2] == 1.0 or x[3] == 1.0:
    continue
mut_effects.append(numpy.log(x[3] / x[2]))
```

```python
for pos in range(1, 70):
```

```python
gt_fit_pool = numpy.array([x for x in gt_fit_list if x > 1.0])
numpy.random.seed(n**2 + 3)
numpy.random.choice(gt_fit_pool, size=(n,2), replace=True)
numpy.log(x[1] / x[0])
numpy.std(mut_effects)
numpy.std(null_effects)
```

Locator for these excerpts: `idiosyncrasy_repo_codex/01_trna/32_idiosyncrasy/distributions.ipynb`, code cell 3. The first filter excludes observed effects touching the rescaled floor exactly; the control pool uses values strictly greater than 1.0. In the downloaded data there are no rescaled fitness values below 1.0, so that code behavior is equivalent here to dropping all values at or below 1.0. The per-mutation effects are log fitness ratios, and each control uses the same number of random genotype pairs as observed backgrounds, sampled with replacement and seeded by `n**2 + 3`.

I independently rebuilt mutation backgrounds from the sequence rows, retained the authors’ 1-based positions 1–69, and checked the directed mutation key count **before calculating any index**. The script printed `verified_directed_mutations_before_index_calculation=828`; see `papers/Lyons2020/sources/reproduce_idiosyncratic_index_codex.py`, lines 39–59. The run found 65,537 source rows; 28,530 rows had `Fit/0.5 > 1.0`, 37,007 were exactly 1.0, and none were below 1.0. These are local data counts, **NOT STATED IN PAPER.** The row-count fields are emitted by that script at lines 117–125. The 72-character sequence check is at script line 34 and was confirmed directly from the saved source data.

An initial helper run stopped at a sequence-length assertion because I had assumed 70 characters. No statistic was computed in that attempt. The source rows contain 72-character sequences; after correcting the check, the script verified the 828 directed mutations and completed. This was an input-inspection correction, not a preprocessing variant.

## Reproduction and GraphFLA comparison

All values in this table are calculated outputs from `reproduce_idiosyncratic_index_codex.py` (lines 77–159); **NOT STATED IN PAPER.** The paper’s printed result remains the verbatim quote in `record.json`.

| Calculation | Result | Interpretation |
|---|---:|---|
| Authors’ seeded, matched-size resampled control; all directed mutations | 0.6121000731 | Rounds to the printed 0.612 |
| Standard error from population SD of the 828 mutation ratios divided by √828 | 0.0045646619 | Rounds to the printed 0.005 SE |
| Same mutation SDs, GraphFLA analytic baseline on the full viable input | 0.5942481263 | Analytic-vs-resampled baseline shift: −0.0178519468 |
| Resampled result after applying `min_pairs=3` | 0.6121000731 | Zero mutations excluded; minimum retained background count is 3; contribution of this filter is 0 |
| Actual `global_idiosyncratic_index` call after `build_from_data` | 0.5873601678 | GraphFLA mismatch |
| Total GraphFLA minus independent paper procedure | −0.0247399053 | Combined gap |

The known baseline difference explains −0.0178519468 of the gap when both procedures use the same full viable input. `min_pairs=3` explains 0 because every mutation has at least 3 retained backgrounds. There is an additional input-handling effect in the actual GraphFLA call: `build_from_data` removed 3,903 isolated viable configurations, leaving 24,627 of the 28,530 rows passed in. The run emitted the warning “3903 isolated configuration(s) with no neighbors detected and removed from the landscape graph” (`graphfla/utils.py`, lines 221–225). The analytic baseline SD rose from 0.2780709971 on all viable inputs to 0.2813319290 after graph construction, shifting the index by another −0.0068879585. The two shifts sum to the measured −0.0247399053 total gap.

The implementation locates fitness with `data = landscape.get_data()` and uses `std_baseline = float(np.sqrt(2.0) * np.std(f))` (`graphfla/analysis/epistasis/idiosyncrasy.py`, lines 230–238); its API documents `min_pairs: int, default=3` at lines 216–218. The builder’s isolated-node removal is visible in `graphfla/utils.py`, lines 221–225. Recomputing the analytic ratio on the post-build GraphFLA data gives 0.5873601678, exactly matching the function output within the script’s numeric tolerance.

No preprocessing variants were tuned or selected. The independent procedure reproduces the printed target at its precision; the GraphFLA mismatch is retained as a useful result. The detailed fields, including input hashes and reason, are in `record.json`.

Runtime side effect: importing GraphFLA warned that the default Matplotlib cache directory was not writable and built a temporary font cache under the system temp directory. The metric run completed. I made no changes in the GraphFLA repository.
