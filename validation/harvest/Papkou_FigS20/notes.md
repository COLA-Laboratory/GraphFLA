# Papkou et al. (2023), Supplementary Fig. S20

## Bibliography and acquisition

The DOI title and author list were verified against the saved Crossref record at [Papkou2023_crossref_adh3860_codex.json](../../papers/Papkou2023/sources/Papkou2023_crossref_adh3860_codex.json). The supplied `selected_references.bib` entry is wrong: it lists Andrei Papkou, Aviv Regev, and Joanna Masel. Crossref returns Andrei Papkou, Lucia Garcia-Pastor, José Antonio Escudero, and Andreas Wagner for DOI `10.1126/science.adh3860`.

The publisher SI PDF request returned HTTP 403 and saved an HTML verification page. I reused the byte-identical SI text derivative already present in the prior verification dossier; its SHA256 is `abd424ad6fee922b61e34ccb961490e28f820cdc17a4057dc123dc783b293072`. The native SI artwork was not available for independent visual inspection. The archived author notebooks and bypass RDS were available; each range download and its SHA256 is logged in `papers/Papkou2023/acquisition_codex.jsonl`.

## Verbatim quotes and figure checks

The short published-SI excerpts and locators are stored in `record.json` alongside their targets. For Fig. S20A, the caption's relevant phrase is “between variant ab and AB” (Fig. S20 caption, printed p. 40); the overlap record also stores “accessible paths.” The caption describes the path-count framing, while the exact per-class `2/1/0` numerals are not stated in the caption text: **NOT STATED IN PAPER.** The task description supplies that mapping, and GraphFLA's controlled-square fixtures are consistent with it. In the code, `tests/_landscapes.py:73-78` names the magnitude, sign, and reciprocal-sign squares; `tests/test_metrics.py:83-97` maps those fixtures to the corresponding `classify_epistasis` fields. I inspected those fixtures but did not run tests.

The paper's network-method passage uses “fitness-increasing mutation” (Supplementary Methods, printed p. 15). The preprocessing threshold appears as “-0.507774” (Supplementary Methods, printed p. 14). I used the dossier's recorded preprocessing without trying alternatives: the processed CSV, inverse alphabet decoding `A/C/G/T -> A/G/T/C`, `tau=-0.507774`, `filter_mode='both'`, and `epsilon=0`.

For Fig. S20B, the caption's sample size is recorded in the matching JSON overlap (Fig. S20B caption, printed p. 40). The exact three class counts are **NOT STATED IN PAPER** as printed numbers; the figure bars are plotted. The archived author notebook supplies the values used to construct them. Its cell 6 output is:

```text
  count  type                  Freq
1 408065 magnitude\n& additive 0.5512820
2 246943 single\nsign          0.3336116
3  85203 reciprocal\nsign      0.1151064
```

Locator: `papers/Papkou2023/sources/zenodo_8228920/14.reciprocal_sign_epistasis/01.count_different_types_of_epistasis.ipynb`, cell 6 stored output. The same cell computes `Freq = count/ (sum(count))`; cell 7 uses `y=Freq` for the bars. Thus the author data yield 55.1282%, 33.3612%, and 11.5106% respectively, consistent with the three plotted bar heights. I did not digitize the bars.

The exhaustive GraphFLA run used the same file recorded in the dossier (SHA256 `fca2fb47cad417a049698f43b20f90b320a4927418d502e527ad112ea46c82da`). `classify_epistasis(..., sample_cut_prob=0)` returned 0.551281999321815 magnitude, 0.33361163235888147 sign, and 0.11510636831930356 reciprocal sign. Multiplying by the caption denominator and rounding to integer reproduces 408,065 / 246,943 / 85,203 exactly. The successful run command was `MPLCONFIGDIR=/Users/arwen/Documents/GraphFLA-validation/2026-09-30/papers/Papkou2023/.mplconfig_codex PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/arwen/Documents/GitHub/GraphFLA python papers/Papkou2023/reproduce_fig_s20_codex.py`.

For Fig. S20D, the caption defines the plotted y-value with the verbatim excerpts stored in the matching JSON overlap (Fig. S20D caption, printed p. 40). Supplementary Text S3 prints the rounded plateau (printed p. 20). I did not read any values from the plotted curve.

The author bypass notebook's cell 12 stores these outputs:

```text
[1] 0.2274216
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
  3.000   3.000   4.000   4.354   5.000  29.000
```

Locator: `papers/Papkou2023/sources/zenodo_8228920/14.reciprocal_sign_epistasis/02.extradimentional_bypass_codex.ipynb`, cell 12 stored output. Cell 4 gives `[1] 85203`. The accompanying `bypass_squares_rec_epi_codex.rds` contains the source data used to construct the cumulative curve.

The author notebook's cell 10 searches an outgoing shortest path from each motif's second-highest to highest variant after removing the two lowest motif vertices. GraphFLA's `extradimensional_bypass` makes the same endpoint choice and uses `distances(..., mode="out")` (`graphfla/analysis/epistasis/motifs.py:465-519`). Its landscape edges point only uphill, so a path from the second-highest motif variant cannot traverse either lower valley; the explicit valley removal does not change this path test.

GraphFLA returned `bypass_proportion=0.22742156966304003`, `motifs_with_bypass=19377`, `total_motifs=85203`, and `average_bypass_length=4.353769933426227`. The proportion and mean agree with the author notebook to its stored precision. Because the longest finite author shortest bypass length is 29, GraphFLA's unbounded any-bypass proportion equals the cumulative curve's full plateau at maximum length `L=29` (and remains equal for larger cutoffs). GraphFLA returns this endpoint and a mean length, not the full length-resolved curve. This is an endpoint comparison, not reproduction of every plotted point.

## Outcome and limitations

Panel B is an exact class-count reproduction. Panel D reproduces the printed whole-percent plateau and the author-data endpoint to stored precision. Panel A's path-count framing agrees with GraphFLA's directed square taxonomy, although the exact 2/1/0 mapping was not independently checked against the native SI image because the publisher PDF was blocked. No preprocessing variants were tried, and no plotted point was digitized.

Execution note: the first script launch stopped at its input-hash preflight because I had mistyped the expected SHA256 in the script; it did not build the landscape or call a metric. I corrected the expected hash to the dossier's recorded value and the later run completed. That failed import attempt caused Matplotlib to place a font cache in the OS temporary directory outside this workspace. For the successful run, I redirected `MPLCONFIGDIR` to `papers/Papkou2023/.mplconfig_codex`; `PYTHONDONTWRITEBYTECODE=1` was set on both GraphFLA launches, so Python did not write bytecode under the GraphFLA repository.
