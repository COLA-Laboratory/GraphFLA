# Baeza-Centurion et al. (2019): raw validation notes

## Bibliography and acquired sources

Crossref's DOI work record confirms the publisher title, the five-author order, journal, year, volume, and issue. The selected bibliography has the same title and authors; its page range is abbreviated. Crossref's `page` field is `"549-563.e23"` (Crossref work record, `message.page`), while [selected_references.bib](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/selected_references.bib:188) gives `549--563`.

I read the existing `papers/BaezaCenturion2019/` dossier first; it had only empty `logs/`, `figures/`, and `sources/` directories. I acquired the publisher's supplementary PDF and Table S3 workbook, plus Crossref metadata. The publisher API response saved with a `fulltext` filename contains core metadata only, not the article body. Direct Cell/ScienceDirect PDF retrieval returned 403, but the publisher article page was accessible for text review. These details and all successful downloads are recorded in `papers/BaezaCenturion2019/acquisition_codex.jsonl`.

The workbook sheet is titled `"Combinatorially Complete"` (publisher supplementary workbook, worksheet title), and its genotype table includes the `"Mean.PSI"` field (worksheet header row). The local CSV's compact sequence codes match the publisher table after projecting each full sequence onto its variable positions; every local position column maps to the corresponding source coordinate. Its `fitness` values agree with `Mean.PSI` at the workbook's stored precision. This confirms that the `Centurion2019.csv` name refers to this paper's FAS exon 6 landscape, despite omitting “Baeza-”. The exact paper count and position quotes, plus the GraphFLA results, are in the `n_configs` and `n_vars` entries of `record.json`.

## Overlap decisions

- `n_configs` and `n_vars` are direct structural comparisons. The full CSV genotype set agrees with the publisher workbook, and `DNALandscape` returns matching configuration and variable counts. No fitness transformation or filtering was applied.
- `fitness_distribution` is a conceptual candidate only: the paper describes a bimodal PSI distribution and threshold shares, while GraphFLA's function returns scale-free moments and ratios. These are different outputs, so no reproduction was attempted.
- `global_idiosyncratic_index` is marked `definition_match: "unknown"` because the paper examines context-dependent mutation effects, but its fitted scaling-law parameter is not a reported global idiosyncratic index. The GraphFLA metric is under review, and a comparable paper-level scalar is `NOT STATED IN PAPER.` No index value was computed as a reproduction.
- The higher-order analysis uses a Walsh-Hadamard decomposition and reports RMSE through Figure S5E. The figure's values are marked `plotted`; I did not read them. GraphFLA's `higher_order_epistasis` is an R-squared from polynomial regression, so the definitions differ. A printed variance-explained-by-order value is `NOT STATED IN PAPER.`
- A peak or local-optimum count is `NOT STATED IN PAPER.` Epistasis class fractions are `NOT STATED IN PAPER.` The paper's fitted mutation-response curves, model RMSEs, and selected pairwise model terms are not standalone GraphFLA landscape statistics.

For metric definitions, the live GraphFLA implementation was checked. Its fitness-distribution docstring says `"unitless statistics about the fitness distribution"` and lists skewness, kurtosis, and CV ([fitness.py](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/analysis/fitness.py:12)). Its higher-order routine describes a polynomial-regression R-squared ([higher_order.py](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/analysis/epistasis/higher_order.py:1)); its epistasis classifier uses four-node motifs ([motifs.py](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/analysis/epistasis/motifs.py:215)).

## Reproduction execution

The reproduction script is `papers/BaezaCenturion2019/logs/reproduction_codex.py`. It was run from the validation directory with:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/arwen/Documents/GitHub/GraphFLA python papers/BaezaCenturion2019/logs/reproduction_codex.py
```

It queried `graphfla.analysis.list_metrics()` and built `DNALandscape` from `pos1` through `pos11` and `fitness`, using `epsilon=0` and `verbose=False`. The script also checked the source genotype set and `Mean.PSI` correspondence. No GraphFLA tests were run.
