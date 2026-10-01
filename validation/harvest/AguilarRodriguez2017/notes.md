# Aguilar-Rodríguez, Payne & Wagner (2017)

Raw validation notes for `n_lo`, epistasis-class fractions, and the printed-versus-plotted accessibility results.

## Bibliography and sources

The Nature publisher citation gives the title *A thousand empirical adaptive landscapes and their navigability*, DOI `10.1038/s41559-016-0045`, and authors José Aguilar-Rodríguez, Joshua L. Payne, and Andreas Wagner. The task bibliography agrees with the publisher page. I reused the existing main PDF and supplementary-information PDF in `papers/_core_lit_navigability/sources/`; their original URLs and checksums remain in that dossier's `acquisition_codex.jsonl`.

The publisher's **Data availability** section says: “All data analysed during this study are available in public repositories; accession information is provided in Supplementary Tables 1 and 2.” (Methods, printed p. 7.) The Methods identify PBM data “from the UniPROBE database” and “from the CIS-BP database.” (Methods, printed p. 7.) The required accession workbook was not available in the existing dossier: its publisher URL returned HTTP 200 with an HTML security challenge rather than an XLSX. This failed acquisition is preserved in `papers/_core_lit_navigability/acquisition_codex.jsonl` and referenced in the new dossier log. No complete per-factor E-score input was assembled from the public databases.

## Peak-count result

The only printed corpus-wide peak-count summary I found is: “42% of the empirically derived landscapes (478 of 1,137) have multiple peaks of unequal height ... peak numbers ranging from 2 to 36.” (Results, “Landscape navigability: the number of peaks,” printed p. 2.) This is a useful aggregate overlap with GraphFLA `n_lo`, but the input needed to recompute it was not recovered.

The paper's peak procedure includes plateaus: “as an individual sequence or as a member of a plateau.” (Methods, printed p. 8.) It compares fitness using a transcription-factor-specific noise threshold: “For each TF, we calculated δ as the residual standard error” and “Otherwise, we considered that we could not differentiate between the two affinity values.” (Methods, printed p. 7.) This differs from the task's required `epsilon=0` baseline, where only exact ties are neutral. I therefore leave `definition_match` unknown and mark the aggregate target input unavailable; I did not tune epsilon or reconstruct values from the Figure 2a histogram.

Requested summary-statistic checks:

- Printed mean number of peaks per landscape: **NOT STATED IN PAPER**.
- Printed median number of peaks per landscape: **NOT STATED IN PAPER**.
- Printed count of single-peaked landscapes: **NOT STATED IN PAPER**.
- Printed overall mean or median peak-accessibility fraction across the corpus: **NOT STATED IN PAPER**.

The paper prints the count of landscapes with multiple peaks and the range of peak counts within that group. It does not print the complementary single-peak count; I have not substituted a calculated complement for a published target.

## Epistasis-class result

The Results section prints: “sign epistasis is more frequent and affects 4.7% of squares, on average.” (Printed p. 3.) The same section partitions the classes as “magnitude, simple sign and reciprocal sign.” (Printed p. 3.) This is a candidate overlap with GraphFLA `classify_epistasis`, specifically a combined sign fraction averaged over landscapes. The complete per-factor input and TF-specific δ values were not acquired, so the class fraction was not run. A reproduction would also have to match the paper's per-square thresholding and landscape averaging.

## Accessibility results

Figure 2c is plotted data. Its caption says: “For each of the 1,137 adaptive landscapes we show the mean (symbols) and standard deviation (error bars)” for the fraction of accessible paths to the highest-affinity site. (Figure 2c caption, printed p. 3.) Those are per-landscape plotted symbols, not printed numeric values; I did not read any point from the figure. No aggregate mean or median over the landscapes is stated in the paper.

The main text separately checks whether expression rises along already-accessible paths in the two yeast TF subsets: “For Gcn4, all 71 accessible mutational paths ... for Fhl1 all except one of the 37 accessible mutational paths.” (Results, “Gene expression reflects landscape topography,” printed p. 5.) These are expression-monotonicity checks on selected paths, not a corpus-wide fraction of paths accessible to a global peak, so they are recorded as definition-incompatible rather than an accessibility target.

## Reproduction status

No GraphFLA calculation was run for this paper. The `n_lo` and `classify_epistasis` aggregate targets both require the complete transcription-factor input set; the accessibility values are plot-only. The exact limitation and the failed Supplementary Table 1 retrieval are recorded in `record.json` and `papers/_navigability_lit/acquisition_codex.jsonl`.
