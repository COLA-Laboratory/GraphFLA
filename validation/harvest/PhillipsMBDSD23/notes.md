# PhillipsMBDSD23 — acquisition and reproduction notes

## Bibliography verification

Publisher record, “Cite this article”:

> 1. Angela M Phillips
> 2. Daniel P Maurer
> 3. Caelan Brooks
> 4. Thomas Dupic
> 5. Aaron G Schmidt
> 6. Michael M Desai
>
> (2023)
>
> Hierarchical sequence-affinity landscapes shape the evolution of breadth in an anti-influenza receptor binding site antibody
>
> eLife 12:e83628.
>
> https://doi.org/10.7554/eLife.83628

Locator: [eLife publisher article page](https://elifesciences.org/articles/83628), “Cite this article.” The title, year, and six-author list in selected_references.bib match. The supplied BibTeX entry omits the DOI. Bibliography verification used the publisher page, not the supplied author list as a search key.

The supplied entry prints:

> author  = {Angela M Phillips and Daniel P Maurer and Caelan Brooks and Thomas Dupic and Aaron G Schmidt and Michael M Desai},
> title   = {Hierarchical sequence-affinity landscapes shape the evolution of breadth in an anti-influenza receptor binding site antibody},
> volume  = {12},
> journal = {eLife},
> year    = {2023},
> pages   = {e83628},

Locator: selected_references.bib, PhillipsMBDSD23 entry, lines 94–102. That entry has no doi field.

The article’s Data availability section says:

> “Data and code used for this study are available at https://github.com/amphilli/CH65-comblib, (copy archived at swh:1:rev:cea336dfd05fa31d675f79038428bf7f0d177e78).”

Locator: Data availability, p.27. The downloaded repository archive contains that commit.

## Measurement and local-data audit

The paper describes its assay and reported phenotype as follows:

> “We then use Tite-Seq (Adams et al., 2016), a high-throughput method that couples flow cytometry with sequencing, to measure equilibrium binding affinities to three H1 strains bound by CH65.”
>
> “We log-transform the binding affinities and report -logKD, which is proportional to the free energy change of binding (and is thus expected to combine additively).”

Locator: Results, Figure 1 discussion, pp.4–5.

The publisher Figure 1 source-data workbook has a KD sheet whose header is:

> “geno, MA90_rep1, MA90_rep2, MA90_mean, MA90_sem, SI06_rep1, SI06_rep2, SI06_mean, SI06_sem, G189E_rep1, G189E_rep2, G189E_mean, G189E_sem”

Locator: Figure 1—source data 1, workbook sheet KD, row 1. Its description is “CH65 library expression and -logKD to MA90, MA90-G189E, and SI06.” Locator: Figure 1 caption, p.5.

The author browser CSV header includes “MA90_log10Kd”, “SI06_log10Kd”, and “G189E_log10Kd”. Locator: author repository snapshot, CH65_browser/data/CH65.csv, header row. The supplied GraphFLA CSVs reproduce those browser columns as fitness, so their sign is opposite the paper’s plotted -logKD values. For the reproduction, I used the author QC-processed table’s positive -logKD means. When auditing the supplied CSVs, I compared -fitness with the publisher’s positive -logKD means.

The paper states how non-binders and poor fits were handled:

> “Following the KD,s inference, non-binding sequences with KD,s<6 or As – Bs <1 were pinned to the titration boundary with -logKD,s = 6. Subsequently, KD,s values resulting from poor fits (r2 < 0.8, σ > 1) were removed from the dataset, KD,s were averaged across biological replicates, and KD,s with large SEM (>0.5 log units) were excluded from subsequent analyses.”

Locator: Materials and Methods, “Data quality and filtering,” p.20.

Thus the boundary value for non-binders is a censored/pinned value at -logKD = 6. The paper removes poor-fit and high-SEM records from subsequent analyses. The local browser CSVs retain additional values absent from the QC-processed table and omit some accepted records; numeric values shared with the publisher source data agree after sign correction. This is my input audit, not a paper claim: five accepted G189E rows and three accepted SI06 rows are blank in the local CSVs, and there are zero sign-corrected mismatches among shared accepted values. The audit is implemented in the reproduction script’s audit_local function. I used the author’s processed 20221008_CH65_QCfilt_REPfilt.csv from the repository archive for the GraphFLA runs. Its uncompressed SHA-256 is d526633e53a0c84f6739109d123438db6cc25f1e1e277012082f23156607811f.

The paper prints the common plotted count and per-antigen retained counts:

> “N = 62,926 after filtering poor KD measurements from the Tite-Seq data (see 'Materials and methods').”

Locator: Figure 1 caption, p.5.

> “This filtering retained 65,530, 63,840, and 64,619 genotypes for the MA90, G189E, and SI06 Tite-Seq experiments, respectively (Figure 1—source data 1).”

Locator: Materials and Methods, “Data quality and filtering,” p.20.

GraphFLA n_configs reproduced all four counts exactly from the author QC-processed input: the shared intersection and each antigen-specific set.

## Epistasis and model-fit overlaps

The main text reports fourth-order MA90 and fifth-order G189E/SI06 models:

> “Using a cross-validation approach, we find that the optimal order model for affinity is fourth-order for MA90 and fifth-order for MA90-G189E and SI06, and we report coefficients at each order from these best-fitting models (Figure 3A).”

Locator: Results, “Structural and biophysical basis of epistasis in CH65,” p.7.

The article’s model-fitting description is:

> “We then average performance across the eight folds, select the order that maximizes the prediction performance, and retrain the entire dataset on a model truncated at this optimal order, this time by ordinary least-squares regression.”

Locator: Materials and Methods, “Epistasis analysis,” p.21.

Figure 3—source data 1 contains Performance values for the fitted models:

- MA90 worksheet, cells A2:B2: “Performance: 0.986822098380123”.
- G189E worksheet, cells A2:B2: “Performance: 0.98188082017006”.
- SI06 worksheet, cells A2:B2: “Performance: 0.965915764698917”.

The author archive’s corresponding coefficient files print:

> “Performance: 0.9868220983801239”
>
> “Performance: 0.9818808201700604”
>
> “Performance: 0.9659157646989173”

Locators: author repository snapshot, Epistasis_Inference/MA90/biochemical/CH65_MA90_102022_4order_biochem.txt; Epistasis_Inference/G189E/biochemical/CH65_G189E_102022_5order_biochem.txt; Epistasis_Inference/SI06/biochemical/CH65_SI06_102022_5order_biochem.txt, lines 1–2. GraphFLA higher_order_epistasis produced those same full-precision values. The publisher workbook’s MA90 cell differs from the GraphFLA and author-output value only in its last stored digit; the record marks that tiny difference as a mismatch.

The paper prints the epistatic variance shares:

> “These mutations interact epistatically for each of the three antigens, though the magnitude of epistasis is higher for SI06 (explaining ~34% of the variance in KD, relative to ~24% for MA90, and ~25% for MA90-G189E, see Figure 3—figure supplement 6).”

Locator: Results, “Structural and biophysical basis of epistasis in CH65,” p.7.

It further states:

> “In the statistical epistasis inference, the coefficients at different orders are statistically independent and so we partition the variance explained by the model for each interaction order (Figure 3—figure supplement 6).”

Locator: Materials and Methods, “Epistasis analysis,” p.21.

The author variance-partition script makes the calculation explicit. It states “var_expl = np.array(order_r2_G189E)/order_r2_G189E[-1]” and “total_ep = 1-var_expl[0]” (Figures/Epistasis_figures/frac_order_plots.py, lines 186 and 192). The script sets “order_MA90 = 5” (line 36), “order_G189E = 6” (line 45), and “order_SI06 = 5” (line 54). For SI06 it states “df_SI06 = df.dropna(subset=['SI06_mean'])” and “df_SI06 = df_SI06.loc[df_SI06['pos'+mut] == 1]” (lines 223–225), then builds predictors from the remaining SI06 mutation positions (lines 227–230).

GraphFLA documents higher_order_epistasis as returning “The R² score representing the fraction of variance explained by polynomial terms up to the specified order” (GraphFLA graphfla/analysis/epistasis/higher_order.py, lines 33–35). I therefore compared 1 − R²(order 1)/R²(maximum order), consistent with the authors’ independent-order partition. The GraphFLA shares were 0.2372983 for MA90 and 0.3416469 for SI06, matching the printed approximations to whole-percent precision. The exact G189E variance-share run was not attempted: the author partition uses order 6, and the dense GraphFLA design matrix would need about 7.6 GB before regression workspace while about 8.2 GB was available. An order-5 sensitivity returned 0.2492679, but it is not substituted for the author’s order-6 statistical analysis.

## Non-overlaps: binding fractions, paths, and peaks

The Results text says:

> “While the entire library binds MA90, ~83% of variants bind MA90-G189E and ~51% of variants bind SI06 (Figure 1C).”

Locator: Results, “CH65 sequence-affinity landscape,” p.6.

These are antigen-binding fractions. No GraphFLA metric directly returns a thresholded binder fraction, so I did not treat them as a reproduction target. Figure 1C says “total number of binding variants (N) indicated on plot” (Figure 1 caption, p.5); I did not read values from the plot.

The pathway section describes weighted trajectory likelihoods:

> “the probability of any mutational step is higher if -logKD increases and lower if -logKD decreases”

Locator: Results, “Likelihood of mutational pathways to CH65,” p.11.

Figure 5’s caption says:

> “Total log probability (in arbitrary units) of all mutational paths from the unmutated common ancestor (UCA) to CH65 (left) or paths from the UCA to CH65 that pass through I-2 (right), assuming specific antigen selection scenarios are shown.”

Locator: Figure 5 caption, p.11. The exact numerical fraction of accessible paths compatible with GraphFLA global_optima_accessibility is NOT STATED IN PAPER. Figure 5 values are plotted and were not read. GraphFLA defines global_optima_accessibility as “the fraction of configurations in the landscape that can reach the global optimum via any monotonic, fitness-improving path” (GraphFLA graphfla/analysis/navigability.py, lines 120–125). That is not the paper’s scenario-weighted log probability for paths with fixed endpoints.

A printed peak/local-optimum count is NOT STATED IN PAPER. No n_lo or local_optima_ratio target was available.

## Reproduction command and provenance

The reproduction script is papers/Phillips2023/PhillipsMBDSD23_reproduce_codex.py. It reads the author processed CSV from the downloaded archive, the Figure 3 source-data workbook for its target values, and the local CSVs for provenance checks. It ran with PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=/Users/arwen/Documents/GitHub/GraphFLA. Acquisition URLs, HTTP status, hashes, and saved paths are in papers/Phillips2023/acquisition_codex.jsonl.
