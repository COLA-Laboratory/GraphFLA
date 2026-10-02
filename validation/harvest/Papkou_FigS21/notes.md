# Papkou et al. 2023 — Fig. S21 raw reproduction notes

## Bibliography verification

Crossref's DOI record gives the title “A rugged yet easily navigable fitness landscape” and the authors Andrei Papkou, Lucia Garcia-Pastor, José Antonio Escudero, and Andreas Wagner ([Crossref DOI record](https://api.crossref.org/works/10.1126/science.adh3860)). The supplied corpus author list, “Andrei Papkou and Aviv Regev and Joanna Masel,” is incorrect. The machine-readable record uses the corrected Crossref authors.

## Paper method, quoted with locators

Supplementary Materials, Methods, “Higher-order epistasis,” printed p. 18:

> “We determined higher-order epistasis with an extension of the Walsh-Hadamard transformations for non-biallelic landscapes (134), i.e.,”

> “ε = V · H · y”

The same section defines the vectors and matrices:

> “where ε is a vector of epistatic coefficients of different orders, H is a Hadamard matrix, V is a weighting matrix, and y is a vector of fitness values.”

The paper therefore describes a Walsh-Hadamard basis extended to non-biallelic landscapes. It does not describe the analysis as a one-hot/reference-dummy regression.

The paper explicitly identifies the genotype subset and its size. Supplementary Materials, Methods, “Higher-order epistasis,” printed pp. 18–19:

> “Therefore, we focused on the subset of only Asp27 and Glu27 variants, which include most functional variants and all high fitness peaks (N=16,353).”

> “The resulting matrix dimensionality is 16,353×16,353.”

The Fig. S21 caption independently states, printed p. 41:

> “This genotype subset represents the most comprehensive landscape with 89% of functional variants (N=16,353).”

The reference genotype or reference allele for the transform is **NOT STATED IN PAPER.**

For regularization and model selection, Supplementary Materials, Methods, “Higher-order epistasis,” printed p. 19:

> “To avoid overfitting, we used a penalized LASSO regression with the help of the scikit-learn Python library version 1.30 (135).”

> “We determined the optimal regularization parameter λ=10-5 using k-fold cross-validation with k=5 and five different random initializations.”

> “In addition, we fitted reduced models which contain predictors only up to a specific order of epistasis.”

> “We quantified model performance using the coefficient of determination R-squared and the root mean square error (RMSE).”

The paper selects the LASSO penalty by cross-validation. It does not say whether the R-squared values shown for the reduced models are calculated on training data, held-out folds, or a separate test set: **NOT STATED IN PAPER.** Thus, “the paper cross-validates R-squared” would overstate what the Methods say. The Methods do establish that the regression is LASSO-regularized, and the Fig. S21 caption says coefficients can be zeroed by that penalty:

> “However, all of the seventh-order coefficients are set to zero by a penalized regression (see (G)) and not shown here.”

> “(G) depicts the coefficients obtained from the full lasso model (regularization penalty λ = 10-5).”

The Fig. S21 caption describes panels H–K as follows, printed p. 41:

> “(H), (I), (J), (K) present the goodness of fit for lasso regression models restricted up to specific epistatic orders.”

This establishes that the fit is on the Asp27/Glu27 subset described above, and that the plotted models use LASSO. The paper's exact evaluation split for the R-squared values remains **NOT STATED IN PAPER.**

## Target labels and outcomes

The four panel labels below are quoted as provided in the task brief and located to Fig. S21H–K, printed p. 41. The local SI text extraction contains the caption but not the embedded panel annotations; the publisher PDF request returned an HTML security page. I therefore did not independently verify the image text in this session. Each target is recorded as printed per the assignment.

- Panel H, order 1: “R2 = 0.28” — definition-incompatible; no GraphFLA R-squared computed.
- Panel I, order 2: “R2 = 0.78” — definition-incompatible; no GraphFLA R-squared computed.
- Panel J, order 3: “R2 = 0.93” — definition-incompatible; no GraphFLA R-squared computed.
- Panel K, order 4: “R2 = 0.97” — definition-incompatible; no GraphFLA R-squared computed.

The assignment explicitly identifies the individual values in panel E as plot-only. The SI caption says:

> “(E) displays the R-squared values.”

No panel E point was digitized. The reported panel H–K targets above are the only R-squared targets used here. Fig. S21A's coefficient magnitudes and panel G's coefficient proportions/counts have no direct GraphFLA output; panel F reports RMSE, for which the listed GraphFLA inventory has no matching metric.

## Input subset check

The paper's quoted target is “N=16,353.” I reused the specified processed CSV at /Users/arwen/Documents/GitHub/GraphFLA/data/BioSequence/Papkou2023_DHFR.csv (SHA-256 fca2fb47cad417a049698f43b20f90b320a4927418d502e527ad112ea46c82da; 135,178 rows, as recorded in papers/Papkou2023/data_audit.json). I applied the dossier's processed-to-biological nucleotide mapping and filtered the second codon for Asp/Glu. That file contains 16,332 matching sequences, so it fails the required exact-subset check by 21 rows.

As a diagnostic only, the dossier's full RAW CSV (/Users/arwen/Documents/GitHub/GraphFLA/data/BioSequence/Papkou2023_DHFR_RAW.csv, SHA-256 fcd57493a10caecd4f12d510809fc96ece4f98b09ac5a9c5769d35179ccaaf05; 261,333 rows in the dossier audit) contains 16,353 matches under the same filter. The 21 sequences in that set are absent from the specified processed input. The RAW file was not substituted for or used to supplement the assigned input. Because the exact 16,353-row subset was not present in the assigned input, I stopped before calculating any GraphFLA R-squared.

## GraphFLA comparison

The read-only implementation inspected was /Users/arwen/Documents/GitHub/GraphFLA/graphfla/analysis/epistasis/higher_order.py.

Relevant source excerpts and locators:

> “This function uses polynomial regression with degree=order to model interactions up to the specified order.” — lines 39–41

> encoder = OneHotEncoder( / drop="first" — lines 89–93

> poly = PolynomialFeatures( / interaction_only=True — lines 103–108

> model = LinearRegression(n_jobs=n_jobs) — line 109

> model.fit(X_poly, y) — line 115

> r2 = r2_score(y, y_pred) — line 127

GraphFLA fits ordinary unregularized linear regression and scores predictions on the same configurations used to fit it. For the nucleotide landscape, it uses one-hot encoded variables with the first level dropped, then interaction-only polynomial features. This differs from the paper's Walsh-Hadamard coefficient basis and penalized LASSO; GraphFLA has no coefficient thresholding. The paper's held-out/in-sample status remains **NOT STATED IN PAPER.** These definition differences alone make numeric agreement an invalid reproduction. The assigned input's 21-row deficit is an additional blocker. No model orders were run, and no preprocessing or model parameters were tuned.

## Acquisition notes

The dossier already held the main paper and Crossref artifacts. The direct publisher SI URL, https://www.science.org/doi/suppl/10.1126/science.adh3860/suppl_file/science.adh3860_sm.pdf, returned HTTP 403. The corresponding existing attempt artifact is HTML security-verification content, not a PDF. I reused the pre-existing SI text extraction from /Users/arwen/Documents/GitHub/GraphFLA/_verify/papers/papkou_science_SM.txt, copied it into the dossier as papers/Papkou2023/sources/Papkou2023_science_adh3860_SM_codex.txt, and appended that local-copy entry to papers/Papkou2023/acquisition_codex.jsonl. Hashes and paths for Crossref and the publisher attempt are in that same log.
