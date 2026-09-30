# WongKK18 acquisition and reproduction notes

## Bibliography and acquired material

- Verified the title, DOI, authors, journal, year, volume, and pages against the downloaded Crossref JSON record. The registry author list matches: Mandy S. Wong, Justin B. Kinney, and Adrian R. Krainer. The supplied title is lowercased, braces the initial Q, and uses ASCII 5' rather than the publisher’s 5′.
- Reused the dossier’s existing `papers/Wong2018/sources/fullTextXML.xml`; it was already present, so it was not overwritten. Crossref metadata and the authors’ summary PSI file were newly saved with `_codex` suffixes and logged in `papers/Wong2018/acquisition_codex.jsonl`.
- Downloaded the authors’ processed 9-mer table from [15_splicing/local/output/psi_9nt.txt](https://github.com/jbkinney/15_splicing/blob/master/local/output/psi_9nt.txt) to `papers/Wong2018/sources/Wong2018_psi_9nt_codex.txt`. It has sequence plus BRCA2, IKBKAP, and SMN1 PSI and standard-error columns. This is an author-repository output, not a publisher “Source Data” file.
- The paper’s Data and Software Availability section (P66) names SRA, the author code repository, and Mendeley Data. The rendered Mendeley page had no file rows in its Files section. The old dossier supplement file is an XML error response, and the direct PMC PDF request returned 404; the full-text XML was sufficient for the main-text quotes and targets below.

## Quotes and locators

- Abstract: “NNN/GYNNNN”. The printed design-space count is quoted in `record.json`; the local CSVs establish the exact position alphabets.
- STAR Methods → Quantification and statistical analysis → PSI quantification, P59: “PSI = 100*r/rcon”.
- Figure 2C caption: “100 PSI” for the consensus sequence in each context. This is the scale check against the local `fitness` column.
- STAR Methods → Pairwise dependency, P61 and Figure 4 caption: the threshold, ridge, residual, and GU-only method fragments are quoted in the `record.json` R² overlap.
- STAR Methods → Construction and sequencing of libraries, P53, records a “low-quality” data exclusion upstream of the summarized PSI table; the local NaNs were not imputed.
- The exact short quotes for all printed numeric targets, including the pairwise R² percentages, are in each `record.json` overlap’s `quote` field with its locator.

## Sequence encoding and fitness meaning

The three read-only inputs under `/Users/arwen/Documents/GitHub/GraphFLA/data/BioSequence/` each contain 32,768 rows and 32,768 unique RNA sequences of length 9. The observed alphabets are: `pos1–pos3 = A/C/G/U`, `pos4 = G`, `pos5 = C/U`, and `pos6–pos9 = A/C/G/U`. In order, `pos1–pos3` are the three exon-side bases, `pos4` is the invariant +1 G, `pos5` is the +2 C/U choice, and `pos6–pos9` are the remaining four intron bases. That gives seven four-letter positions and one two-letter position, matching the stated 32,768 sequence space. RNA letters are used, so `RNALandscape` was the GraphFLA class.

The CSV `fitness` is not final PSI. It is the paper’s pre-consensus relative splicing ratio `r`. Evidence: the consensus-row `fitness` values are BRCA2 20.60801036, IKBKAP 130.4247469, and SMN1 13.65484679, whereas the authors’ `psi_9nt` table reports 100 for each consensus row. Applying the paper’s stated formula `100*r/rcon` to every finite local CSV row matches the authors’ PSI table: maximum absolute error is 3.87e-8 for BRCA2, 7.04e-8 for IKBKAP, and 1.09e-7 for SMN1. This scale conversion is directly justified by P59; no fitted or tuned transformation was used.

The local CSV files contain 295, 678, and 2,036 blank fitness cells for BRCA2, IKBKAP, and SMN1, respectively. The authors’ PSI table has the same missing-cell pattern. A full build using the local CSV with NaNs failed with GraphFLA’s `ValueError` for non-finite fitness. I did not impute those entries or replace them with zero.

## Reproduction attempts

- **Full sequence-space `n_configs`:** the raw tables each enumerate the full design space, but GraphFLA cannot build those full graphs while the context PSI has missing values. Finite-only author-PSI builds gave 32,473 BRCA2, 32,090 IKBKAP, and 30,732 SMN1 graph nodes, so the printed full-space value 32,768 is a mismatch as a built-graph count. There is no paper-authorized imputation rule for the missing contexts.
- **PSI ≥20 `n_configs`:** I ran `tau=20, filter_mode="any", epsilon=0` on the author-reported PSI. BRCA2 selected 1,279 rows but GraphFLA removed 12 isolated nodes; SMN1 selected 2,892 but lost 16 isolated nodes. IKBKAP selected 250 and retained all 250, reproducing the printed count exactly. The filter is paper-stated; the isolate pruning is GraphFLA behavior.
- **Epistasis R² / `higher_order_epistasis`:** I ran order 1 and order 2 on (a) every finite PSI value and (b) the GU-only, PSI≥20 subset. I also ran order 1 after PSI≥20 filtering without the GU restriction for the matrix-model comparison. All variants are in `record.json`. The paper-filtered GU-only order-1 numbers round to the published matrix R² values, but GraphFLA uses ordinary least squares and the paper uses ridge regression and a residual-fit procedure for pairwise terms. GraphFLA order-2 values still differ from the paper’s pairwise R². I therefore record this as a mismatch, not a confirmed reproduction.
- **Cross-context active-set counts:** the paper also prints set intersections and asymmetric-threshold counts across contexts. These are not native single-landscape GraphFLA statistics, so I did not construct a derived overlap landscape.

## No numeric target in the paper

- Peak/local-optimum counts and their operational definition: **NOT STATED IN PAPER**. No GraphFLA `n_lo` comparison was made.
- `n_edges`: **NOT STATED IN PAPER**. No edge-count comparison was made.
- Epistasis class fractions: **NOT STATED IN PAPER**. Figure 4 displays interaction heat maps; no sign/magnitude/reciprocal-sign fractions were read from them.
- Skewness, kurtosis, CV, or other numeric PSI-distribution moments: **NOT STATED IN PAPER**. Figure 3A is histogram-only for these purposes; no values were read from plotted bars.

## Data limits

The supplementary PDF remains unavailable from the route tried, so supplemental-only numerical claims are not audited. The MPSA per-sequence author PSI output is available and was used for the main-text reproduction attempts. The package CSV provenance is now resolved for scale (`fitness = r`) but still contains context-specific missing measurements that prevent a full 32,768-node GraphFLA build without an unreported imputation.
