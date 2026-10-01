# Pokusaeva2019 acquisition and reproduction notes

## Bibliography and sources

Crossref record `10.1371/journal.pgen.1008079` verifies the publisher title and the author order. The assignment’s abbreviated list (Pokusaeva, Usmanova, Putintseva, … Kondrashov) is consistent with Crossref; no substantive bibliography error was found.

The dossier is `papers/_prominent_dms/`. It contains the publisher PDF, Crossref metadata, supplementary Figures S1–S9, Supporting Information S1 (orthologue alignment), S2 (library/statistical workbook), S3 (visualization), GEO metadata, and all twelve processed `GSE99990_Sxx_fitness.txt.gz` files. Every downloaded file has a URL, HTTP status, SHA-256, and saved path in `papers/_prominent_dms/acquisition_codex.jsonl` and the matching record’s `sources` array. The TIFF figures and GIF were acquired as part of the supplement, but no values were read from their plotted marks.

## Verbatim paper evidence

### Landscape scope, measured subset, and genotype counts

> “Across 11 experiments, we measured fitness for a total of 4,018,105 genotypes (875,151 unique amino acid sequences) with high accuracy. Of these, 422,717 consist solely of combinations of extant amino acid states from His3 orthologues, while the remaining genotypes incorporate other amino acid changes.”

Results → “Estimating fitness of evolutionary-relevant genotypes,” printed article p. 3.

> “For each segment we measured fitness for 60% - 99.8% of all possible genotypes from the combinatorial set of selected extant amino acid states found in 21 yeast species and a smaller fraction of combinations found across all domains of life.”

Results → “Estimating fitness of evolutionary-relevant genotypes,” printed article p. 4.

> “For segment 3 for instance, 11 out of 17 amino acid sites had more than one extant amino acid state: L145 = 2, L147 = 2, Q148 = 3, K151 = 2, V152 = 2, D154 = 3, L164 = 3, E165 = 4, A168 = 2, E169 = 4, A170 = 4, with the full yeast combinatorial set consisting of 2*2*3*2*2*3*3*4*2*4*4 = 55,296 genotypes out of which we determined the fitness for 48,198, or 87% of the possible yeast extant states combinations in our library.”

Results → “Estimating fitness of evolutionary-relevant genotypes,” printed article pp. 4–5.

> “Number of unique amino acid sequences with measured fitness”

Supplementary Information S2, “Table1 property of segments,” column heading. The verbatim cell strings in that column, rows 1–12 and “Total without 9,” are: “125915”; “122167”; “94031”; “69050”; “99891”; “83625”; “26364”; “73781”; “81497”; “79905”; “37167”; “63255”; “875151”.

> “Number of unique amino acid sequences containing only extant amino acids”

Supplementary Information S2, “Table1 property of segments,” column heading. The verbatim cell strings in that column, rows 1–12 and “Total without 9,” are: “58066”; “45657”; “48198”; “44255”; “51122”; “45280”; “4313”; “46307”; “29992”; “29837”; “16763”; “32919”; “422717”.

> “Number of unique nucleotide sequences with measured fitness”

Supplementary Information S2, “Table1 property of segments,” column heading. The verbatim cell strings in that column, rows 1–12 and “Total without 9,” are: “511950”; “403885”; “646447”; “460731”; “683161”; “253242”; “176879”; “198306”; “152398”; “259360”; “162789”; “261355”; “4018105”. The “Total” row cell is “4170503”.

> “For one segment, 9, the accuracy of our experiment was low, and it was not used in cumulative analyses.”

Results → “Estimating fitness of evolutionary-relevant genotypes,” printed article p. 3.

### Fitness scale and below-threshold/non-functional genotypes

> “We scaled fitness such that lethal genotypes have fitness 0 and neutral genotypes have fitness 1. We assumed that genotypes with a stop codon or frame shift are lethal. Thus, for each segment we linearly rescaled the fitness distribution so that 95% of genotypes with nonsense mutations have a fitness of 0 and so that the local maximum of the fitness distribution of genotypes with extant amino acids is 1.”

> “All fitness values that became smaller than 0 were set to 0.”

Materials and Methods → “Fitness rescaling,” printed article p. 21.

> “99.63% of nonsense genotypes have a fitness > 0.4 while 23.46% of extant amino acid combinations have fitness < 0.6.”

Supporting Information → S2 Fig caption, “Sequencing strategy and accuracy analysis of His3 segment libraries.” These are printed caption values; no values were read from the histogram.

### Sign epistasis and model fit

> “Just 15% of amino acids found in yeast His3 orthologues were always neutral while the impact on fitness of the remaining 85% depended on the genetic background.”

Abstract. Neutrality is deliberately outside the current literature-validation scope, so this number is noted without a GraphFLA comparison.

> “We found that 86 out of 128 (67%) sites in our library exhibit sign epistasis and 46% (59/128) exhibit reciprocal sign epistasis with 8% (968/11597) of all pairs of sites exhibiting sign epistasis.”

Results → “Ruggedness and multidimensional epistasis of the His3 fitness landscape,” printed article p. 10.

> “For each amino acid replacement … we considered only those that exhibit a large fitness effect (abs. difference > 0.4) … we identified secondary amino acid replacements that significantly alter the ratio of large increases to large decreases in fitness (Fisher’s exact test, Bonferroni corrected p-value < 0.05).”

> “We only consider a site to be under sign epistasis if there is a second site that alters the frequency of sign epistasis in a statistically significant manner, i.e. more frequently than expected by chance alone.”

Materials and Methods → “Quantifying sign epistasis,” printed article pp. 23–24.

> “Remarkably, 85% (330/389) of replacements between extant amino acid states had substantially different effects on fitness in different backgrounds.”

Results → “Estimating fitness of evolutionary-relevant genotypes,” printed article p. 6.

> “The ability of the cliff-like threshold fitness function to predict fitness from genotype varied between the His3 segments from near perfect (r² = 0.97) in segment 7, to relatively poor (r² = 0.44) in segment 5.”

Results → “Unidimensional epistasis of the His3 fitness landscape,” printed article p. 7.

> “The single neuron of the first layer computes the fitness potential, which is then mapped to a fitness value obtained from f(p), the function of the fitness potential which is found by the three layers of the neural network architecture.”

Results → “Unidimensional epistasis of the His3 fitness landscape,” printed article p. 7; Methods → “Predicting fitness using deep learning,” printed article pp. 22–23.

### Path access, peaks, and edge definitions

> “All genotypes one amino acid replacement apart are connected by an unweighted edge.”

Materials and Methods → “Paths between pairs of fit genotypes,” printed article p. 23.

> “Accessible paths are those that incorporate only fit genotypes.”

Figure 7 caption, panel d, “Analysis of evolutionary pathway accessibility.” The article describes the fraction over shortest paths between pairs of fit genotypes in prose and Figure 7, but gives no printed numeric fraction: **NOT STATED IN PAPER**. No value was read from the plot. This endpoint definition is also different from GraphFLA’s directed paths to a global optimum.

> “Remarkably, no fitness functions showed a defined optimum.”

Results → “Unidimensional epistasis of the His3 fitness landscape,” printed article p. 7. A numeric count of local optima/peaks is **NOT STATED IN PAPER**.

## GraphFLA comparisons and failures

- **`n_configs`:** the paper prints unique sequence counts, including per-segment counts and totals, but its analyzed genotype sets are incomplete combinatorial samples. The S03 processed table has variable-length `AAseq` strings. I attempted `SequenceLandscape().build_from_data(AAseq, s, epsilon=0, verbose=False)` on the archived S03 values; GraphFLA stopped before graph construction with `ValueError: All sequences must have the same length (expected 25, got 27 for sequence 3)`. I did not pad, align, or otherwise alter genotypes to force a build. The overlap is recorded as `input_unavailable`.
- **`classify_epistasis`:** paper denominators are sites and site pairs classified by effects across backgrounds with a >0.4 threshold and a Fisher test; GraphFLA fractions are among four-node graph motifs. This is definition-incompatible, so no GraphFLA motif run was made.
- **`higher_order_epistasis`:** the printed R² values are fits from a deep neural network implementing a nonlinear sigmoid/threshold function of one or more fitness potentials. GraphFLA’s metric is polynomial regression by interaction order. The estimators and target definitions differ; no run was made.
- **`evolvability_enhancing_mutations`:** the 85% background-dependence result is a candidate comparison only. Per the suite instruction, `definition_match` is `unknown` because the GraphFLA metric is under review; no run was made.
- **`fitness_distribution`:** the paper prints thresholded class proportions in the S2 Fig caption, but GraphFLA’s named distribution metric returns shape statistics, not these threshold fractions. No GraphFLA value was compared.
- **Other candidates:** a local-optimum count and a numeric edge total are **NOT STATED IN PAPER**. The paper’s peak language is qualitative; Figure 2 describes edges as one-replacement neighbors but does not print an edge count. No plot-only values were read.
- **Scalar ruggedness:** A value for GraphFLA `r_s_ratio` is **NOT STATED IN PAPER**; the printed sign-epistasis proportions above are the paper’s site-level proxy, not an `r_s_ratio` statistic.

## Accession-metadata discrepancy

The saved GEO series summary says: “Furthermore, in 63% of sites substitutions were strongly positive in one genetic background and strongly negative in another, with 41% of sites showing reciprocal sign epistasis.” (GEO accession GSE99990, `Series_summary`.) The publisher article instead prints “67%” and “46%” in the Results passage quoted above. I retain the article’s published values as paper targets and flag the GEO summary as a conflicting archive description, not as a replacement result.
