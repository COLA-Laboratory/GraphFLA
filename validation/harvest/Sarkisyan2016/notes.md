# Sarkisyan2016 acquisition and reproduction notes

## Bibliography and sources

Crossref record `10.1038/nature17995` verifies the publisher title and full author list. The assignment’s abbreviated author string (Sarkisyan, Bolotin, Meer, … Kondrashov) follows the Crossref author order; no substantive bibliography error was found.

The dossier is `papers/_prominent_dms/`. It contains the full-text XML, publisher supplementary-information PDF, Crossref and Figshare metadata, the amino-acid and nucleotide genotype/brightness tables, and the reference FASTA. Each saved source’s URL, HTTP status, SHA-256, and path is listed in `papers/_prominent_dms/acquisition_codex.jsonl` and in the matching record’s `sources` array.

The PMC PDF endpoint returned HTTP 200 but saved a small HTML challenge page rather than a PDF. That failed response is preserved as `Sarkisyan2016_main_codex.pdf`; the complete Europe PMC XML was acquired and used for the article text. The supplementary PDF is the publisher file. Figshare’s data description identifies the processed amino-acid and nucleotide tables as genotype/brightness tables. Its mutation-string convention is not stated in the paper: **NOT STATED IN PAPER**. The archive description says: “Briefly, all positions are zero-based (i.e. first nucleotide has index 0) and type of mutation (substitution, deletion or insertion) is indicated as the first letter of mutation description.” (Figshare item 3102154, description → mutation notation.)

## Verbatim paper evidence

> “We show that the fitness landscape of avGFP is narrow, with 3/4 of the derivatives with a single mutation showing reduced fluorescence and half of the derivatives with four mutations being completely non-fluorescent.”

Abstract. These are mutation-count-stratified phenotype proportions; GraphFLA does not expose these as named whole-landscape metrics, so they are noted but not treated as direct reproduction targets.

### Dataset size, sampling, and paper-stated filtering

> “Our final dataset included 56,086 unique nucleotide sequences coding for 51,715 different protein sequences.”

Main article, Results, paragraph beginning “We applied several strategies to minimize the error of our estimate of fluorescence.”

> “We excluded genotypes containing insertions, deletions or stop codons, and grouped barcodes with the same nucleotide or amino acid genotypes. The final dataset consisted of 56,086 unique nucleotide genotypes and 51,715 unique amino acid genotypes.”

Supplementary Information, S3.5 “Building of the final dataset,” p. 8.

> “Still, since the total number of possible sequences grows exponentially with the number of mutations, the fraction of sampled sequences was tiny for sequences containing more than two mutations.”

Main article, Results, paragraph beginning “Our procedure introduced an average of 3.7 mutations per gene sequence.” This is the paper’s stated sampling caveat; the measured set is a local, sparse neighborhood rather than a complete combinatorial protein landscape.

### Non-fluorescent variants and epistasis definition

> “Sequences with log-fluorescence < 3.0 have light intensity less than wild-type by 10^{3.72-3.0} ≈ 5 times (Figure 2a). 9.4% of genotypes had such low intensity, which we considered non-fluorescent.”

Supplementary Information, S4.1 “Single mutant analysis,” p. 8.

> “Thus, we defined epistasis e as the deviation from additivity of effects of single mutations on the logarithmic scale.”

> “We compared the decrement of the log-fluorescence of a multiple mutant F_{mult} to the sum of decrements of individual mutants, such that e = (F_{mult} − F_{wt}) − Σ_{i}(F_{i} − F_{wt}), where F_{wt} and F_{i} are the fluorescence log-values conferred by avGFP and avGFP with the i-th single missense mutation, respectively.”

> “We defined strong epistasis as |e| > 0.7, or as cases where the observed fluorescence differed from the expected by at least fivefold, with a false discovery rate of < 1%.”

Main article, Results, paragraph beginning “Interaction of deleterious mutations can manifest in positive epistasis.”

> “Negative epistasis affected up to 30% of all genotypes, depending on the number of mutations.”

Main article, same Results paragraph. This is a genotype-level proportion under the paper’s strong-epistasis procedure, not a GraphFLA four-node epistasis-class fraction.

### Variance explained

> “We used multiple regression considering a non-epistatic fitness function whereby log-fluorescence, F, is equal to the linear predictor, the fitness potential, p, such that F=f(p)=p.”

> “This simplest, non-epistatic model explained only 70% of the initial sample variance (σ²=1.12 and σ²=0.34 before and after the application of the model, respectively).”

Main article, Results, paragraph beginning “In a unidimensional landscape fitness is a monotonic function of an intermediate variable.”

> “We, therefore, modeled F as a sigmoid function of p, which explained 85% of the initial sample variance (σ² = 0.17).”

> “A more complex sigmoid-shaped fitness function refined with a neural network approach … explained 93.5% of the initial sample variance (σ² = 0.065).”

Main article, same Results paragraph.

> “The threshold fitness function does a remarkably good job in approximating the entire fitness landscape explaining ~95% of all variance.”

Main article, Results, paragraph beginning “The threshold fitness function does a remarkably good job.”

### Path accessibility

> “For each graph we calculated how many paths of length four, the shortest possible paths, exist between the two genotypes and how many of them were accessible for evolution.”

> “Following Maynard Smith … we considered a path as accessible if all of the three intermediate genotypes were neutral or advantageous, that is conferred fluorescence at least as high as points at the ends of the path.”

Supplementary Information, S4.7 “Path accessibility on the fitness landscapes,” pp. 13–14.

> “The lower bound fraction of inaccessible paths was 0/596,228,074 (0%), 1,201,621/70,670,872 (1.7%), 13,324/202,308 (6.6%) and 6/174 (3.4%) for paths 1.2 (blue) 1.6 (orange), 2.0 (red), and 2.4 (purple), substitutions away from the wildtype, respectively (colour-coded with Supplementary Fig. 3).”

Supplementary Information, S4.7, p. 14. These are printed text values, not estimates read from a plot. The paper’s endpoints are pairs of fluorescent double mutants; GraphFLA’s `global_optima_accessibility` is a different endpoint/path statistic.

## GraphFLA comparisons and failures

- **Protein `n_configs`:** I removed rows whose amino-acid mutation string contains a stop (`*`), as required by S3.5, reconstructed amino-acid sequences against the archived reference, and used the archive’s `medianBrightness` column unchanged. The resulting input count agrees with the paper’s quoted count. `ProteinLandscape().build_from_data(..., epsilon=0, verbose=False)` retained 17,604 configurations and 35,092 directed improving edges. GraphFLA reported that 34,111 isolated configurations had no one-amino-acid mutational neighbors and removed them. This is a mismatch against the paper’s measured-genotype count, not a peak-count reproduction.
- **Order-1 `higher_order_epistasis`:** GraphFLA returned `0.8700741822810955`, versus the printed 70%. The paper and GraphFLA both fit a linear additive model in broad terms, but GraphFLA runs on the post-pruning graph nodes; the paper’s statistic uses the paper’s full sample. I record the numeric difference as a mismatch and do not treat the rounded result as equivalent.
- **Nonlinear model R² values:** The printed 85%, 93.5%, and approximately 95% values are sigmoid/threshold or neural-network model fits. They are not GraphFLA’s polynomial order-k regression, so they are definition-incompatible; no run was made.
- **Epistasis:** The paper’s “up to 30%” denominator is genotypes with multiple mutations and uses its clipped additive expectation and `|e|`/fivefold/FDR rule. GraphFLA `classify_epistasis` returns fractions of four-node motif classes. The definitions do not match; no run was made.
- **Path accessibility:** The paper reports lower-bound inaccessible shortest-path fractions between two fluorescent genotypes, grouped by mean distance from wild type. GraphFLA’s global-optimum accessibility is not the same statistic. No GraphFLA path metric was run.
- **Nucleotide `n_configs`:** The paper’s quoted nucleotide count is a candidate target, but full nucleotide sequences could not be rebuilt without unsupported correction. Applying the archive-described zero-based substitutions to the archived reference produced reference-allele conflicts for 915 filtered rows, all at position 191 (`SA191…` states reference A while the archived reference has G). No alternate coordinate, allele, or reference correction was applied; the `DNALandscape` run was not attempted. This is recorded as `input_unavailable`.
- **Peak count and GraphFLA distribution moments:** The paper gives no numeric local-optimum count: **NOT STATED IN PAPER**. Its fluorescence cutoff fraction (9.4%) is printed, but `fitness_distribution` does not return a thresholded fraction; no distribution-moment target is printed: **NOT STATED IN PAPER**. No plotted values were read.
- **Edges and scalar ruggedness:** A printed `n_edges` or `r_s_ratio` target is **NOT STATED IN PAPER**.
