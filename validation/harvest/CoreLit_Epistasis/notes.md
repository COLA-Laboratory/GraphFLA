# Core literature: epistasis decomposition and metric provenance

Study key: `CoreLit_Epistasis`  
Dossier: `papers/_core_lit_epistasis`  
Comparison paper: Huang, Zhou & Li (2025), DOI `10.52202/085713-1180`.

This is raw validation material, not a promoted benchmark. Numeric targets below come only from printed tables; no value was read from a plotted point. The invalid PDF responses, corrections, downloaded-source hashes, and paths are in [acquisition_codex.jsonl](../../papers/_core_lit_epistasis/acquisition_codex.jsonl).

## 1. Bibliography verification

Metadata was checked by title and DOI against publisher pages, PubMed, and the institutional record below. All named starting-point author lists checked out; I found no incorrect author list within the references in this assignment.

| Work | Verified citation | Metadata source |
|---|---|---|
| Sign epistasis | Daniel M. Weinreich, Richard A. Watson & Lin Chao (2005), “Perspective: Sign epistasis and genetic constraint on evolutionary trajectories,” *Evolution* 59(6):1165–1174, DOI [10.1111/j.0014-3820.2005.tb01768.x](https://doi.org/10.1111/j.0014-3820.2005.tb01768.x). | [PubMed](https://pubmed.ncbi.nlm.nih.gov/16050094/) |
| Empirical paths | Frank J. Poelwijk, Daniel J. Kiviet, Daniel M. Weinreich & Sander J. Tans (2007), “Empirical fitness landscapes reveal accessible evolutionary paths,” *Nature* 445:383–386, DOI [10.1038/nature05451](https://doi.org/10.1038/nature05451). | [Nature](https://www.nature.com/articles/nature05451), [PubMed](https://pubmed.ncbi.nlm.nih.gov/17251971/) |
| Higher-order epistasis perspective | Daniel M. Weinreich, Yinghong Lan, C. Scott Wylie & Robert B. Heckendorn (2013), “Should evolutionary geneticists worry about higher-order epistasis?”, *Current Opinion in Genetics & Development* 23:700–707, DOI [10.1016/j.gde.2013.10.007](https://doi.org/10.1016/j.gde.2013.10.007). | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0959437X13001535), [PubMed](https://pubmed.ncbi.nlm.nih.gov/24290990/) |
| Linkage of formalisms | Frank J. Poelwijk, Vinod Krishna & Rama Ranganathan (2016), “The Context-Dependence of Mutations: A Linkage of Formalisms,” *PLoS Computational Biology* 12(6):e1004771, DOI [10.1371/journal.pcbi.1004771](https://doi.org/10.1371/journal.pcbi.1004771). | [PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004771), [PubMed](https://pubmed.ncbi.nlm.nih.gov/27337695/) |
| Nonlinear maps | Zachary R. Sailer & Michael J. Harms (2017), “Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps,” *Genetics* 205(3):1079–1088, DOI [10.1534/genetics.116.195214](https://doi.org/10.1534/genetics.116.195214). | [Genetics/OUP](https://academic.oup.com/genetics/article/205/3/1079/6066378), [PubMed](https://pubmed.ncbi.nlm.nih.gov/28100592/) |
| High-order trajectory effects | Zachary R. Sailer & Michael J. Harms (2017), “High-order epistasis shapes evolutionary trajectories,” *PLoS Computational Biology* 13(5):e1005541, DOI [10.1371/journal.pcbi.1005541](https://doi.org/10.1371/journal.pcbi.1005541). | [PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005541), [PubMed](https://pubmed.ncbi.nlm.nih.gov/28505183/) |
| Landscape comparison | Ivan G. Szendro, Martijn F. Schenk, Jasper Franke, Joachim Krug & J. Arjan G. M. de Visser (2013), “Quantitative analyses of empirical fitness landscapes,” *Journal of Statistical Mechanics: Theory and Experiment* 2013(1):P01005, DOI [10.1088/1742-5468/2013/01/P01005](https://doi.org/10.1088/1742-5468/2013/01/P01005). | [Wageningen University record](https://research.wur.nl/en/publications/quantitative-analyses-of-empirical-fitness-landscapes/) |
| Landscape review | J. Arjan G. M. de Visser & Joachim Krug (2014), “Empirical fitness landscapes and the predictability of evolution,” *Nature Reviews Genetics* 15(7):480–490, DOI [10.1038/nrg3744](https://doi.org/10.1038/nrg3744). | [Nature Reviews Genetics](https://www.nature.com/articles/nrg3744), [PubMed](https://pubmed.ncbi.nlm.nih.gov/24913663/) |
| GraphFLA feature paper | Mingyu Huang, Shasha Zhou & Ke Li (2025), “Augmenting Biological Fitness Prediction Benchmarks with Landscapes Features from GraphFLA,” *Advances in Neural Information Processing Systems 38*, Datasets and Benchmarks Track, DOI [10.52202/085713-1180](https://doi.org/10.52202/085713-1180). | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/323fcf83a0f26c7c04eae0d91125d518-Abstract-Datasets_and_Benchmarks_Track.html) |

The 2007 Nature PDF URL returned HTML despite HTTP 200. An author-hosted URL returned 404, and attempted USPTO copies returned 403 challenge responses. The Nature/PubMed metadata is verified; a usable article PDF was not obtained. The source folder retains the failed HTML responses and the acquisition log records them. The definition excerpt below was checked against the surfaced article copy and its publisher metadata. The 2017 PMC PDF endpoint also returned HTML, but Europe PMC full-text XML was obtained and used.

## 2. Definitions, normalization, coding, and qualifying landscape sizes

### 2.1 Weinreich, Watson & Chao (2005)

Verbatim: “Sign epistasis at a locus means that a mutation there is beneficial on some genotypic backgrounds and deleterious on others” (Figure 2 caption, printed p.1169).

- Defining algebraic equation: `NOT STATED IN PAPER`.
- Quantitative normalization: `NOT STATED IN PAPER`.
- Numeric genotype coding (0/1 or −1/+1): `NOT STATED IN PAPER`.
- A printed, reproducible target on a landscape with >1,024 variants: `NOT STATED IN PAPER`; the paper is a conceptual analysis of small mutation landscapes, and contains no such table target.

### 2.2 Poelwijk, Kiviet, Weinreich & Tans (2007)

Verbatim: “For magnitude epistasis the fitness effect differs in magnitude, but not in sign. For sign epistasis, the sign of the fitness effect changes.” (Box 1, printed p.383.)

- Defining algebraic equation: `NOT STATED IN PAPER`.
- Quantitative normalization: `NOT STATED IN PAPER`.
- Numeric genotype coding (0/1 or −1/+1): `NOT STATED IN PAPER` (the paper uses allele labels such as `ab`, `Ab`, `aB`, and `AB`).
- A printed, reproducible target on a landscape with >1,024 variants: `NOT STATED IN PAPER`; no such landscape table or source-data target was found. No plot-only values were used.

### 2.3 Weinreich, Lan, Wylie & Heckendorn (2013)

Verbatim transform and normalization: `E = 2^(−L) C W`, where the paper calls `C` a symmetric Hadamard transform matrix (Box 1, printed p.701). Verbatim coding: “digits 1 and 0, respectively, signal the presence or absence of the mutation” (Box 1, printed p.701). To report classical kth-order interaction coefficients, the paper says it computes “the kth order epistatic coefficient as 2^k times the corresponding Walsh coefficient” (Box 1, printed p.702); thus ε_k = 2^k E_k. The (2^{-L}) normalization is on the Walsh transform; the additional (2^k) converts a kth-order Walsh coefficient to the classical epistatic coefficient.

Table 1 includes a nine-mutation system whose largest complete subset is six sites, and the largest complete subset reported in the table is seven sites (“7” mutations; “7” in the largest-complete-subset column; printed p.703). A complete seven-site binary subset has 128 genotypes. Thus this paper has no >1,024-variant empirical target. Its squared coefficient-by-order figure is plotted, so plot values were not used as reproduction targets.

### 2.4 Poelwijk, Krishna & Ranganathan (2016)

The main text defines the background-averaged operator as `ε = V H y` (Eq. 8) and later writes `O_epi = V H` for the full-space case (Eq. 16). Its diagonal order weights are `v_ii = (-1)^q_i / 2^(n-q_i)`, where `q_i` is the interaction order (text beside Eq. 8). This is an order-dependent normalization, not the unweighted Walsh coefficient. For the local biochemical formalism the paper gives `λ = G y` (Eq. 4).

Verbatim genotype coding: “0” and “1” represent “the wild-type and mutant state of each position” (Basic definitions, p.2); its Fourier/regression representation also uses `σ_i ∈ {−1,+1}` (Eq. 11 and S4 Text). No >1,024-variant printed target with obtainable data appears in its examples or tables (`NOT STATED IN PAPER`). The supplementary text explicitly compares the Fourier and Taylor codings.

### 2.5 Sailer & Harms (2017a,b)

Verbatim model equations: `P = Xβ` and `β = X^-1 P` (Eqs. 3–4). The paper describes `X` as the Hadamard design matrix. The paper’s exact stated normalization is the inverse matrix `X^-1`; a separate scalar `1/2^L` normalization is `NOT STATED IN PAPER` as a standalone equation. Inferring `X^-1 = X^T/2^L` for the complete orthogonal Hadamard design is an algebraic consequence, not a separately printed normalization.

Verbatim coding in the Genetics paper: “−1 (wild type) or +1 (mutant)” (Methods, “High-order epistasis model”). Its high-order contribution uses a “Pearson coefficient” change as interaction orders are added (Results, “High-order epistasis is a common feature of genotype-phenotype maps”). A standalone equation and exact normalization for that statistic are `NOT STATED IN PAPER`; this increment in Pearson fit differs from GraphFLA’s cumulative order-\(k\) \(R^2\). The abstract reports a “2.2 to 31.0%” high-order contribution range. One mixed DNA/protein map has “128 genotypes” (Table 1 and Methods); neither paper has a >1,024-variant target.

The PLOS paper gives the order-specific contribution explicitly: \(\phi=\rho_i^2-\rho_{i-1}^2\) (Eq. 7, Materials and methods, printed p.11). Here \(\rho_x^2\) is the squared Pearson correlation between the original linearized fitness values and the model truncated at order \(x\) (Eq. 7 and accompanying definition). The normalization is a difference between successive squared Pearson correlations; there is no extra division by total variance. Its Walsh decomposition uses \(\tilde{\beta}=X^{-1}\tilde{F}_{linear}\) (Eq. 4) and reconstructs truncated fitness with \(\tilde{F}_{linear,trunc}=X\tilde{\beta}_{trunc}\) (Eq. 5). Global coding is “−1 (wildtype) or +1 (mutant)” and the local reference coding is “0 (wildtype) or 1 (mutant)” (Materials and methods, printed p.11). The paper says each map contains “all possible combinations of 5 mutations (2^5 = 32 genotypes)” (Results, printed p.2; Table 1). Figure 1B plots per-order values; the paper does not provide a printed per-order table on a >1,024-variant map, so those plotted values were not treated as targets.

### 2.6 Szendro et al. (2013)

Verbatim coding: `σ̃_j = 2σ_j − 1 = ±1`, with `σ_j` binary (Eq. 6, printed p.12). Their exact-order energy and normalization are `F_n = β_n / ∑_(j=1)^L β_j`, with `β_n = ∑_(j=1)^(L choose n) (b_j^(n))^2` (Eq. 7); `∑_(n=1)^L F_n = 1`. Their total epistasis measure is `F_sum = ∑_(j=2)^L F_j` (Eq. 8, printed p.13).

This is an exact-order variance-energy spectrum, not GraphFLA’s cumulative polynomial-regression (R^2) through order (k). The largest Table 1 binary landscapes have nine loci / at most 512 possible variants; one has 418/512 observed (Table 1, printed p.11). No >1,024-variant target appears.

Code-side caution for `walsh_hadamard`: GraphFLA encodes Boolean inputs as 0/1 strings (`graphfla/analysis/epistasis/walsh_hadamard.py`, lines 87–90), constructs an inverse H transform and per-term V weighting (lines 295–308 and 369–412), and solves the ensemble design for coefficients. This implementation detail is not a paper definition. I did not equate its returned coefficients to any paper’s unweighted Walsh vector or to (2^kE_k) without a qualifying printed target.

### 2.7 de Visser & Krug (2014) review and Huang et al. (2025)

The de Visser–Krug review’s Box 1 gives `f(σ) = a^(0) + ∑_i a_i^(1)σ_i + ∑_(i<j) a_ij^(2)σ_iσ_j + ⋯`, with `σ_i ∈ {−1,+1}` (Eq. 1, printed p.481). It defines order weights as `b^(n) = ∑ (a^(n))^2` (Eq. 2) and obtains a single epistasis fraction by summing weights for `n > 1` and dividing by all order weights (Box 1, printed p.481). The review describes empirical analyses “to a maximum of nine mutations” (printed p.482); a complete binary space of nine sites has 512 genotypes, so there is no >1,024-variant target.

For the modern printed targets:

- NFC is “NFC = Cor[f(g), ⟨f⟩_N(g)]”, and Eq. A3 gives `⟨f⟩_N(g) = 1/|N(g)| Σ_(g′∈N(g)) f(g′)` (Appendix C.1.3, printed p.28). The paper calls it a “Pearson correlation coefficient.” This is the same neighborhood-mean statistic used in the reproduction calls.
- Verbatim square-class equation (Appendix C.3.1, Eq. A24, printed pp.34–35):

  \[
  e(g,i,j)=\begin{cases}
  \mathrm{None} & \text{if }\epsilon_{ij}=0\\
  \mathrm{Magnitude} & \text{if }\epsilon_{ij}\ne0\text{ and }[s_i(g)\,s_i(g[j])\ge0\text{ and }s_j(g)\,s_j(g[i])\ge0]\\
  \mathrm{Reciprocal\ Sign} & \text{if }s_i(g)\,s_i(g[j])<0\text{ and }s_j(g)\,s_j(g[i])<0\\
  \mathrm{Sign} & \text{otherwise (i.e. if }\epsilon_{ij}\ne0\text{ and exactly one sign product is negative)}
  \end{cases}
  \]

  The paper further states `ε_ij = s_i(g[j]) − s_i(g) = 0` defines the non-epistatic case (same equation). Its accompanying text states that prevalence is enumerated over pairs of single mutations from reference genotypes and maps these cases to directed four-node motifs. Exact GraphFLA runs on the Kehe and Butyrate inputs reproduce all five printed class fractions to three decimals.
- For DRI, the verbatim definition fragments are “Pearson correlation coefficient,” “fitness of the background genotype, f(g),” and “positive selection coefficient” (Appendix C.3.2, printed p.36). For ICI the paper uses “absolute value of the corresponding negative selection coefficient” (same locator). The paper applies these to beneficial and deleterious single-step transitions, respectively. They are edge-pool definitions.
- Higher-order ε^(2) is a polynomial linear-regression (R^2) with interactions through order two (Appendix C.3.4, printed p.37; the paper calls it an “R² score”). This matches the nominal order-2 statistic in `higher_order_epistasis`.
- Table A2 prints the Kehe [50] and Butyrate [111] targets (printed p.47); Table A3 prints Kuo2020 (printed p.48); Table A4 prints ProteinGym tasks including RCD1 (printed p.52). Each cited value in `record.json` is from these printed tables.

## 3. Reproduction results and honest divergences

Inputs were used as stored, with no score transformation or manual filtering. The API calls used default maximization and ε=0 where a Boolean landscape was built. For Kuo2020, `DNALandscape` first failed with `ValueError` because sequences contain `U`, which is outside the DNA alphabet `[A,C,G,T]`; the input symbols justify the documented change to `RNALandscape` `[A,C,G,U]`. No other preprocessing variant was tuned.

| Printed target and locator | Input and GraphFLA result | Outcome |
|---|---|---|
| NFC “.790”; classes “.404/.395/.201/.841/.159”; DRI “.043”; ICI “.816”; ε^(2) “.547” — Huang 2025 Table A2, p.47, [50] / Kehe | `Skwara2023_Kehe_data.csv`: 21,198 rows, 4,239 unique sequence strings, 4,125 GraphFLA configurations and 15,454 improving edges. NFC `.7902639`; exact class values `.4041645/.3946480/.2011875/.8406605/.1593395`; API DRI `.0426770`; API ICI `.8164257`; API order-2 (R^2=.7338619). | NFC and all five class fractions reproduced to printed precision. DRI/ICI match numerically only; definitions differ (see below). Higher-order (R^2) mismatch. A raw-row order-2 OLS check gave `.7272276`, also not `.547`. |
| NFC “.635”; classes “.457/.354/.189/.774/.226”; DRI “−.319”; ICI “.583”; ε^(2) “.675” — Huang 2025 Table A2, p.47, [111] / Butyrate | `Skwara2023_Butyrate.csv`: 1,561 rows, 577 unique sequence strings, 336 GraphFLA configurations after 241 isolated configurations were removed, 505 improving edges. NFC `.7096302`; exact class values `.4573171/.3536585/.1890244/.7743902/.2256098`; API DRI `−.3191680`; API ICI `.5834917`; order-2 (R^2=.9362648). | All five class fractions and DRI/ICI rounded values match; NFC and order-2 (R^2) mismatch. This is not a >1,024-unique-variant case: its 1,561 table records reduce to 577 unique sequences and 336 connected GraphFLA configurations. Raw-row OLS (R^2=.8709447), also not `.675`. |
| NFC “.917”; ε_reci “.206”; DRI “.391” — Huang 2025 Table A3, p.48, Kuo2020 | `Kuo2020.csv`: 197,890 unique RNA sequences/configurations and 2,180,702 improving edges. NFC `.9175257` (rounds to `.918`); auto-sampled reciprocal-sign estimate `.2067209` (sample cut probability `.5`, seed 0; rounds to `.207`); API DRI `.3914520`. | NFC and the sampled reciprocal-sign estimate miss printed precision. DRI matches numerically only: the paper’s edgewise DRI formula applied to these transitions gives `.2383588`, not `.3914520`. Exact four-motif enumeration was not attempted at this graph size. |
| NFC “.696”; ε_reci “.182” — Huang 2025 Table A4, p.52, RCD1 | Downloaded ProteinGym v1.3 assay input has 1,261 rows; all retained. GraphFLA made 1,261 configurations and 11,776 improving edges. NFC `.6959898`; exact reciprocal-sign proportion `.1820834`. | Both reproduced to printed precision. |

The paper’s Table A2 prints “21198” for reference [50] and “1561” for [111] (printed p.47); Table A3 prints Kuo2020 size `4^9 = 262,144` (p.48); Table A4 prints RCD1 size “1261” (p.52). Its duplicate-record and genotype-weighting procedure is `NOT STATED IN PAPER`. GraphFLA’s builds collapse duplicate configurations and remove isolates. This prevents treating all API/Table A2 differences as an implementation-only issue.

The numerical DRI/ICI matches are not method reproductions. Huang et al.’s p.36 formulas pool individual mutation transitions. By contrast, GraphFLA’s current code documents DRI as correlating node fitness with the mean improvement over outgoing edges (`graphfla/analysis/epistasis/idiosyncrasy.py`, lines 263–270), and ICI as correlating node fitness with the mean cost over incoming edges (same file, lines 400–408). Applying the paper’s stated edge-pool formulas to the same improving-edge data produced:

- Kehe: DRI `.0253673`, ICI `.6697017`.
- Butyrate: DRI `−.2918116`, ICI `.5006600`.
- Kuo2020: DRI `.2383588`.

These values differ from the table. The API values remain recorded separately in `record.json`; the three-decimal numerical matches are not evidence that the definitions agree.

## 4. Job B: provenance of four candidate scalar metrics

### `gradient_intensity`

**NO published origin found; searched:** the exact phrase “gradient intensity” with fitness landscape; mean absolute fitness difference over mutation edges; normalized fitness-gradient magnitude; the core empirical landscape papers/reviews above; and the full Huang et al. 2025 paper. No matching published definition was located. The current package describes the code quantity as “the average absolute fitness difference (`delta_fit`) across all edges” (`graphfla/analysis/ruggedness.py`, lines 113–117), then divides by mean fitness (lines 135–145). That is code provenance, not a verified published origin.

### `fitness_flattening_index`

**NO published origin found; searched:** the exact phrase “fitness flattening index”; flattening index + fitness landscape; adaptive-path fitness-difference slope/correlation; the landscape reviews above; and the full Huang et al. 2025 paper. Papers do discuss qualitative flattening or smoothing of landscapes, but I found no paper defining this exact scalar. Current GraphFLA describes FFI as assessing whether the landscape is flatter around the global optimum “by evaluating adaptive paths” (`graphfla/analysis/correlation.py`, lines 150–169); its implementation averages correlations between path step and successive fitness differences (lines 172–202). This implementation description does not establish a literature origin.

### `neighbor_fitness_correlation`

**Originates in Huang, Zhou & Li (2025), Appendix C.1.3, Eqs. A2–A3, printed p.28.** The exact printed formula is “NFC = Cor[f(g), ⟨f⟩_N(g)]”, and the neighborhood mean is the arithmetic mean of all one-mutant neighbors. It is Pearson correlation across genotypes. This agrees with `A.neighbor_fitness_correlation(ls, method="pearson")` on landscapes with a complete comparable neighbor set. Table A2 Butyrate and Table A3 Kuo values do not reproduce to printed precision; those discrepancies are retained.

### `diminishing_returns_index` / `increasing_costs_index`

**A scalar published origin exists in Huang, Zhou & Li (2025), Appendix C.3.2, printed p.36.** DRI is Pearson correlation over all beneficial single-step transitions, pairing background fitness with positive effect. ICI is Pearson correlation over all deleterious transitions, pairing background fitness with absolute cost. This is an edge-pool statistic. Lyons et al. 2020 is treated here only as the already-established per-mutation distribution analysis specified in the task; the scalar search instead found Huang et al.’s across-transition Pearson definitions.

The divergence is in the current GraphFLA implementation: `diminishing_returns_index` averages improving out-edge effects by source node before correlating; `increasing_costs_index` averages incoming costs by target node before correlating. These node aggregates do not equal the published edge-pool correlations. The numeric matches in Table A2 and Table A3 are therefore flagged `definition_match: "no"` in `record.json`.

Searches for orphan origins used title/phrase search and the exact source terms above, plus direct reading of the listed review papers and the NeurIPS paper text. Search-result hits for qualitative landscape smoothing or generic gradient language were not assigned as origins because they do not define the exact GraphFLA quantity.

## 5. Acquisition and execution notes

- The source folder contains valid PDFs/XML and the invalid HTTP-200 HTML responses; every fetched file and failed request is logged in `papers/_core_lit_epistasis/acquisition_codex.jsonl` with status, hash when saved, and path.
- ProteinGym archive SHA-256: `3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921`. Extracted RCD1 CSV SHA-256: `d06d09413321d0c45661dcbd7a22fb7e8e28226fdeffb8fd60e2ecf82adf5286`.
- GraphFLA was run from the validation workspace with bytecode writes suppressed and the repository only on `PYTHONPATH`. No file under `/Users/arwen/Documents/GitHub/GraphFLA` was modified.
- No older-paper plotted curve values were used. The modern targets listed above are table values. `NOT STATED IN PAPER` is used where the source does not report the requested coding, normalization, or qualifying target.
