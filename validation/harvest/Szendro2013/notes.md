# Szendro et al. (2013): acquisition and reproduction notes

## Bibliography and acquisition

The focal citation was verified by DOI against Crossref, not by the supplied author names. Crossref lists the title Quantitative analyses of empirical fitness landscapes, *Journal of Statistical Mechanics: Theory and Experiment* 2013(1), P01005, DOI `10.1088/1742-5468/2013/01/P01005`, by Ivan G. Szendro, Martijn F. Schenk, Jasper Franke, Joachim Krug, and J. Arjan G. M. de Visser. The supplied focal title and author order match the Crossref record. The metadata is saved at [Szendro2013_crossref_codex.json](../../papers/Szendro2013/sources/Szendro2013_crossref_codex.json).

The arXiv v2 PDF and source archive were acquired. The archive contains the article TeX and five figure files, and no separate supplement. The article PDF is saved at [Szendro2013_arxiv_v2_codex.pdf](../../papers/Szendro2013/sources/Szendro2013_arxiv_v2_codex.pdf). The PLOS Table S1 for the A. niger source landscape was acquired at [Franke2011_A_niger_TableS1_codex.pdf](../../papers/Szendro2013/sources/Franke2011_A_niger_TableS1_codex.pdf).

For O’Maille et al. (2008), the PMC record identifies Supplementary Tables 1–7, including the M9 genotype/product data, in `NIHMS67307-supplement-1.pdf`. The direct PMC download returned HTTP 200 but file inspection showed an NCBI proof-of-work page titled “Preparing to download ...”, not a PDF. The companion DOC response was HTML as well. Both failed responses are preserved for audit; acquisition attempts and SHA-256 values are in [acquisition_codex.jsonl](../../papers/Szendro2013/acquisition_codex.jsonl).

## Table 1: all landscapes and source publications

The Table 1 caption (printed p.9) describes the locus count, available genotype count, fitness proxy, and mutation information. Each quoted size cell below is from Table 1, printed p.9; source publication titles were checked separately against publisher, PubMed, Crossref, institutional, or Dryad records.

| ID | System | L | Available / possible | Fitness proxy | Mutation direction / known effect | Ref. | Original source publication |
|---|---|---:|---:|---|---|---|---|
| A | *Methylobacterium extorquens* | 4 | 16/16 | Growth rate | Beneficial / combined | [26] | Chou et al. (2011), “Diminishing returns epistasis among beneficial mutations decelerates adaptation,” *Science* 332:1190–1192, [DOI](https://doi.org/10.1126/science.1203799). |
| B | *Escherichia coli* | 5 | 32/32 | Fitness | Beneficial / combined | [56] | Khan et al. (2011), “Negative epistasis between beneficial mutations in an evolving bacterial population,” *Science* 332:1193–1196, [DOI](https://doi.org/10.1126/science.1203801). |
| C–D | Dihydrofolate reductase | 4 | 16/16 | Resistance / growth rate | Beneficial / individual and combined | [27] | Lozovsky et al. (2009), “Stepwise acquisition of pyrimethamine resistance in the malaria parasite,” *PNAS* 106:12025–12030, [DOI](https://doi.org/10.1073/pnas.0905922106). |
| E | β-lactamase | 5 | 32/32 | Resistance | Beneficial / combined | [57] | Weinreich et al. (2006), “Darwinian evolution can follow only very few mutational paths to fitter proteins,” *Science* 312:111–114, [DOI](https://doi.org/10.1126/science.1123539). |
| F | β-lactamase | 5 | 32/32 | Resistance | Beneficial / combined | [58] | Tan et al. (2011), “Hidden Randomness between Fitness Landscapes Limits Reverse Evolution,” *Physical Review Letters* 106:198102, [DOI](https://doi.org/10.1103/PhysRevLett.106.198102). |
| G | *Saccharomyces cerevisiae* | 6 | 64/64 | Growth rate | Deleterious / individual | [59] | Hall, Agan & Pope (2010), “Fitness epistasis among 6 biosynthetic loci in the budding yeast Saccharomyces cerevisiae,” *Journal of Heredity* 101(Suppl.1):S75–S84, [DOI](https://doi.org/10.1093/jhered/esq007). |
| H | *Aspergillus niger* | 8 | 186/256 | Growth rate | Deleterious / individual | [30] | Franke et al. (2011), “Evolutionary Accessibility of Mutational Pathways,” *PLOS Computational Biology* 7(8):e1002134, [DOI](https://doi.org/10.1371/journal.pcbi.1002134). |
| I–J | Terpene synthase | 9 | 418/512 | Enzymatic specificity | Not specified | [60] | O’Maille et al. (2008), “Quantitative exploration of the catalytic landscape separating divergent plant sesquiterpene synthases,” *Nature Chemical Biology* 4:617–623, [DOI](https://doi.org/10.1038/nchembio.113). |
| — | Dihydrofolate reductase | 5* | 29/48 | Resistance / growth rate | Beneficial / individual and combined | [61] | Brown et al. (2010), “Compensatory mutations restore fitness during the evolution of dihydrofolate reductase,” *Molecular Biology and Evolution* 27:2682–2690, [DOI](https://doi.org/10.1093/molbev/msq160). |
| — | Dihydrofolate reductase | 5* | 29/48 | Resistance / growth rate | Beneficial / individual and combined | [62] | Costanzo, Brown & Hartl (2011), “Fitness Trade-Offs in the Evolution of Dihydrofolate Reductase and Drug Resistance in Plasmodium falciparum,” *PLOS ONE* 6:e19636, [DOI](https://doi.org/10.1371/journal.pone.0019636). |
| — | HIV-1 envelope glycoprotein gp120 | 7 | 56/128 | Infectivity | Beneficial / individual and combined | [63] | da Silva et al. (2010), “Fitness epistasis and constraints on adaptation in a human immunodeficiency virus Type 1 protein region,” *Genetics* 185:293–303, [DOI](https://doi.org/10.1534/genetics.109.112458). |
| — | Isopropylmalate dehydrogenase | 6* | 164/512 | Performance / fitness | Not specified | [64] | Lunzer et al. (2005), “The biochemical architecture of an ancient adaptive landscape,” *Science* 310:499–501, [DOI](https://doi.org/10.1126/science.1115649). |

Exact Table1 size-cell quotes and locators: A “4 16/16”; B “5 32/32”; C–D “4 16/16”; E/F “5 32/32”; G “6 64/64”; H “8 186/256”; I–J “9 418/512”; Brown/Costanzo “5 29/48”; da Silva “7 56/128”; Lunzer “6 164/512” (all Table 1, printed p.9). The `record.json` overlap entries quote the exact Table2 target cell for each individual statistic.

Table1 footnotes (printed p.9), paraphrased: (a) the DHFR mutations were selected for resistance, not growth in drug-free conditions; (b) the most resistant β-lactamase genotype was created by gene-shuffling, so an accessible path was not guaranteed; (c) F uses the E mutation set in piperacillin plus inhibitor, where the wild type was expected to be unusually fit; (d) missing entries can be chance omissions or nonviable phenotypes, and the Brown/Costanzo data were omitted from quantitative analysis for missingness; (e) I and J are relative product proportions for TEAS and HPS; (f) some loci include multiple possible mutations, so the genotype count exceeds `2^L`; (g) the remaining HIV and Lunzer combinations were not engineered and those data were omitted from analysis.

Table1-only Brown, Costanzo, da Silva, and Lunzer rows have no Table2 statistics printed in Szendro, so they are data-availability checks rather than reproduction targets here.

## Table 2: complete numerical transcription

The Table2 caption is at printed p.15. Values below are the printed columns in order; each paper target in `record.json` has its own quoted cell and locator. `Ncp` and `fmm` are retained here even though GraphFLA has no direct metric for either one.

| ID | r/s | F1 | F2 | Fsum | Nmax | Ncp | fr | fs | fmm | Locator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | “0.122” | “0.989” | “0.009” | “0.011” | “1” | “24” | “0” | “0” | “1” | Table 2, p.15, row A |
| B | “0.290” | “0.942” | “0.040” | “0.058” | “1.10” | “16.80” | “0.013” | “0.150” | “1.92” | Table 2, p.15, row B |
| C | “0.517” | “0.267” | “0.400” | “0.733” | “2” | “16” | “0.083” | “0.250” | “0.67” | Table 2, p.15, row C |
| D | “0.986” | “0.537” | “0.197” | “0.463” | “2” | “10” | “0.125” | “0.458” | “0.67” | Table 2, p.15, row D |
| E | “0.418” | “0.894” | “0.064” | “0.106” | “1.50” | “6.53” | “0.025” | “0.150” | “1.09” | Table 2, p.15, row E |
| F | “0.380” | “0.921” | “0.061” | “0.079” | “1.30” | “8.75” | “0.050” | “0.250” | “3.03” | Table 2, p.15, row F |
| G | “1.180” | “0.658” | “0.179” | “0.342” | “2.13” | “2.10” | “0.229” | “0.358” | “3.16” | Table 2, p.15, row G |
| H | “1.304” | “0.547” | “0.269” | “0.453” | “2.61” | “2.31” | “0.154” | “0.262” | “2.19” | Table 2, p.15, row H |
| I | “1.317” | “0.376” | “0.368” | “0.624” | “2.66” | “2.02” | “0.240” | “0.292” | “1.71” | Table 2, p.15, row I |
| J | “1.199” | “0.383” | “0.372” | “0.617” | “2.48” | “2.51” | “0.227” | “0.300” | “1.92” | Table 2, p.15, row J |
| HoC reference | “2.423” | “0.267” | “0.402” | “0.733” | “3.20” | “1” | “0.333” | “0.333” | “2.20” | Table 2, p.15, HoC row |
| PA reference | “0” | “1” | “0” | “0” | “1” | “24” | “0” | “0” | “1” | Table 2, p.15, PA row |

Table2 footnotes (printed p.15), paraphrased: C is the pyrimethamine-resistance proxy; D is the drug-free growth-rate proxy; F is piperacillin plus inhibitor resistance; I is 5-epi-aristolochene output; J is premnaspirodiene output.

## Definitions, coordinate system, and GraphFLA mapping

### r/s

For the r/s fit, Eqs. (3)–(5), printed pp.10–11, are the exact regression, slope, and roughness formulas:

- Eq. (3): `f^fit(σ⃗)=a^(0)+Σ_(j=1)^L a_j^(1)σ_j`.
- Eq. (4): `s=(1/L)Σ_(j=1)^L |a_j^(1)|`.
- Eq. (5): `r=√(2^(−L)Σ_σ⃗(f(σ⃗)−f^fit(σ⃗))²)`.

Section 2, printed p.5, writes the genotype coordinates as `σ_i=0 (1)` for absent (present) mutation. Equation (3) uses those binary coordinates. The later ±1 spin recoding is `σ̃_j=2σ_j−1=±1` in Eq. (6), printed p.11, for the Fourier expansion. It is not the r/s regression coding. GraphFLA Boolean inputs also use 0/1.

### Fourier fractions

Eq. (7), printed pp.11–12, gives `F_n=β_n/(Σ_j β_j)` and `β_n=Σ_j(b_j^(n))²`. Eq. (8), printed p.12, gives `Fsum=Σ_(j=2)^L F_j`. For a complete, balanced four-locus Boolean cube, GraphFLA order-1 polynomial OLS R² equals F1; order-2 R² minus order-1 R² equals F2; and 1−order-1 R² equals Fsum. GraphFLA `higher_order_epistasis` itself returns cumulative R², so the F2 and Fsum comparisons use those explicit differences/complements, not a direct label equality.

### Peaks, class fractions, and accessible paths

- For `Nmax`, the paper uses the phrase “local fitness maxima Nmax” (Section 3.2.1, item 3, printed p.12). Strict handling of a tied neighbor is NOT STATED IN PAPER. GraphFLA epsilon=0 treats ties as neutral and they disqualify a strict peak.
- For `f_s` and `f_r`, Section 3.2.1 item 4, printed p.12, identifies “simple sign epistasis” and “reciprocal sign epistasis” among local Hamming-distance-2 motifs. GraphFLA `classify_epistasis` partitions only the detected magnitude/sign/reciprocal square classes and uses a directed improving-edge graph. The paper does not give an algebraic denominator formula; the exact formula is NOT STATED IN PAPER. GraphFLA uses a different denominator and representation, so the metrics are definition-incompatible.
- `Ncp` is described in Section 3.2.1 item 5, printed p.12, as “crossing accessible paths”; the paper counts monotone shortest paths from the antipodal genotype to the fittest genotype. No direct GraphFLA path-count metric overlaps Ncp.
- `fmm` is introduced in Section 3.2.1 item 6, printed p.13. It compares accessible paths allowing detours against the additive-landscape path count. No direct GraphFLA metric overlaps it.

### Subgraphs and fitness preprocessing

The paper’s Section 3.2.2, printed p.13, defines a subgraph as the hypercube over 2^m combinations of m selected mutations and says “all subgraphs of size m=4” with at least eight viable and known states are averaged. Whether backgrounds outside the WT context are included is NOT STATED IN PAPER. I interpreted “all possible” literally as all 4-locus faces including each fixed background of the other loci. The cited Franke 2011 method uses WT-anchored subsets, so I also ran that interpretation and saved both outputs. No variant was chosen because it matched.

The paper says it applied logarithms consistently for resistance/fitness inputs, except H, whose nonviable states make the log undefined; Table2 H was not logged. Tan F is stated already in logs. All local CSVs were passed through as stored with no additional transformation. Whether every stored repository vector already contains the article’s log transform is NOT STATED IN PAPER and is not specified in the CSV headers.

## GraphFLA reproduction results

All GraphFLA calls used `BooleanLandscape().build_from_data(X, fitness, epsilon=0, verbose=False)` for binary maps, and `A.list_metrics()` was checked. The primary method averages every eligible all-face 4-locus subgraph. The exact full per-face run output is in `papers/Szendro2013/logs/reproduction_all_faces_zero_codex.json`; the WT-anchored, observed-only, and all-face observed-only outputs are also retained in that folder.

| ID | r/s candidate | mean n_lo candidate | F1 | F2 | Fsum | Outcome summary |
|---|---:|---:|---:|---:|---:|---|
| A | .121797 | 1.000 | .989489 | .009206 | .010511 | Reproduced at printed precision for all five comparable values. |
| B | .734826 | 1.600 | .701689 | .142192 | .298311 | All five mismatch. |
| C–D | candidate 1.174829 | candidate 2.000 | .434193 | .274471 | .565807 | Diagnostic only; exact resistance/growth input mapping unresolved. |
| E | .421989 | 1.100 | .893836 | .063757 | .106164 | r/s and Nmax mismatch; F1/F2/Fsum reproduce at printed precision. |
| F | .390976 | 1.100 | .914056 | .063124 | .085944 | All five mismatch. |
| G haploid candidate | 1.182961 | 3.000 | .502303 | .229223 | .497697 | All five mismatch at printed precision; exact ploidy is ambiguous. |
| G diploid alternative | 1.187341 | 3.083 | .514826 | .197644 | .485174 | All five mismatch. |
| H, `m`→0 | 1.225123 | 2.233 | .551261 | .265431 | .448739 | All five mismatch; H input is rounded in Table S1 and zero treatment is inherited from the cited Franke method. |
| I/J | — | — | — | — | — | Input unavailable; exact publisher supplement URL is in `blockers`. |

The WT-anchored alternative yields different means for L>4, preserved in `logs/reproduction_wt_anchored_zero_codex.json`. For example, it gives r/s and mean Nmax: B (.539215, 1.600), E (.518575, 1.200), F (.404520, 1.200), G haploid (1.811473, 2.867), and H (1.409928, 2.943). These are retained as a sensitivity run, not substituted for the all-face primary method. The observed-only H variants are also logged; omitting nonviable states changes r/s but does not resolve the printed target.

## Input availability checks for all Table1 rows

- A: `data/BioSequence/Chou2011.csv`, 16 unique genotypes.
- B: `data/BioSequence/Khan2011Flynn2013_DM25.csv`, 32 unique genotypes. The EGTA and guanazole settings also exist but were not substituted for DM25.
- C/D: the repository has `Lozovsky2009Jiang2013_Lozovsky2009.csv` and `..._Jiang2013.csv`, both 16 rows, but each has one unlabeled `fitness` column. The original Table2 uses two proxies. The PNAS SI URL for the original [27] Tables S1–S2 returned HTTP 403; exact source mapping remains unresolved.
- E: `data/BioSequence/Weinreich2006Tan2011_Weinreich2006.csv`, 32 unique genotypes. It was used unchanged. A separate GraphFLA validation audit notes one source-value discrepancy in this CSV; it was not corrected here.
- F: `data/BioSequence/Weinreich2006Tan2011_Tan2011.csv`, 32 unique genotypes. Used unchanged.
- G: `Hall2010_haploid.csv` and `Hall2010_diploid.csv`, each 64 unique genotypes. Both were run because Szendro Table1 specifies growth rate; exact ploidy is NOT STATED IN PAPER.
- H: the downloaded Franke Table S1 contains 256 possible 8-bit states, 186 measured and 70 marked m. The source article states missing genotypes were assigned zero fitness; the 70 m states were filled with zero for this primary input, without log transform.
- I/J: no matching input exists in GraphFLA’s `data/` tree. O’Maille’s supplementary Table 3 is the exact source needed; the NCBI direct response was an anti-bot proof page, not the data file.
- Brown2010 and Costanzo2011: no matching GraphFLA input found. Brown’s supplementary material is linked from [10.1093/molbev/msq160](https://doi.org/10.1093/molbev/msq160); Costanzo’s File S1 is linked at [10.1371/journal.pone.0019636.s001](https://doi.org/10.1371/journal.pone.0019636.s001). Szendro excludes both from Table2, so no numeric target exists here.
- daSilva2010: GraphFLA has 5-position, 32-row CCR5 and CXCR5 files; Szendro Table1 describes a 7-locus 56/128 gp120 landscape. The available files are not an exact match. Source article: [10.1534/genetics.109.112458](https://doi.org/10.1534/genetics.109.112458). No Table2 target exists here.
- Lunzer2005: GraphFLA has three 512-row files, but Szendro Table1 lists 164/512 measured variants. Dryad metadata identifies `Lunzer Curated Biochemistry.xls` at [10.5061/dryad.7nd70](https://doi.org/10.5061/dryad.7nd70); the file stream returned 403 and its API download returned 401. No Table2 target exists here.

The plot-only Figures 4–5 were not used as reproduction targets. All numeric Table2 targets in `record.json` use the printed table cell as `quote` with the exact row/column locator; computed GraphFLA numbers are supported by the saved reproduction scripts and logs.
