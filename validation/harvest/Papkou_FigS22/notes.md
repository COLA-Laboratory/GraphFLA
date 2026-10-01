# Papkou et al. 2023 — Fig. S22 raw reconstruction

Harvest directory: Papkou_FigS22. The registry key in record.json is PapkouRM23, matching the corpus registry. I read the existing Papkou2023 dossier before fetching anything and reused its processed input and prior audits.

## Bibliography and source provenance

The DOI lookup was made by title and DOI, not by the supplied author string. The saved Crossref REST response is papers/Papkou2023/sources/Papkou2023_crossref_adh3860_codex.json. It verifies the title “A rugged yet easily navigable fitness landscape,” DOI 10.1126/science.adh3860, year 2023, and authors Andrei Papkou, Lucia Garcia-Pastor, José Antonio Escudero, and Andreas Wagner. The supplied registry author string (“Andrei Papkou and Aviv Regev and Joanna Masel”) is wrong.

I reused the existing main paper PDF and the SI text extract at papers/Papkou2023/sources/Papkou2023_science_adh3860_SM_codex.txt. That dossier copy is byte-identical to the read-only extract at /Users/arwen/Documents/GitHub/GraphFLA/_verify/papers/papkou_science_SM.txt (SHA-256 abd424ad6fee922b61e34ccb961490e28f820cdc17a4057dc123dc783b293072). The paper dossier's source index already records that the publisher SI PDF route had an access gap. A fresh request to the Science SI PDF URL returned HTTP 403 and an HTML security-verification response, retained as papers/Papkou2023/sources/Papkou2023_science_adh3860_supplement_attempt_codex.html and logged in acquisition_codex.jsonl. I did not retrieve a publisher Source Data attachment. I did not read any value from the plotted line or plotted points.

## Verbatim paper evidence

### Study sequence and target site

Main paper, printed p.1:

> “We performed CRISPR-Cas9 (45) deep mutagenesis to edit the folA gene on the bacterial chromosome, randomizing nine nucleotide positions in a part of the gene that is both conserved and implicated in the evolution of antibiotic resistance (Fig. 1A).”

Main paper, printed p.1:

> “three successive amino acids of DHFR (wild-type sequence: 26A-27D-28L)”

This establishes that position 27 is the middle of the three targeted codons. The exact nucleotide triplets used to define the Fig. S22A subset are NOT STATED IN PAPER.

### Global-epistasis method and panel A population

Supplementary Materials, “Estimating global epistasis,” printed p.19:

> “We detected the presence of global epistasis as a nonlinear dependence between fitness values and the sum of linear predictors of the first order (72). To this end, we estimated first-order additive effects of each allele for each position using a linear regression model. We sum the first order effects and mapped them to fitness values of corresponding variants via a nonlinear monotonically increasing function. In particular, we used I-splines basis functions (72).”

Supplementary Fig. S22 caption, printed p.42:

> “Panel (A) demonstrates a nonlinear function (blue line), mapping the sum of first-order predictors onto the fitness of 24,542 variants with the Asp27/Glu27/Cys27 genotype.”

The caption, rather than the Methods subsection, is where the count and amino-acid subset are stated. The Methods do not provide a nucleotide-level membership list or state that this subset is restricted to the processed graph component: NOT STATED IN PAPER.

Input-side count check, not a new paper target: I translated the middle codon in the full RAW input using the existing dossier's validated symbol mapping. That file yields 24,542 Asp/Glu/Cys-at-27 sequences, matching the caption. The trusted processed component input yields 23,745 such sequences, so 797 group members are absent from that component. These file-derived counts and the inferred full-raw scope are NOT STATED IN PAPER. The processed file alone therefore does not contain the entire panel-A population. I did not refit the panel-A I-spline.

The Supplementary Materials also describe a different, smaller set for higher-order epistasis. Printed p.19:

> “Therefore, we focused on the subset of only Asp27 and Glu27 variants, which include most functional variants and all high fitness peaks (N=16,353).”

That Asp/Glu-only set is not the Asp/Glu/Cys panel-A population in Fig. S22.

### Beneficial-mutation population for panels B and C

Supplementary Materials, “Analysis of the fitness landscape—Constructing a network of variants,” printed p.15:

> “Exclude pairs where both variants are nonfunctional (i.e., their fitness is below the cut-off)”

> “Construct a network by connecting each pair of neighboring variants with a directed edge (arrow). Any one such edge corresponds to a fitness-increasing mutation, i.e., it points from the variant with lower fitness to the neighbor with higher fitness.”

> “This connected subgraph of variants is the fitness landscape we analyze. It contains 135,178 variants and 324,044 edges between them.”

Supplementary Materials, “Determining nonfunctional mutations,” printed pp.14–15:

> “This procedure resulted in a relative fitness cut-off of 𝑟 𝑖 − 𝑟 WT = -0.507774.”

Supplementary Fig. S22 caption, printed p.42:

> “The fitness gain of beneficial mutations (vertical axis) decreases in genetic backgrounds with higher fitness (horizontal axis). The blue line represents a linear regression model (fitness gain is the response variable and the fitness of the genetic background is the predictor variable). N=324,044 mutations. (B) shows a magnified part of (C).”

Supplementary Fig. S23 caption, printed p.43:

> “N= 324,044, N is the total number of connected pairs of variants in the landscape.”

These statements identify the Fig. S22 “mutations” as the same population of one-mutant neighbor pairs represented by the directed improving edges: the Methods explicitly say one directed edge is one fitness-increasing mutation, and both captions use the same N. GraphFLA and a separate exhaustive one-nucleotide adjacency enumeration returned 324,044 entries and the exact same oriented edge set. Thus this count is not doubled to represent undirected pairs.

### Computed reconstruction from the trusted input

Input reused for panels B/C and DRI: data/BioSequence/Papkou2023_DHFR.csv; SHA-256 fca2fb47cad417a049698f43b20f90b320a4927418d502e527ad112ea46c82da. Existing dossier preprocessing and input audit were reused. The GraphFLA run used DNALandscape, epsilon=0, tau=-0.507774, and filter_mode="both". All direct neighboring pairs were oriented from lower fitness to higher fitness after excluding pairs with both endpoints below the stated cutoff.

For the reconstruction, I computed gain as fitness at the fitter endpoint minus fitness at the background endpoint. The paper's exact gain equation is NOT STATED IN PAPER. Panel B was independently reconstructed by direct one-nucleotide neighbor enumeration; panel C was independently reconstructed from the GraphFLA graph edge list. They contain the same exact observations; the caption says panel B is a magnified part of panel C.

| Panel reconstruction | OLS slope | Intercept | Pearson correlation | n |
| --- | ---: | ---: | ---: | ---: |
| B, direct neighbor enumeration | -0.432920741883921 | 0.43477401242456315 | -0.4764872794862937 | 324044 |
| C, GraphFLA graph edge list | -0.43292074188392254 | 0.4347740124245627 | -0.47648727948629294 | 324044 |

The regression line and N are printed/described in the caption, but its slope, intercept, and correlation are not printed: NOT STATED IN PAPER. The values above are reconstructions from the trusted input, not reproductions of printed coefficients. The two calculations differ only at floating-point round-off.

## GraphFLA diminishing_returns_index comparison

The live analysis inventory includes diminishing_returns_index. On the same processed input and graph, its default result is -0.3834071314404972. This value is NOT STATED IN PAPER.

GraphFLA's read-only function docstring, graphfla/analysis/epistasis/idiosyncrasy.py, function diminishing_returns_index, lines 263–270, states:

> “This function quantifies this trend by calculating the correlation between the fitness of each genotype (node) and the average fitness improvement provided by its direct successors (fitter one-mutant neighbors).”

The docstring also specifies the default method as Pearson correlation. The reported DRI is therefore a node-level Pearson correlation on each genotype's mean outgoing gain, while Fig. S22's blue line is an edge-level regression on individual mutation gains. The DRI collapses all outgoing edges from a background to one average and gives each background with improving edges one observation; the panel regression gives each mutation edge one observation. The graph has 134,664 distinct source backgrounds with improving edges, a file-derived count that is NOT STATED IN PAPER. Both quantities are negative, so they agree on direction, but DRI is not a faithful numeric summary of the panel slope: its statistic, sample weighting, and scale differ.

## Execution and acquisition notes

- The direct neighbor enumeration and GraphFLA edge-list regressions were both completed; edge sets were exactly equal.
- The first scratch-script execution stopped during result formatting because SciPy's LinregressResult did not expose an n attribute. I corrected the script to use the input-vector length and reran successfully; no statistics from the failed execution were retained.
- The first GraphFLA import also triggered Matplotlib to create a temporary cache under system temp because its default cache directory was not writable. That write occurred outside the validation workspace inadvertently. The successful rerun set MPLCONFIGDIR inside the validation workspace. PYTHONDONTWRITEBYTECODE=1 was set on both runs, and the GraphFLA repository was not modified.
- The official SI request failure, response hash, and Crossref DOI metadata download are recorded in papers/Papkou2023/acquisition_codex.jsonl.
