---
api_narrative: true
---

Epistasis – interactions among mutations that produce nonadditive effects on phenotype and fitness – has long been recognized as fundamentally important to understanding the structure and function of genetic pathways and the evolutionary dynamics of complex genetic systems. It plays a major role in evolution by determining the accessibility of mutational pathways and thereby influencing the rate of adaptation and the diversity and robustness of genetic mutants. 

In simple terms, it describes the phenomenon that the fitness effect of a mutation depends on what other mutations are already present (i.e., genetic background). `GraphFLA` provides various analysis tools as introduced in this page to quantify epistasis in empirical data. 

## Overview

| API | Purpose |
| --- | --- |
| [`classify_epistasis`](#basic-types-of-epistasis) | Proportions of magnitude / sign / reciprocal-sign / positive / negative epistasis via 4-node motifs. |
| [`walsh_hadamard`](#walsh-hadamard-coefficients) | Coefficients and contributions by interaction order, using OLS or Lasso. |
| [`diminishing_returns_index`](#diminishing-returns-epistasis) | Correlation between background fitness and improvement size. |
| [`increasing_costs_index`](#increasing-cost-epistasis) | Correlation between background fitness and detrimental-mutation cost. |
| [`gamma`](#gamma) | Correlation of mutation effects across single-mutant neighbors (Ferretti et al. 2016). |
| [`gamma_star`](#gamma-star) | Sign-only variant of `gamma`. |
| [`idiosyncratic_index`](#per-mutation-index) | Per-mutation idiosyncrasy (Lyons et al. 2020). |
| [`global_idiosyncratic_index`](#global-landscape-wide-index) | Landscape average of `idiosyncratic_index` across all mutations. |
| [`extradimensional_bypass`](#extradimensional-bypass-analysis) | Fraction of reciprocal-sign motifs that admit a fitness-improving bypass. |

## Basic Types of Epistasis


Calculates the prevalence of five basic types of pairwise epistasis in a fitness landscape. 

**Description**

Specifically, epistatic interactions can be either classified into (e.g., [Phillips 2008](https://www.nature.com/articles/nrg2452)): 

- **Negative epistasis:** It occurs when the combined effect of two or more mutations on fitness is less favorable than would be predicted from their individual effects. 
    - For deleterious mutations, it means their joint impact results in a synergistic fitness defect that is greater, or more severe, than expected. In extreme cases, this can lead to synthetic lethal or sick interactions, in which a combination of two individually viable mutations causes death ([Wang *et al.* 2017](https://www.cell.com/cell/fulltext/S0092-8674(17)30061-2)).
    - For beneficial mutations, it means their combined beneficial effect on fitness is smaller than anticipated. This phenomenon is often described as *diminishing-returns epistasis* (see later) or antagonistic.
- **Positive epistasis:** It occurs when the combined effect of two or more mutations on fitness is more favorable than would be predicted from their individual effects.
    - For beneficial mutations, this means their joint impact leads to a synergistic fitness increase that is greater than anticipated.
    - For deleterious mutations, this means their combined detrimental effect on fitness is less severe, or weaker, than expected (i.e., antagonistic). An extreme form of positive interaction is genetic suppression, where the double mutant exhibits better fitness than the least fit single mutant ([Leeuwen *et al.* 2016](https://www.science.org/doi/10.1126/science.aag0839)).

Or (e.g., see [Poelwijk 2007](https://www.nature.com/articles/nature05451)):

- **Magnitude epistasis:** In magnitude epistasis, the effect of a mutation (e.g., beneficial or deleterious) maintains its sign (direction) regardless of the genetic background. However, the magnitude (strength) of this effect changes depending on what other mutations are present. For example, a mutation might be strongly beneficial in one genetic background but only weakly beneficial in another, yet it remains beneficial in both contexts. The fitness effects are non-additive, but the qualitative outcome (beneficial/deleterious) for a given mutation does not change.
- **Sign epistasis:** Sign epistasis occurs when the effect of a mutation changes its sign (i.e., from beneficial to deleterious, or vice versa) depending on the genetic background (the presence of other mutations). A mutation that is advantageous in one genetic context might be detrimental in another, or a deleterious mutation might become beneficial when another specific mutation is present. This type of epistasis is significant because it can create rugged fitness landscapes with multiple fitness peaks, potentially trapping evolving populations at suboptimal states as some evolutionary paths become inaccessible.
- **Reciprocal sign epistasis:** Reciprocal sign epistasis is a specific and stronger form of sign epistasis. It occurs when the sign of the fitness effect of *each* of two interacting mutations is reversed by the presence of the other mutation. A common scenario is where two mutations are individually deleterious (or less fit than the wild type), but their combination results in a genotype that is more fit than either single mutant, and potentially even more fit than the original wild type. This type of interaction is a necessary condition for the existence of multiple peaks on a fitness landscape.

This implementation (from [Papkou *et al.*, 2023](https://www.science.org/doi/10.1126/science.adh3860)) calculates the fraction of each of these 5 types of epistasis among all pairs of mutations (i.e., pairwise epistasis) using the `motif` function in `igraph`. 

!!! note

    The two sets of fractions summarize the directed motifs found in the constructed graph. Their normalization and behavior when no relevant motifs are found are described in Returns below.

!!! warning

    This method can be quite computationally expensive for large landscapes (e.g., >10,000 mutants). Use `sample_cut_prob="auto"` for automatic sampling, or choose a sampling probability explicitly. Set `seed` for reproducible sampling.

::: graphfla.analysis.classify_epistasis

## Higher-order Epistasis

In addition to pairwise epistasis, higher-order epistasis that involves multiple mutations are also common in empirical fitness landscapes (e.g., [Weinreich *et al.* 2013](https://www.sciencedirect.com/science/article/pii/S0959437X13001421), [Domingo *et al.* 2018](https://www.nature.com/articles/s41586-018-0170-7)). `GraphFLA` uses `walsh_hadamard` to estimate coefficients and summarize contributions by interaction order.

### Walsh-Hadamard coefficients

To identify which variables interact, fitness can be represented as a sum of individual effects and interactions. For binary variables, let $z_i=x_i-\tfrac12$, where $x_i\in\{0,1\}$. The fitted landscape takes the form

$$
\widehat f(\mathbf{x})=\varepsilon_0
+\sum_i\varepsilon_i z_i
+\sum_{i<j}\varepsilon_{ij}z_i z_j+\cdots.
$$

Here, $\varepsilon_0$ is the model's mean over uniformly weighted configurations. First-order coefficients describe background-averaged effects of individual changes, second-order coefficients describe pairwise interactions, and higher orders capture effects involving additional variables. The centered encoding determines the scaling of these coefficients.

`GraphFLA` implements the multistate extension of [Faure et al. (2024)](https://doi.org/10.1371/journal.pcbi.1012132). For a variable with $s_i$ states, $z_i$ is replaced by centered indicators $\phi_{i,a}(x_i)=\mathbf{1}[x_i=a]-1/s_i$, one for each nonreference state $a$. Interaction terms multiply indicators from distinct variables.

One call returns coefficients and an order summary. Cumulative R² and its increments describe refitted models through each order; the model variance fractions describe the highest-order fit over uniformly weighted backgrounds. Lasso provides regularized estimates. Its model spectrum and observed-data fit gains answer different questions and are reported separately.

::: graphfla.analysis.walsh_hadamard

## Global Epistasis

Beyond interactions between specific pairs or sets of mutations, global epistasis describes a broader pattern where the fitness effect of a mutation systematically depends on the overall fitness of the genetic background in which it arises. This often approximately one-dimensional relationship can emerge from the combined effects of many specific, idiosyncratic interactions, which may also explain variation around the overall trend.

Common forms of these fitness-correlated trends are diminishing returns for beneficial mutations and increasing costs for deleterious mutations, with the same mutation having a less positive or more negative effect on fitter backgrounds. The returns and costs indices pool different mutations, so their signs alone do not establish mutation-specific epistasis or statistical significance.

### Diminishing Returns Epistasis

Diminishing returns epistasis is a form of global epistasis that applies to beneficial mutations. It describes the pattern where the positive fitness effect of a mutation is smaller when it occurs in an already fit genetic background than in a less fit background. As beneficial mutations accumulate, shrinking gains can contribute to decelerating adaptation ([Chou et al., 2011](https://doi.org/10.1126/science.1203799)).

GraphFLA calculates the correlation or regression slope between background fitness and the positive fitness increase of beneficial transitions, with each retained transition contributing equally. A negative value describes diminishing gains across this pooled set. The result depends on the fitness scale and the mutations available in each background.

::: graphfla.analysis.diminishing_returns_index

### Increasing Cost Epistasis

Increasing cost epistasis is the analog of diminishing returns for deleterious mutations. It describes a pattern where a mutation's fitness cost becomes more severe in fitter backgrounds. The same mutation may have a weaker detrimental effect in less fit backgrounds ([Johnson et al., 2019](https://doi.org/10.1126/science.aay4199)).

GraphFLA correlates background fitness with the positive magnitude of fitness loss, or fits a regression slope, weighting each retained deleterious transition equally. A positive value describes increasing costs. Each transition reverses a stored improving edge and starts at its better endpoint.

::: graphfla.analysis.increasing_costs_index

## Gamma Statistic

### Gamma

The gamma statistic ($\gamma$) measures the amount of epistasis in a fitness landscape through the correlation between fitness effects of mutations across different genetic backgrounds ([Ferretti *et al.*, 2016](https://doi.org/10.1016/j.jtbi.2016.01.037)). It compares the effects of the same mutation in backgrounds differing by a single mutation at another locus. In simpler terms, it quantifies how the effect of a particular mutation is altered by another mutation in the genetic background, averaged across the landscape. For individual two-locus, two-allele squares, the following relationships hold; the ranges overlap and do not define classification thresholds for an entire landscape:

- $\gamma = 1$: no epistasis; mutation effects are unchanged across backgrounds.
- *Magnitude epistasis* $\rightarrow 0 \leq \gamma < 1$: mutation effects retain their signs but differ in magnitude, giving a nonnegative correlation.
- *Sign epistasis* $\rightarrow -\frac{1}{3} \leq \gamma < 1$: the effect of one mutation changes sign, and the correlation can be positive, zero, or negative.
- *Reciprocal sign epistasis* $\rightarrow -1 \leq \gamma < 0$: the effects of both mutations change sign, giving a negative correlation.

::: graphfla.analysis.gamma

### Gamma Star

The gamma-star statistic ($\gamma^*$) is a variant of $\gamma$ that focuses only on sign consistency, ignoring the magnitude of fitness effects. It indicates whether mutations tend to have consistent directional effects across different genetic backgrounds.

::: graphfla.analysis.gamma_star

## Idiosyncratic Epistasis

The fitness-correlated trends described above summarize how mutation effects change with background fitness. Individual mutations can also depend on the specific alleles present in a background. For example, a substitution that improves a protein's activity in one sequence may become harmful after another residue changes. Such interactions are described as idiosyncratic epistasis and help explain why a mutation's effect in one genotype may not predict its effect in another.

[Lyons et al. (2020)](https://doi.org/10.1038/s41559-020-01286-y) introduced the idiosyncratic index to quantify background dependence. For a given mutation, it compares the standard deviation of its effects across matching backgrounds with that of fitness differences in an equally sized sample of random genotype pairs. GraphFLA summarizes the landscape by averaging these ratios across eligible directed mutations, giving each mutation equal weight.

When defined, an index of zero indicates constant mutation effects across the observed backgrounds. A value near one indicates variation comparable to random genotype-pair differences; estimates can exceed one. The index measures variation, rather than the fraction of epistatic mutations. Because a nonlinear global fitness relationship can also generate background dependence, a positive index alone does not distinguish specific interactions from global epistasis.

### Per-mutation Index

::: graphfla.analysis.idiosyncratic_index

### Global (Landscape-Wide) Index

::: graphfla.analysis.global_idiosyncratic_index

## Extradimensional Bypass Analysis


Detects extradimensional bypasses around reciprocal-sign-epistasis (RSE) motifs.

**Description**

Reciprocal sign epistasis occurs when the wildtype (`ab`) and the double mutant (`AB`) are both fitter than the two intermediate single mutants (`aB`, `Ab`). This creates a fitness valley along the direct path from `ab` to `AB` that natural selection cannot cross. However, when the landscape has more than two relevant dimensions, the population can sometimes traverse a *third-position* intermediate (i.e., gain and later lose an extra mutation) to bypass the valley. Such indirect uphill paths are called **extradimensional bypasses** and they greatly affect a landscape's effective navigability.

For each RSE motif in the landscape (igraph isomorphism class 19), this function checks whether the directed improving graph admits any path from `ab` to `AB`. The fraction of RSE motifs that *do* admit such a bypass is the **bypass proportion**.

::: graphfla.analysis.extradimensional_bypass
