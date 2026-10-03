---
icon: material/cube-outline
---

Epistasis – interactions among mutations that produce nonadditive effects on phenotype and fitness – has long been recognized as fundamentally important to understanding the structure and function of genetic pathways and the evolutionary dynamics of complex genetic systems. It plays a major role in evolution by determining the accessibility of mutational pathways and thereby influencing the rate of adaptation and the diversity and robustness of genetic mutants. 

In simple terms, it describes the phenomenon that the fitness effect of a mutation depends on what other mutations are already present (i.e., genetic background). `GraphFLA` provides various analysis tools as introduced in this page to quantify epistasis in empirical data. 

## Overview

| API | Purpose |
| --- | --- |
| [`classify_epistasis`](#basic-types-of-epistasis) | Proportions of magnitude / sign / reciprocal-sign / positive / negative epistasis via 4-node motifs. |
| [`walsh_hadamard_coefficient`](#walsh-hadamard-coefficients) | Exact Walsh–Hadamard coefficients (multi-state extension, Faure et al. 2024). |
| [`higher_order_epistasis`](#regression-approximation) | $R^2$ of decision-tree fits up to a specified interaction order (cheap approximation). |
| [`diminishing_returns_index`](#diminishing-returns-epistasis) | Correlation between background fitness and improvement size. |
| [`increasing_costs_index`](#increasing-cost-epistasis) | Correlation between background fitness and detrimental-mutation cost. |
| [`gamma_statistic`](#default-gamma-statistic) | Correlation of mutation effects across single-mutant neighbors (Ferretti et al. 2016). |
| [`gamma_star`](#gamma-star-statistic) | Sign-only variant of `gamma_statistic`. |
| [`idiosyncratic_index`](#per-mutation-index) | Per-mutation idiosyncrasy (Lyons et al. 2020). |
| [`global_idiosyncratic_index`](#global-landscape-wide-index) | Landscape average of `idiosyncratic_index` across all mutations. |
| [`extradimensional_bypass_analysis`](#extradimensional-bypass-analysis) | Fraction of reciprocal-sign motifs that admit a fitness-improving bypass. |

## Basic Types of Epistasis


```api
def graphfla.analysis.classify_epistasis(landscape, approximate=False, sample_cut_prob=0.2) → Dict[str, float]
```

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

    The fractions for positive and negative epistasis always sum to 1, as they are mutually exclusive and exhaustive. Likewise, the fractions for magnitude, sign, and reciprocal sign epistasis also sum to 1.

!!! warning

    This method can be quite computationally expensive for large landscapes (e.g., >10,000 mutants). Using `approximate=True` and higher `sample_cut_prob` value is highly recommended.

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   The fitness landscape object.

    **approximate** : `bool`, default=`False`
    :   - If `True`, the function will estimate motif counts instead of performing exact enumeration. This can significantly speed up computation for large graphs, but at the cost of precision and introduce stochasticity.
        - If `False` (the default), exact motif counts are used. Not recommended for large landscapes (e.g., >10,000 mutants).

    **sample_cut_prob** : `float`, default=`0.2`
    :   This parameter is only relevant when `approximate` is `True`. It represents the probability used for pruning the search tree at each level during the sampling process for motif instances. Higher values will lead to faster computation but potentially less accurate estimations. The value must be between 0 and 1.

!!! api-returns "Returns"

    **dict[str, float]**
    :   A dictionary where keys are 5 epistasis types and values are their calculated proportions.

**References**

- Patrick C. Phillips. "Epistasis--the essential role of gene interactions in the structure and evolution of genetic systems." *Nat. Rev. Genet.* (2008).
- Tim Wang, "Gene Essentiality Profiling Reveals Gene Networks and Synthetic Lethal Interactions with Oncogenic Ras." *Cell*. (2017).
- Frank J. Poelwijk *et al.*, "Empirical fitness landscapes reveal accessible evolutionary paths." *Nature.* (2007).
- Jolanda van Leeuwen *et al.*, "Exploring genetic suppression interactions on a global scale." *Science* (2016).
- Papkou *et al.*, "A rugged yet easily navigable fitness landscape", *Science* (2023).

**Example**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import classify_epistasis

# synthetic landscape with moderate epistasis
problem = NK(n=10, k=5)
X, f = problem.get_data()

# construct landscape instance
landscape = BooleanLandscape()
landscape.build_from_data(
    X, f,
    verbose=False
)

# run epistasis analysis
epistasis = classify_epistasis(landscape)
print(epistasis)
```
```
Output:

{'magnitude epistasis': 0.48723958333333334,
 'sign epistasis': 0.3337673611111111,
 'reciprocal sign epistasis': 0.17899305555555556,
 'positive epistasis': 0.7474826388888889,
 'negative epistasis': 0.2525173611111111}
```

## Higher-order Epistasis

In addition to pairwise epistasis, higher-order epistasis that involves multiple mutations are also common in empirical fitness landscapes (e.g., [Weinreich *et al.* 2013](https://www.sciencedirect.com/science/article/pii/S0959437X13001421), [Domingo *et al.* 2018](https://www.nature.com/articles/s41586-018-0170-7)). `GraphFLA` provides two functions to measure these.

### Walsh-Hadamard Coefficients


```api
def graphfla.analysis.walsh_hadamard_coefficient(landscape, max_order: int = 2, max_cells: float = 1e9, chunk_size: int = 1000) → dict
```

Computes Walsh-Hadamard coefficients to quantify background-averaged epistasis in a fitness landscape. 

**Description**

This function calculates Walsh-Hadamard coefficients for base and interaction terms up to a specified order. Consider a genotype written as binary strings representing the presence/absence of single-point mutations $\sigma=(\sigma_1, \sigma_2, \dots, \sigma_n)$. The fitness function can be decomposed into products of the single-point mutations according to the following expansion ([Hansen & Wagner 2001](https://www.sciencedirect.com/science/article/pii/S0040580900915089); [Weinreich *et al.* 2013](https://www.sciencedirect.com/science/article/pii/S0959437X13001421)):

$$f(\sigma) = a^{(0)} + \sum a_i^{(1)}\sigma_i + \sum a_{ij}^{(2)}\sigma_i\sigma_j + \sum a_{ijk}^{(3)}\sigma_i\sigma_j\sigma_k + \dots + a^{(n)}\sigma_1\sigma_2 \dots \sigma_n$$

There are $C_k^n$ coefficients of type $a^{(k)}$ in this expansion, one for each subset of $k$ of $n$ mutations. The first-order coefficient $a^{(1)}$ describes the linear, non-epistatic effects, the second-order coefficient $a^{(2)}$ denotes pairwise epistatic interactions and so on.

While the traditional Walsh-Hadamard transform is strictly limited to biallelic (binary) sequences, this function implements the generalized multi-state extension detailed by [Faure *et al.* (2024)](https://doi.org/10.1371/journal.pcbi.1012132). This allows the transform to accommodate empirical landscapes of arbitrary shape and complexity.

The function internally identifies the wildtype sequence, dynamically generates all theoretical interaction features up to the requested `max_order`, maps sequences via the extended Walsh-Hadamard ensemble encoding (building the $H$ matrix in memory-optimized chunks), and ultimately computes the exact coefficients using a numerically stable least-squares regression against the phenotypic fitness values.

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   An initialized and built fitness landscape object containing the genotype configurations and corresponding phenotypic fitness data.

    **max_order** : `int`, optional, default=`2`
    :   The maximum interaction order to compute. For example, an order of `1` computes only single-mutation effects, `2` incorporates pairwise genetic interactions, and `3` captures three-way higher-order epistasis. Higher orders reveal more complex epistatic interactions but exponentially increase the number of sequence features and computational load.

    **max_cells** : `float`, optional, default=`1e9`
    :   The maximum number of feature matrix cells allowed in memory during interaction generation. This acts as a safety limit to prevent out-of-memory errors on large combinatorial landscapes. If exceeded, a `ValueError` is raised.

    **chunk_size** : `int`, optional, default=`1000`
    :   The size of the chunks used when constructing the Walsh-Hadamard $H$ matrix. Chunking significantly optimizes memory usage by generating the extended sequence transforms in batches, which is critical for highly multiallelic or deep landscapes.

!!! api-returns "Returns"

    **dict**
    :   A nested dictionary containing the computed coefficients, sorted and organized by their interaction order:

        - Keys are integers representing the interaction order: `0` for the wildtype baseline, `1` for first-order single mutations, `2` for pairwise interactions, etc.
        - Values are sub-dictionaries mapping string feature names to their corresponding floating-point coefficient values.
    
        Feature names use a precise nomenclature formatted as `{original}_{position}_{mutant}` (1-indexed). Higher-order interactions concatenate these individual mutation strings with a hyphen (`-`).

**References**

- Thomas F. Hansen & Günter P. Wagner, "Modeling genetic architecture: a multilinear theory of gene interaction." *Theor. Popul. Biol.* (2001).
- Daniel M. Weinreich *et al.*, "Should evolutionary geneticists worry about higher-order epistasis?" *Curr. Opin. Genet. Dev.* (2013).
- Andre J. Faure *et al.*, "An extension of the Walsh-Hadamard transform to calculate and model epistasis in genetic landscapes of arbitrary shape and complexity." *PLoS Comput Biol.* (2024).

**Example**

```python
from graphfla.landscape import DNALandscape
from graphfla.analysis import walsh_hadamard_coefficient

# Generate a synthetic landscape with epistasis
df = pd.read_csv('GraphFLA/data/Papkou2023_DHFR.csv')
X = df["seq"]
f = df["fitness"]

# Initialize and build the landscape instance
landscape = BooleanLandscape()
landscape.build_from_data(
    X, f,
    verbose=False
)

# Compute Walsh-Hadamard coefficients up to 2nd order (pairwise)
coefficients = walsh_hadamard_coefficient(landscape, max_order=2)

# Retrieve the baseline wildtype phenotypic coefficient
print(f"Wildtype coefficient: {coefficients[0]['WT']}")

# Inspect the first few single-mutation additive effects
single_muts = list(coefficients[1].items())[:2]
print(f"Single mutation effects: {single_muts}")

# Inspect the first pairwise epistatic interaction
pairwise_muts = list(coefficients[2].items())[:1]
print(f"Pairwise interactions: {pairwise_muts}")
```
```
Output:

Wildtype coefficient: -0.6584880710929173
Single mutation effects: [('A_5_C', np.float64(-0.23087751004325047)), ('A_5_G', np.float64(-0.21967308566688537))]
Pairwise interactions: [('A_5_C-C_7_A', np.float64(-0.052641309797158024))]
```

### Regression Approximation


```api
def graphfla.analysis.higher_order_epistasis(landscape, order: int = 2, verbose: bool = False, n_jobs: int = 1) → float
```

Measures the prevalence of higher-order epistasis in the landscape up to a specified order.

**Description**

While Walsh-Hadamard transform can provide exact measures of all orders of epistasis, it is computationally expensive for large landscapes and high-orders. 

As an alternative, `GraphFLA` provides a cheaper yet effective way for approximating the prevalence of high-order epistasis by measuring how much variance in fitness can be explained by considering epistasis up to a specific order. 

This function fits a `DecisionTreeRegressor` from `sklearn` to the landscape data. The order of interaction used for fitting is controlled by `max_depth` of the tree and specified by the user with `order`. It returns the $R^2$ (coefficient of determination) score of the model to indicate how much variance in fitness can be explained by interaction up to the specified order. 

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   The fitness landscape object.

    **order** : `int`, optional, default=`2`
    :   The maximum order of variable interactions to consider. This parameter dictates the maximum depth of the decision tree used for modeling. For instance, an `order` of 2 allows for modeling pairwise interactions. It must be an integer between 1 and the total number of variables in the landscape.

    **verbose** : `bool`, optional, default=`False`
    :   If set to `True`, the function will print progress information during its execution, such as encoding steps and model fitting.

    **n_jobs** : `int`, optional, default=`1`
    :   Number of CPU cores used by the underlying linear regression.

!!! api-returns "Returns"

    **float**
    :   The $R^2$ score, a value between 0.0 and 1.0, indicating the fraction of fitness variance explained by interactions up to the specified `order`. A higher value suggests stronger epistasis.

**References**

- Daniel M Weinreich *et al.*, "Should evolutionary geneticists worry about higher-order epistasis?" *Curr. Opin. Genet. Dev.* (2013).
- Júlia Domingo *et al.*, "Pairwise and higher-order genetic interactions during the evolution of a tRNA." *Nature.* (2018).

**Example**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import higher_order_epistasis

# synthetic landscape with moderate epistasis
problem = NK(n=10, k=5)
X, f = problem.get_data()

# construct landscape instance
landscape = BooleanLandscape()
landscape.build_from_data(
    X, f,
    verbose=False
)

# run epistasis analysis
r2 = higher_order_epistasis(landscape, order=2)
print(r2)
```
```
Output:

0.2970588665015169
```

## Global Epistasis

Beyond interactions between specific pairs or sets of mutations, global epistasis describes a broader pattern where the fitness effect of a mutation *systematically* depends on the overall fitness of the genetic background in which it arises. Instead of the effect being determined by the presence or absence of one or a few specific other mutations, global epistasis implies a more general, often "nonspecific" or "unidimensional," relationship. While global epistasis can emerge from the sum of many specific, idiosyncratic interactions, it presents as an overarching trend.

Typically, global epistasis manifests as "fitness-correlated trends (FCTs)" as either diminishing returns for beneficial mutations or increasing costs for deleterious mutations, with mutations having a less positive or more negative effect on fitter backgrounds.

### Diminishing Returns Epistasis


```api
def graphfla.analysis.diminishing_returns_index(landscape, method: Literal["pearson", "spearman", "regression"] = "pearson") → float
```

Measures diminishing returns epistasis in a fitness landscape.

**Description**

Diminishing returns epistasis is a specific form of global, and often negative, epistasis that applies to beneficial mutations. It describes the phenomenon where the positive fitness effect of a beneficial mutation is smaller when it occurs in a genetic background that is already highly fit (i.e., smoother up-hill paths), compared to its effect in a less fit background. As a population or genotype approaches a fitness peak, the incremental gains from new beneficial mutations tend to shrink, resulting in decelerated adaptation ([Chou *et al.* 2011](https://www.science.org/doi/10.1126/science.1203799)).

This function calculates the correlation (or regression slope) between the fitness increase of beneficial mutations and the fitness of the genetic background in which they occur. A significant negative correlation (or a negative slope) indicates diminishing returns.

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   An initialized and built fitness landscape object.

    **method** : `{'pearson', 'spearman', 'regression'}`, default=`'pearson'`
    :   The method used to calculate the diminishing returns index:

        -   `'pearson'`: Calculates the Pearson correlation coefficient between genotype fitnesses and average fitness improvements.
        -   `'spearman'`: Calculates the Spearman's rank correlation coefficient between genotype fitnesses and average fitness improvements.
        -   `'regression'`: Calculates the slope of a linear regression between genotype fitnesses and average fitness improvements.

!!! api-returns "Returns"

    **correlation or slope** : `float`
    :   -   For `'pearson'` or `'spearman'`: The correlation coefficient between mutant fitness and average fitness improvement.
        -   For `'regression'`: The slope of the linear regression.

**References**

- Hsin-Hung Chou *et al.* "Diminishing returns epistasis among beneficial mutations decelerates adaptation." *Science.* (2011)

**Example**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import diminishing_returns_index

# synthetic landscape with moderate epistasis
problem = NK(n=10, k=5)
X, f = problem.get_data()

# construct landscape instance
landscape = BooleanLandscape()
landscape.build_from_data(
    X, f,
    verbose=False
)

# run epistasis analysis
dr = diminishing_returns_index(landscape, method="spearman")
print(dr)
```
```
Output:

-0.6972993816535936
```

### Increasing Cost Epistasis


```api
def graphfla.analysis.increasing_costs_index(landscape, method: Literal["pearson", "spearman", "regression"] = "pearson") → float
```

Measures increasing cost epistasis in a fitness landscape.

**Description**

Increasing cost epistasis is the analog of diminishing returns but for deleterious mutations. It describes a pattern where the negative fitness effect (or cost) of a deleterious mutation becomes more severe in genetic backgrounds that are already fitter (i.e., steeper down-hill paths). Conversely, in less fit genetic backgrounds, the same deleterious mutation might have a weaker detrimental effect.

It calculates the correlation (or regression slope) between the fitness decrease of deleterious mutations and the fitness of the genetic background in which they occur.  A significant positive correlation (or a positive slope) indicates increasing cost.

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   An initialized and built fitness landscape object.

    **method** : `{'pearson', 'spearman', 'regression'}`, default=`'pearson'`
    :   The method used to calculate the increasing costs index:

        -   `'pearson'`: Calculates the Pearson correlation coefficient.
        -   `'spearman'`: Calculates the Spearman rank correlation coefficient.
        -   `'regression'`: Calculates the slope of a linear regression between mutant fitnesses and average fitness costs.

!!! api-returns "Returns"

    **correlation_or_slope** : `float`
    :   -   For `'pearson'` or `'spearman'`: The correlation coefficient between mutant fitness and average fitness cost.
        -   For `'regression'`: The slope of the linear regression.

**Example**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import increasing_costs_index

# synthetic landscape with moderate epistasis
problem = NK(n=10, k=5)
X, f = problem.get_data()

# construct landscape instance
landscape = BooleanLandscape()
landscape.build_from_data(
    X, f,
    verbose=False
)

# run epistasis analysis
ic = increasing_costs_index(landscape, method="spearman")
print(ic)
```
```
Output:

0.7598145419762149
```

## Gamma Statistic

### Default Gamma Statistic


```api
def graphfla.analysis.gamma_statistic(landscape, n_jobs=-1) → float
```

Calculates the gamma ($\gamma$) statistic for a fitness landscape ([Ferretti *et al.*, 2016](https://www.sciencedirect.com/science/article/pii/S0022519316000771)).

**Description**

The gamma statistic ($\gamma$) is a measure of the amount of epistasis in a fitness landscape by calculating the correlation between fitness effects of mutations across different genetic backgrounds. It is defined as the correlation of fitness effects of the same mutation in single-mutant neighbors. In simpler terms, it quantifies how the effect of a particular mutation is altered by the presence of another mutation at a different locus in the genetic background, averaged across the entire landscape.

- $\gamma = 1 \to$ no epistasis.
- $0 \leq \gamma < 1 \to$ *magnitude epistasis*, which would still result in a positive correlation between fitness effects, therefore $\gamma$ would still be positive even if smaller than 1;
- $-\frac{1}{3} \leq \gamma < 0 \to$ *sign epistasis*, which would contribute with terms of both signs to the correlation, therefore resulting in values centered around 0;  
- $-1 \leq \gamma -\frac{1}{3} \to$ *reciprocal sign epistasis*, which would imply a negative correlation between fitness effects, and therefore a negative value of $\gamma$:  

!!! note
    The calculation involves iterating over pairs of positions and mutations to compare fitness effects across various genetic backgrounds. This process can be computationally intensive for landscapes with a large number of variables or many possible alleles at each position. Utilizing the `n_jobs` parameter for parallel processing is recommended for larger landscapes to improve performance.

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   The fitness landscape object containing fitness data. It must be an initialized and built `Landscape` instance.

    **n_jobs** : `int`, optional, default=`-1`
    :   Number of parallel jobs to use for the computation. A value of -1 utilizes all available CPU cores, which can significantly speed up calculations on multi-core machines. A value of 1 uses a single core (no parallelization).

!!! api-returns "Returns"

    **float**
    :   The gamma statistic value ($\gamma$). Interpretations can be found in the description above.

**References**

- Luca Ferretti *et al.*, "Measuring epistasis in fitness landscapes: The correlation of fitness effects of mutations." *J. Theor. Biol.* (2016) 

**Example**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import gamma_statistic

# synthetic landscape with moderate epistasis
problem = NK(n=10, k=5)
X, f = problem.get_data()

# construct landscape instance
landscape = BooleanLandscape()
landscape.build_from_data(
    X, f,
    verbose=False
)

# run epistasis analysis
gamma = gamma_statistic(landscape)
print(gamma)
```
```
Output:

0.4714147255013242
```

### Gamma Star Statistic



```api
def graphfla.analysis.gamma_star(landscape, n_jobs=-1) → float
```

Calculates the gamma-star ($\gamma^*$) statistic for a fitness landscape ([Ferretti *et al.*, 2016](https://www.sciencedirect.com/science/article/pii/S0022519316000771)).

**Description**

The $\gamma^*$ is a variant of $\gamma$ that focuses only on sign consistency, ignoring the magnitude of fitness effects. It indicates whether mutations tend to have consistent directional effects across different genetic backgrounds. For detailed information, see the description of `gamma_statistic`.

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   The fitness landscape object containing fitness data. It must be an initialized and built `Landscape` instance.

    **n_jobs** : `int`, optional, default=`-1`
    :   Number of parallel jobs to use for the computation. A value of -1 utilizes all available CPU cores, which can significantly speed up calculations on multi-core machines. A value of 1 uses a single core (no parallelization).

!!! api-returns "Returns"

    **float**
    :   The gamma-star statistic ($\gamma^*$) that only considers sign consistency.

**References**

- Luca Ferretti *et al.*, "Measuring epistasis in fitness landscapes: The correlation of fitness effects of mutations." *J. Theor. Biol.* (2016) 

**Example**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import gamma_star

# synthetic landscape with moderate epistasis
problem = NK(n=10, k=5)
X, f = problem.get_data()

# construct landscape instance
landscape = BooleanLandscape()
landscape.build_from_data(
    X, f,
    verbose=False
)

# run epistasis analysis
gs = gamma_star(landscape)
print(gs)
```
```
Output:

0.6516927083333334
```

## Idiosyncratic Epistasis

### Per-mutation Index


```api
def graphfla.analysis.idiosyncratic_index(landscape, mutation: tuple) → float
```

Calculates the idiosyncratic index for a *single specified mutation* in the landscape.

**Description**

The idiosyncratic index of a specific genetic mutation quantifies how sensitive the mutation's fitness effect is to the genetic background. It is defined as the *variation* in the fitness difference between genotypes that differ only by this mutation, divided by the variation in fitness differences between *random* genotype pairs (the same number of pairs).

The index ranges from 0 (minimum idiosyncrasy — the mutation's effect is essentially constant across backgrounds) to 1 (maximum idiosyncrasy — the effect depends strongly on what other mutations are present).

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   The fitness landscape object.

    **mutation** : `tuple(A, pos, B)`
    :   A tuple specifying the mutation:

        - `A`: original allele at the position.
        - `pos`: the position (column name) where the mutation occurs.
        - `B`: the new allele after the mutation.

!!! api-returns "Returns"

    **float**
    :   The idiosyncratic index for this mutation, between 0.0 and 1.0. Returns 0.0 if there are insufficient matched backgrounds.

**References**

-   Daniel M Lyons *et al.*, "Idiosyncratic epistasis creates universals in mutational effects and evolutionary trajectories." *Nat. Ecol. Evol.* (2020).

### Global (Landscape-Wide) Index


```api
def graphfla.analysis.global_idiosyncratic_index(landscape, n_jobs=-1, random_seed=None) → float
```

Calculates the global idiosyncratic index for the entire fitness landscape.

**Description**

In contrast to global epistasis, which describes a *general*, *systematic* trend where a mutation's effect correlates with overall background fitness, idiosyncratic epistasis refers to interactions where the fitness effect of a mutation is highly dependent on the *specific* other mutations present in the genetic background.  Idiosyncratic epistasis emphasizes that these effects arise from particular, often unique, biological and physical interactions between specific loci. The "idiosyncrasy" means that the effect of a given mutation can vary substantially, and sometimes unpredictably, when combined with different sets of other mutations.

This function extends the concept of an idiosyncratic index for individual mutations, as proposed by [Lyons et al. (2020)](https://www.nature.com/articles/s41559-020-01286-y), to a global measure for the entire landscape. It achieves this by averaging the idiosyncratic index across all possible single mutations within the landscape. The global idiosyncratic index quantifies the overall sensitivity of the landscape to idiosyncratic epistasis.

The index is defined as the variation in the fitness effect of a mutation across different genetic backgrounds, relative to the variation in fitness differences between random genotypes. It ranges from 0 to 1, where 0 indicates minimal idiosyncratic epistasis (the effect of a mutation is consistent across backgrounds) and 1 indicates maximum idiosyncratic epistasis (the effect of a mutation is highly variable and background-dependent).

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   The fitness landscape object.

    **n_jobs** : `int`, optional, default=`-1`
    :   Number of parallel workers used to evaluate `idiosyncratic_index` over all candidate mutations. `-1` uses all available cores.

    **random_seed** : `int`, optional, default=`None`
    :   A seed for the random number generator used internally to sample random fitness differences when computing each per-mutation index. Setting a seed ensures reproducibility.

!!! api-returns "Returns"

    **float**
    :   The overall idiosyncratic index for the landscape, calculated as the mean of the per-mutation indices over all valid mutations. Returns `np.nan` if no valid indices can be computed.

**References**

-   Daniel M Lyons *et al.*, "Idiosyncratic epistasis creates universals in mutational effects and evolutionary trajectories." *Nat. Ecol. Evol.* (2020).
-   Christopher W Bakerlee, "Idiosyncratic epistasis leads to global fitness-correlated trends." *Science.* (2022).

**Example**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import global_idiosyncratic_index

# synthetic landscape with moderate epistasis
problem = NK(n=10, k=5)
X, f = problem.get_data()

# construct landscape instance
landscape = BooleanLandscape()
landscape.build_from_data(
    X, f,
    verbose=False
)

# run epistasis analysis
idiosy = global_idiosyncratic_index(landscape)
print(idiosy)
```
```
Output:

0.7693693513651604
```

## Extradimensional Bypass Analysis


```api
def graphfla.analysis.extradimensional_bypass_analysis(landscape, approximate=False, sample_cut_prob=0.2) → dict
```

Detects extradimensional bypasses around reciprocal-sign-epistasis (RSE) motifs.

**Description**

Reciprocal sign epistasis occurs when the wildtype (`ab`) and the double mutant (`AB`) are both fitter than the two intermediate single mutants (`aB`, `Ab`). This creates a fitness valley along the direct path from `ab` to `AB` that natural selection cannot cross. However, when the landscape has more than two relevant dimensions, the population can sometimes traverse a *third-position* intermediate (i.e., gain and later lose an extra mutation) to bypass the valley. Such indirect uphill paths are called **extradimensional bypasses** and they greatly affect a landscape's effective navigability.

For each RSE motif in the landscape (igraph isomorphism class 19), this function checks whether the directed improving graph admits any path from `ab` to `AB`. The fraction of RSE motifs that *do* admit such a bypass is the **bypass proportion**.

!!! api-parameters "Parameters"

    **landscape** : `Landscape`
    :   The fitness landscape object.

    **approximate** : `bool`, default=`False`
    :   If `True`, motif counts are estimated by sampling rather than exact enumeration. Recommended for landscapes with more than ~10,000 mutants.

    **sample_cut_prob** : `float`, default=`0.2`
    :   Pruning probability for the motif sampler, in `[0, 1]`. Only used when `approximate=True`. Larger values mean faster but less accurate sampling.

!!! api-returns "Returns"

    **dict**
    :   A dictionary with the following keys:

        -   `"bypass_proportion"` (*float*): fraction of RSE motifs with at least one extradimensional bypass.
        -   `"average_bypass_length"` (*float*): average length of bypass paths (across motifs that have one). `nan` if no bypasses are found.
        -   `"total_motifs"` (*int*): total number of RSE motifs analyzed.
        -   `"motifs_with_bypass"` (*int*): motifs that had a bypass.

**References**

-   Frank J. Poelwijk *et al.*, "Empirical fitness landscapes reveal accessible evolutionary paths." *Nature* (2007).
-   Daniel M. Weinreich *et al.*, "Sign epistasis and the evolution of complex adaptations." *Annu. Rev. Ecol. Evol. Syst.* (2013).

**Example**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import extradimensional_bypass_analysis

problem = NK(n=10, k=5)
X, f = problem.get_data()

landscape = BooleanLandscape()
landscape.build_from_data(X, f, verbose=False)

result = extradimensional_bypass_analysis(landscape, approximate=True)
print(result)
```

## Examples

```python

import pandas as pd
from graphfla.landscape import BooleanLandscape
from graphfla.problems import NK
from graphfla.analysis import (
    classify_epistasis,
    higher_order_epistasis,
    diminishing_returns_index,
    increasing_costs_index,
    gamma_statistic,
    gamma_star,
    global_idiosyncratic_index,
)

# demo landscape construction
problem = NK(4, 1)
X, fitness = problem.get_data()

landscape = BooleanLandscape()
landscape = landscape.build_from_data(X, fitness, verbose=False)

# types of epistasis
>>> classify_epistasis(landscape, approximate=True, sample_cut_prob=0.1)
{'magnitude epistasis': 0.908663967611336,
 'sign epistasis': 0.07562753036437248,
 'reciprocal sign epistasis': 0.0157085020242915,
 'positive epistasis': 0.36534343600724756,
 'negative epistasis': 0.6346565639927524}

# higher-order epistasis
>>> higher_order_epistasis(landscape, order=3)
0.7116261680761315

# global epistasis
>>> diminishing_returns_index(landscape, method="spearman")
-0.6735265536401572
>>> increasing_costs_index(landscape, method="spearman")
0.5595161241004356

# gamma statistics
>>> gamma_statistic(landscape)
0.4714147255013242
>>> gamma_star(landscape)
0.6516927083333334

# idiosyncratic epistasis
>>> global_idiosyncratic_index(landscape)
0.7693693513651604
```
