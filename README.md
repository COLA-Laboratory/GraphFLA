# GraphFLA

![Alt text](images/landscape.jpg)

<div align="center">
    <a href="https://www.python.org/" rel="nofollow">
        <img src="https://img.shields.io/pypi/pyversions/graphfla" alt="Python" />
    </a>
    <a href="https://pypi.org/project/graphfla/" rel="nofollow">
        <img src="https://img.shields.io/pypi/v/graphfla" alt="PyPI" />
    </a>
    <a href="https://github.com/COLA-Laboratory/GraphFLA/blob/main/LICENSE" rel="nofollow">
        <img src="https://img.shields.io/pypi/l/graphfla" alt="License" />
    </a>
    <a href="https://github.com/COLA-Laboratory/GraphFLA/actions/workflows/test.yml" rel="nofollow">
        <img src="https://github.com/COLA-Laboratory/GraphFLA/actions/workflows/test.yml/badge.svg" alt="Test" />
    </a>
    <a href="https://github.com/psf/black" rel="nofollow">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
</div>
<br>

**GraphFLA** (Graph-based Fitness Landscape Analysis) is a Python framework for constructing, analyzing, manipulating and visualizing **fitness landscapes** as graphs. It provides a broad collection of features rooted in evolutionary biology to decipher the topography of complex fitness landscapes of diverse modalities.

This is also the official code & data repository for the **NeurIPS 2025 (Spotlight)** paper "Augmenting Biological Fitness Prediction Benchmarks with Landscapes Features from GraphFLA". 

Feel free to explore examples in Google Colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zRsU6V0iNucXmeSXqRtwnbaipWfxFGKA?usp=sharing)

## Key Features
- **Versatility:** applicable to arbitrary discrete, combinatorial sequence-fitness data, ranging from biomolecules like DNA, RNA, and protein, to functional units like genes, to complex ecological communities.
- **Comprehensiveness:** offers a holistic collection of 20+ features for characterizing 4 fundamental topographical aspects of fitness landscape, including ruggedness, navigability, epistasis and neutrality.
- **Interoperability:** works with the same data format (i.e., `X` and `f`) as in training machine learning (ML) models, thus being interoperable with established ML ecosystems in different disciplines.
- **Scalability:** heavily optimized to be capable of handling landscapes with even millions of variants.
- **Extensibility:** new landscape features can be easily added via an unified API.

## Quick Start

Our documentation website is currently under development, but `GraphFLA` is quite easy to get started with!

### 1. Installation

Official installation (pip)

```
pip install graphfla
```

### 2. Prepare your data

`GraphFLA` is designed to interoperate with established ML frameworks and benchmarks by using the same data format as in ML model training: an `X` and an `f`. 

Specifically, `X` can either be a list of sequences of strings representing genotypes, or a `pd.DataFrame` or an `numpy.ndarray`, wherein each column represents a loci; `f` can either be a list, `pd.Series` or `numpy.ndarray`.

To make landscape construction faster, we recommended removing redundant loci in `X` (i.e., those that are never mutated across the whole library) .

```python
import pandas as pd

# Load data:
data = pd.DataFrame({
    "sequences": ["AAA", "AAG", "AGA", "AGG", "GAA", "GAG", "GGA", "GGG"],
    "fitness": [0.10, 0.25, 0.25, 0.40, 0.25, 0.40, 0.40, 0.91]
})
# 3 positions (A/G), 8 variants; all connected via single mutations; unimodal (GGG optimum)

X = data["sequences"]
f = data["fitness"]
```

### 3. Create the landscape object

Creating a landscape object in `GraphFLA` is much like training an ML model: we first initialize a `Landscape` class, and then build it with our data. 

Here, assume we are working with DNA sequences. `GraphFLA` provides registered methods for performance optimization for this type, which can be triggered by specifying `kind="dna"`. Alternatively, you can directly use the `DNALandscape` class to get the same effect, which is natively built for DNA data.

The `maximize` parameter specifies the direction of optimization, i.e., whether `f` is to be optimized or minimized.

```python
from graphfla.landscape import DNALandscape

# initialize the landscape
landscape = DNALandscape(maximize=True)

# build the landscape with our data
landscape.build_from_data(X, f, verbose=True)
```

### 4. Landscape analysis

The quickest way to characterize a built landscape is `analysis.profile()`: it computes the whole portfolio of landscape-level metrics in one call and returns a tidy `pandas` object — a `Series` for one landscape, or a `DataFrame` (one row each) for several, which is ideal for comparing landscapes.

```python
from graphfla import analysis

# every whole-landscape metric, as a pandas Series
metrics = analysis.profile(landscape)
print(metrics["gamma"], metrics["fdc"], metrics["epistasis.magnitude"])

# restrict to groups (or `include=[...]` specific metrics, `exclude=[...]` to drop some)
analysis.profile(landscape, groups=["ruggedness", "epistasis"])

# compare several landscapes side by side -> DataFrame, one row each.
# here, a panel of NK landscapes with increasing ruggedness (k):
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape

panel = [BooleanLandscape().build_from_data(*NK(n=8, k=k, seed=0).get_data(), verbose=False)
         for k in (0, 2, 4)]
analysis.profile(panel, index=["NK k=0", "NK k=2", "NK k=4"])

analysis.list_metrics()   # discover what's available
```

The expensive motif metrics (`classify_epistasis`, `extradimensional_bypass`) auto-tune their sampling to a time budget, so `profile()` stays tractable even on large landscapes such as GB1 or DHFR. You can also call any metric on its own:

```python
from graphfla.analysis import local_optima_ratio, classify_epistasis, neutrality

local_optima_ratio(landscape)
classify_epistasis(landscape)      # sample_cut_prob="auto" by default
neutrality(landscape)
```
### 5. Playing with arbitrary combinatorial data
The `kind` parameter of the `Landscape` class currently supports `"boolean"`, `"ordinal"`, `"dna"`, `"rna"`, and `"protein"`. However, this does not mean that `GraphFLA` can only work with these types of data; instead, these registered values are only for convenience and performance optimization purpose. 

In fact, `GraphFLA` can handle arbitrary combinatorial search space as long as the values of each variable is discrete. To work with such data, we can initialize a general landscape, and then pass in a dictionary to specify the data type of each variable (options: `{"ordinal", "categorical", "boolean"}`).

```python
import pandas as pd
from graphfla.landscape import Landscape

complex_data = pd.read_csv("path_to_complex_data.csv")

f = complex_data["fitness"]
# data serving as "X"
complex_search_space = complex_data.drop(columns=["fitness"])

# initialize a general fitness landscape without specifying `kind`
landscape = Landscape(maximize=True)

# create a data type dictionary
data_types = {
  "x1": "ordinal",
  "x2": "categorical",
  "x3": "boolean",
  "x4": "categorical"
}

# build the landscape with our data and specified data types
landscape.build_from_data(complex_search_space, f, data_types=data_types, verbose=True)
```

## Landscape Analysis Features

`GraphFLA` ships 20+ landscape-level metrics spanning the major aspects of landscape topography. Grab the whole portfolio in one call with `analysis.profile()`, restrict it with `profile(..., groups=[...])` (the group tokens are the section headers below), or call any function on its own. The collapsible tables below catalog every landscape-level metric — expand the aspect you care about.

> Mutation- and position-specific tools (`fitness_effect_distribution`, `idiosyncratic_index`, `single_mutation_effects`) characterize a single element rather than the whole landscape and are not listed here.

<details>
<summary><b>Ruggedness</b> · <code>groups="ruggedness"</code> — multimodality and local structure</summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `local_optima_ratio` | Fraction of configurations that are local optima | [0, 1] | more peaks |
| `r_s_ratio` | Roughness-to-slope ratio | [0, ∞) | more rugged |
| `autocorrelation` | Autocorrelation of fitness along random walks | [-1, 1] | less rugged |
| `gradient_intensity` | Mean absolute fitness change per edge | [0, ∞) | steeper gradients |

</details>

<details>
<summary><b>Epistasis</b> · <code>groups="epistasis"</code> — interactions between mutations</summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `gamma` | Gamma statistic — overall magnitude of epistasis | [-1, 1] | more epistasis |
| `gamma_star` | Gamma-star — consistency of sign epistasis | [-1, 1] | more consistent sign epistasis |
| `classify_epistasis` | Fraction of pairwise interactions of each type: magnitude, sign, reciprocal-sign, positive, negative | [0, 1] | — (composition) |
| `higher_order_epistasis` | Variance fraction (R²) from interactions beyond pairwise | [0, 1] | more higher-order interactions |
| `global_idiosyncratic_index` | How context-dependent (idiosyncratic) mutation effects are | [0, 1] | more idiosyncratic |
| `diminishing_returns_index` | Diminishing-returns epistasis (background fitness vs. gains) | [-1, 1] | weaker diminishing returns |
| `increasing_costs_index` | Increasing-costs epistasis (background fitness vs. costs) | [-1, 1] | stronger increasing costs |
| `extradimensional_bypass` | Reciprocal-sign motifs bypassed via extra dimensions (proportion, avg. length) | [0, 1] | more bypasses → more navigable |
| `walsh_hadamard` † | Walsh–Hadamard epistasis spectrum (pairwise and higher-order coefficients) | — | returns a table |

</details>

<details>
<summary><b>Navigability</b> · <code>groups="navigability"</code> — reachability of optima</summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `global_optima_accessibility` | Fraction of configs on a fitness-monotone path to the global optimum | [0, 1] | more accessible |
| `mean_path_length_to_global_optimum` | Mean shortest adaptive-walk length to the global optimum | [0, ∞) | farther to reach |
| `mean_distance_to_global_optimum` | Mean Hamming distance to the global optimum | [0, ∞) | more spread out |
| `local_optima_accessibility` † | Accessibility of one or more specified local optima | [0, 1] | more accessible |
| `mean_path_length_to_local_optima` † | Mean adaptive-walk length to specified local optima | [0, ∞) | farther to reach |
| `mean_distance_to_local_optima` † | Mean Hamming distance to specified local optima | [0, ∞) | more spread out |

</details>

<details>
<summary><b>Correlation</b> · <code>groups="correlation"</code> — fitness–distance and basin structure</summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `fdc` | Fitness–distance correlation to the global optimum | [-1, 1] | more navigable |
| `neighbor_fitness_correlation` | Correlation of a config's fitness with its neighbors' mean | [-1, 1] | less rugged |
| `basin_fitness_correlation` | Correlation between basin size and local-optimum fitness | [-1, 1] | fitter peaks have larger basins |
| `fitness_flattening_index` | Whether fitness flattens approaching the global optimum | [-1, 1] | flatter near the peak |

</details>

<details>
<summary><b>Robustness</b> · <code>groups="robustness"</code> — neutrality and evolvability</summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `neutrality` | Fraction of neutral (equal-fitness) edges | [0, 1] | more neutral |
| `evolvability_enhancing_mutations` | Fraction of mutations that open access to fitter regions | [0, 1] | more evolvable |

</details>

<details>
<summary><b>Fitness distribution</b> · <code>groups="fitness"</code> — shape statistics</summary>

| Function | Measures | Range |
|---|---|---|
| `fitness_distribution` | Unitless shape of the fitness distribution: skewness, kurtosis, coefficient of variation, quartile coefficient, median/mean ratio, relative range, Cauchy location | various |

</details>

<sub>† Not part of the default `profile()` portfolio — call directly. These require a focal optimum (`lo=...`) or return a variable-length table rather than a single value.</sub>

## Landscape Classes

<details>
<summary><b>Seven landscape classes</b> — pick the one matching your data (all share the same <code>build_from_data</code> API)</summary>

| Class | Search space | Notes |
|---|---|---|
| `Landscape` | Any discrete combinatorial space — categorical, boolean, and/or ordinal columns (possibly mixed) | Base class, most general; pass `data_types=` for mixed columns |
| `SequenceLandscape` | Categorical sequences over a shared alphabet | General sequence data |
| `BooleanLandscape` | Boolean (binary) space | Optimized for bit-strings |
| `OrdinalLandscape` | Ordinal variables (ordered levels) | Optimized for ordinal data |
| `DNALandscape` | DNA sequences (A/C/G/T) | Optimized for DNA |
| `RNALandscape` | RNA sequences (A/C/G/U) | Optimized for RNA |
| `ProteinLandscape` | Protein sequences (20 amino acids) | Optimized for protein |

</details>

## Synthetic Problem Generators

GraphFLA provides synthetic fitness landscape generators for benchmarking and experimentation. All generators produce binary (boolean) search spaces. Use `get_data()` to obtain configurations and fitness values in the same format as ML training data:

```python
from graphfla.problems import NK, RoughMountFuji, Additive, Eggbox, HoC
from graphfla.problems import Max3Sat, Knapsack, NumberPartitioning

# Example: NK landscape
problem = NK(n=10, k=1)
X, f = problem.get_data()
# X: list of binary strings, e.g. ["0000000000", "0000000001", "0000000010", ...]
# f: list of fitness values

# Build landscape and analyze
from graphfla.landscape import BooleanLandscape
landscape = BooleanLandscape()
landscape.build_from_data(X, f, verbose=False)
```

Available generators: `NK`, `RoughMountFuji`, `Additive`, `Eggbox`, `HoC` (biological models); `Max3Sat`, `Knapsack`, `NumberPartitioning` (combinatorial). Warning: `get_data()` evaluates all 2^n configurations—use with caution for large n.

## License

This project is licensed under the terms of the [MIT License](./LICENSE).

## Credit

You may cite GraphFLA via the following references:

```
@inproceedings{HuangZL25,
  author    = {Mingyu Huang and
               Shasha Zhou and
               Ke Li},
  title     = {Augmenting Biological Fitness Prediction Benchmarks with Landscape Features
               from GraphFLA},
  booktitle = {{NeurIPS}'25: Proc. of Advances in Neural Information Processing Systems 39},
  year      = {2024},
}
```

```
@article{HuangML25,
  author       = {Mingyu Huang and
                  Peili Mao and
                  Ke Li},
  title        = {Rethinking Performance Analysis for Configurable Software Systems:
                  {A} Case Study from a Fitness Landscape Perspective},
  journal      = {Proc. {ACM} Softw. Eng.},
  volume       = {2},
  number       = {{ISSTA}},
  pages        = {1748--1771},
  year         = {2025},
  doi          = {10.1145/3728954},
}
```

```
@inproceedings{HuangL25,
  author       = {Mingyu Huang and
                  Ke Li},
  title        = {On the Hyperparameter Loss Landscapes of Machine Learning Models:
                  An Exploratory Study},
  booktitle    = {{KDD}'25: Proceedings of the 31st {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining},
  pages        = {555--564},
  publisher    = {{ACM}},
  year         = {2025},
  doi          = {10.1145/3690624.3709229},
}
```

---

**Happy analyzing!** If you have any questions or suggestions, feel free to open an issue or start a discussion.
