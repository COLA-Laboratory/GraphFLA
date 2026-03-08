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

**GraphFLA** (Graph-based Fitness Landscape Analysis) is a Python framework for constructing, analyzing, manipulating and visualizing **fitness landscapes** as graphs. It provides a broad collection of features rooted in evolutoinary biology to decipher the topography of complex fitness landscapes of diverse modalities.

This is also the official code & data repository for the **NeurIPS 2025 (Spotlight)** paper "Augmenting Biological Fitness Prediction Benchmarks with Landscapes Features from GraphFLA". 

Feel free to explore examples in Google Colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zRsU6V0iNucXmeSXqRtwnbaipWfxFGKA?usp=sharing)

## Key Features
- **Versatility:** applicable to arbitrary discrete, combinatorial sequence-fitness data, ranging from biomolecules like DNA, RNA, and protein, to functional units like genes, to complex ecological communities.
- **Comprehensiveness:** offers a holistic collection of 20+ features for characterizing 4 fundamental topographical aspects of fitness landscape, including ruggedness, navigability, epistassi and neutrality.
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

Here, assume we are working with DNA sequences. `GraphFLA` provides registered methods for performance optimization for this type, which can be triggered by specifying `type="dna"`. Alternatively, you can directly use the `DNALandscape` class to get the same effect, which is natively built for DNA data.

The `maximize` parameter specifies the direction of optimization, i.e., whether `f` is to be optimized or minimized.

```python
from graphfla.landscape import DNALandscape

# initialize the landscape
landscape = DNALandscape(maximize=True)

# build the landscape with our data
landscape.build_from_data(X, f, verbose=True)
```

### 4. Landscape analysis

Once the landscape is constructed, we can then analyze its features using the available functions (see later).

```python
from graphfla.analysis import (
    lo_ratio,
    classify_epistasis,
    r_s_ratio,
    neutrality,
    global_optima_accessibility,
)

local_optima_ratio = lo_ratio(landscape)
epistasis = classify_epistasis(landscape)
r_s_score = r_s_ratio(landscape)
neutrality_index = neutrality(landscape)
go_access = global_optima_accessibility(landscape)
```
### 5. Playing with arbitrary combinatorial data
The `type` parameter of the `Landscape` class currently supports `"dna"`, `rna`, `"protein"`, and `"boolean"`. However, this does not mean that `GraphFLA` can only work with these types of data; instead, these registered values are only for convenience and performance optimization purpose. 

In fact, `GraphFLA` can handle arbitrary combinatorial search space as long as the values of each variable is discrete. To work with such data, we can initialize a general landscape, and then pass in a dictionary to specify the data type of each variable (options: `{"ordinal", "cateogrical", "boolean"}`).

```python
import pandas as pd
from graphfla.landscape import Landscape

complex_data = pd.read_csv("path_to_complex_data.csv")

f = complex_data["fitness"]
# data serving as "X"
complex_search_space = complex_data.drop(columns=["fitness"])

# initialize a general fitness landscape without specifying `type`
landscape = Landscape(maximize=True)

# create a data type dictionary
data_types = {
  "x1": "ordinal",
  "x2": "categorical",
  "x3": "boolean",
  "x4": "categorical"
}

# build the landscape with our data and specified data types
landscape.build_from_data(X, f, data_types=data_types, verbose=True)
```

## Landscape Analysis Features

`GraphFLA` currently supports the following features for landscape analysis. Only landscape-level analysis tools are listed; mutation-specific (e.g., `distribution_fit_effects`, `idiosyncratic_index`) and position-specific (e.g., `single_mutation_effects`) tools are excluded.

| **Class** | **Function** | **Feature** | **Range** | **Higher value indicates** |
|--------------------------|----------------------------------|----------------------------------------|---------------|----------------------------------------|
| **Ruggedness** | `lo_ratio`                       | Fraction of local optima               | [0, 1]        | ↑ more peaks                           |
|                          | `r_s_ratio`                      | Roughness-slope ratio                  | [0, ∞)        | ↑ ruggedness                           |
|                          | `autocorrelation`                | Autocorrelation                        | [-1, 1]       | ↓ ruggedness                           |
|                          | `gradient_intensity`             | Gradient intensity                     | [0, ∞)        | ↑ average fitness change per edge      |
|                          | `neighbor_fit_corr`              | Neighbor-fitness correlation           | [-1, 1]       | ↓ ruggedness                           |
| **Epistasis** | `classify_epistasis`             | Magnitude epistasis                    | [0, 1)        | ↓ evolutionary constraints             |
|                          | `classify_epistasis`             | Sign epistasis                         | [0, 1]        | ↑ evolutionary constraints             |
|                          | `classify_epistasis`             | Reciprocal sign epistasis              | [0, 1]        | ↑ evolutionary constraints             |
|                          | `classify_epistasis`             | Positive epistasis                     | [0, 1]        | ↑ synergistic effects                  |
|                          | `classify_epistasis`             | Negative epistasis                     | [0, 1]        | ↑ antagonistic effects                 |
|                          | `global_idiosyncratic_index`     | Global idiosyncratic index             | [0, 1]        | ↑ specific interactions                |
|                          | `diminishing_returns_index`      | Diminishing return epistasis           | [-1, 1]       | ↓ flat peaks (higher = less diminishing returns) |
|                          | `increasing_costs_index`         | Increasing cost epistasis              | [-1, 1]       | ↑ steep descents                      |
|                          | `higher_order_epistasis`         | Higher-order epistasis (R²)            | [0, 1]        | ↑ higher-order interactions            |
|                          | `gamma_statistic`                | Gamma statistic                        | [-1, 1]       | ↑ epistasis (magnitude)                |
|                          | `gamma_star`                     | Gamma star statistic                   | [-1, 1]       | ↑ sign epistasis consistency           |
|                          | `walsh_hadamard_coefficient`     | Pairwise and higher-order epistasis    | -             | -                                      |
|                          | `extradimensional_bypass_analysis`| Extradimensional bypass proportion    | [0, 1]        | ↑ navigability                         |
| **Navigability** | `fitness_distance_corr`          | Fitness-distance correlation           | [-1, 1]       | ↑ navigation                           |
|                          | `fitness_flattening_index`       | Fitness flattening index               | [-1, 1]       | ↑ flatter around global optimum        |
|                          | `global_optima_accessibility`    | Global optimum accessibility           | [0, 1]        | ↑ access to global peaks               |
|                          | `local_optima_accessibility`    | Local optimum accessibility            | [0, 1]        | ↑ access to specified peak(s)          |
|                          | `basin_fit_corr`                 | Basin-fitness corr. (accessible)       | [-1, 1]       | ↑ access to fitter peaks               |
|                          | `basin_fit_corr`                 | Basin-fitness corr. (greedy)           | [-1, 1]       | ↑ access to fitter peaks               |
|                          | `mean_path_lengths`              | Mean path length to LO(s)              | [0, ∞)        | ↑ distance to reach optima             |
|                          | `mean_path_lengths_go`             | Mean path length to global optimum     | [0, ∞)        | ↑ distance to reach global optimum     |
|                          | `mean_dist_lo`                   | Mean distance to LO(s)                 | [0, ∞)        | ↑ spatial distance to optima           |
|                          | `evol_enhance_mutations`         | Evol-enhancing mutation proportion     | [0, 1]        | ↑ evolvability                         |
| **Neutrality** | `neutrality`                     | Neutrality                             | [0, 1]        | ↑ neutrality                           |
| **Fitness Distribution** | `fitness_distribution`           | Skewness                               | (-∞, ∞)       | ↑ asymmetry of fitness values          |
|                          | `fitness_distribution`           | Kurtosis                               | (-∞, ∞)       | ↑ outlier/extreme value prevalence     |
|                          | `fitness_distribution`           | Coefficient of variation (CV)          | [0, ∞)        | ↑ relative fitness variability         |
|                          | `fitness_distribution`           | Quartile coefficient                   | [0, 1]        | ↑ interquartile dispersion             |
|                          | `fitness_distribution`           | Median/Mean ratio                      | [0, ∞)        | ↑ deviation from symmetry              |
|                          | `fitness_distribution`           | Relative range                         | [0, ∞)        | ↑ spread of fitness values             |
|                          | `fitness_distribution`           | Cauchy location parameter              | (-∞, ∞)       | ↑ central tendency estimate            |


## Landscape Classes

`GraphFLA` currently offers the following classes for landscape construction.

|**Classes**|**Supported search space**|**Description**|
|--|--|--|
|`Landscape`|All discrete, combinatorial spaces, where each variable can be either categorical, boolean, or ordinal|The base landscape class, most generalizable|
|`SequenceLandscape`|Categorical data where each variable takes values from the same alphabet.|Class optimized for general sequence data|
|`BooleanLandscape`|Boolean space|Class optimized for boolean data|
|`DNALandscape`|DNA sequence space|Class optimized for DNA data|
|`RNALandscape`|RNA sequence space|Class optimized for RNA data|
|`ProteinLandscape`|Protein sequence space|Class optimized for protein data|

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
