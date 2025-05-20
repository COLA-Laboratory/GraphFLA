# GraphFLA

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
<!-- [![Python Versions](https://img.shields.io/pypi/pyversions/graphfla.svg)](https://pypi.python.org/pypi/graphfla/)
[![Issues](https://img.shields.io/github/issues/yourusername/graphfla.svg)](https://github.com/yourusername/graphfla/issues)
[![Stars](https://img.shields.io/github/stars/yourusername/graphfla.svg)](https://github.com/yourusername/graphfla/stargazers) -->

![Alt text](images/landscape.jpg)

**GraphFLA** (Graph-based Fitness Landscape Analysis) is a Python framework for constructing, analyzing, manipulating and visualizing **fitness landscapes** as graphs. It provides a broad collection of features rooted in evolutoinary biology to decipher the topography of compelx fitness landscapes of diverse modalities.

## Key Features
- **Versatility:** applicable to arbitrary discrete, combinatorial sequence-fitness data, ranging from biomolecules like DNA, RNA, and protein, to functional units like genes, to complex ecological communities.
- **Comprehensiveness:** offers a holistic collection of 20+ features for characterizing 4 fundamental topographical aspects of fitness landscape, including ruggedness, navigability, epistassi and neutrality.
- **Interoperability:** works with the same data format (i.e., `X` and `f`) as in training machine learning (ML) models, thus interoperable with established ML ecosystems in different disciplines.
- **Scalability:** heavily optimized to be capable of handling landscapes with even millions of variants.
- **Extensibility:** new landscape features can be easily added via an unified API. 


## Quick Start

Our documentation website is currently under development, but `GraphFLA` is quite easy to get started with!

### 1. Prepare your data

`GraphFLA` is designed to interoperate with established ML frameworks and benchmarks by using the same data format as in ML model training: an `X` and an `f`. 

Specifically, `X` can either be a list of sequences of strings representing genotypes, or a `pd.DataFrame` or an `numpy.ndarray`, wherein each column represents a loci; `f` can either be a list, `pd.Series` or `numpy.ndarray`.

To make landscape construction faster, we recommended removing redundant loci in `X` (i.e., those that are never mutated across the whole library) .

```python
import pandas as pd

data = pd.read_csv("path_to_data.csv")

X = data["sequences"]
f = data["fitness"]
```

### 2. Create the landscape object

Creating a landscape object in `GraphFLA` is much like training an ML model: we first initialize a `Landscape` class, and then build it with our data. 

Here, assume we are working with DNA sequences. `GraphFLA` provides registered methods for performance optimization for this type, which can be triggered by specifying `type="dna"`. Alternatively, you can directly use the `DNALandscape` class to get the same effect, which is natively built for DNA data.

The `maximize` parameter specifies the direction of optimization, i.e., whether `f` is to be optimized or minimized.

```python
from graphfla.landscape import Landscape

# initialize the landscape
# this is equivalent to:
# from graphfla.landscape import DNALandscape
# landscape = DNALandscape(maximize=True)
landscape = Landscape(type="dna", maximize=True)

# build the landscape with our data
landscape.build_from_data(X, f, verbose=True)
```

### 3. Landscape analysis

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
### 4. Playing with arbitrary combinatorial data
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

`GraphFLA` currently supports the following features for landscape analysis.

| **Class** | **Function** | **Feature** | **Range** | **Higher value indicates** |
|--------------------------|----------------------------------|----------------------------------------|---------------|----------------------------------------|
| **Ruggedness** | `lo_ratio`                       | Fraction of local optima               | [0,1]         | ↑ more peaks                           |
|                          | `r_s_ratio`                      | Roughness-slope ratio                  | [0, ∞)        | ↑ ruggedness                           |
|                          | `autocorrelation`                | Autocorrelation                        | [-1, 1]       | ↓ ruggedness                           |
|                          | `gamma_statistic`                | Gamma statistic                        | [-1, 1]       | ↑ ruggedness                           |
|                          | `gamma_statistic`                | Gamma star statistic                   | [-1, 1]       | ↑ ruggedness                           |
|                          | `neighbor_fit_corr`              | Neighbor-fitness correlation           | [-1, 1]       | ↓ ruggedness                           |
| **Epistasis** | `classify_epistasis`             | Magnitude epistasis                    | [0, 1)        | ↓ evolutionary constraints             |
|                          | `classify_epistasis`             | Sign epistasis                         | [0, 1]        | ↑ evolutionary constraints             |
|                          | `classify_epistasis`             | Reciprocal sign epistasis              | [0, 1]        | ↑ evolutionary constraints             |
|                          | `classify_epistasis`             | Positive epistasis                     | [0, 1]        | ↑ synergistic effects                  |
|                          | `classify_epistasis`             | Negative epistasis                     | [0, 1]        | ↑ antagonistic effects                 |
|                          | `global_idiosyncratic_index`     | Global idiosyncratic index             | [0, 1]        | ↑ specific interactions                |
|                          | `diminishing_returns_index`      | Diminishing return epistasis           | [0, 1]        | ↑ flat peaks                           |
|                          | `increasing_costs_index`         | Increasing cost epistasis              | [0, 1]        | ↑ steep descents                       |
|                          | `higher_order_epistasis`         | Higher-order epistasis                 | [0, 1]        | ↓ higher-order interactions            |
| **Navigability** | `fitness_distance_corr`          | Fitness-distance correlation           | [-1, 1]       | ↑ navigation                           |
|                          | `go_accessibility`               | Global optima accessibility            | [0, 1]        | ↑ access to global peaks               |
|                          | `basin_fit_corr`                 | Basin-fitness corr. (accessible)       | [-1, 1]       | ↑ access to fitter peaks               |
|                          | `basin_fit_corr`                 | Basin-fitness corr. (greedy)           | [-1, 1]       | ↑ access to fitter peaks               |
|                          | `calculate_evol_enhance`         | Evol-enhancing mutation                | [0, 1]        | ↑ evolvability                         |
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

## License

This project is licensed under the terms of the [MIT License](./LICENSE).

---

**Happy analyzing!** If you have any questions or suggestions, feel free to open an issue or start a discussion.
