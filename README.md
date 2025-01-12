# GraphFLA

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
<!-- [![Python Versions](https://img.shields.io/pypi/pyversions/graphfla.svg)](https://pypi.python.org/pypi/graphfla/)
[![Issues](https://img.shields.io/github/issues/yourusername/graphfla.svg)](https://github.com/yourusername/graphfla/issues)
[![Stars](https://img.shields.io/github/stars/yourusername/graphfla.svg)](https://github.com/yourusername/graphfla/stargazers) -->

**GraphFLA** (Graph-based Fitness Landscape Analysis) is a Python framework for constructing, analyzing, and visualizing **fitness landscapes** as graphs. It provides a broad range of visualization tools and quantitative metrics to study how fitness (or performance) changes over a space of configurations (e.g., genetic sequences, hyperparameter settings, etc.).

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [1. Create a `Landscape` object](#1-create-a-landscape-object)
  - [2. Analyze the landscape](#2-analyze-the-landscape)
  - [3. Visualize the landscape](#3-visualize-the-landscape)
- [Pre-collected landscape data](#pre-collected-landscape-data)
    - [1. Artificial landscapes](#1-artificial-landscapes)
    - [2. Biological landscapes](#2-biological-landscapes)
    - [3. Combinatorial optimization landscapes](#3-combinatorial-optimization-landscapes)
    - [4. Hyperparameter optimization landscapes](#4-hyperparameter-optimization-landscapes)
    - [5. Software engineering landscapes](#5-software-engineering-landscapes)
- [API Overview](#api-overview)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features

- **Flexible input**: Accepts a wide range of data types (boolean, categorical, ordinal) for defining configuration spaces.
- **Automated graph construction**: Builds directed graphs representing fitness relationships between configurations.
- **Rich analysis**: Provides numerous methods to analyze ruggedness, epistasis, neutrality, fitness distance correlation (FDC), local optima network (LON) and more.
- **Visualization**: Offers convenient plotting functions for 2D/3D embeddings, distribution plots, epistasis graphs, local neighborhoods, and more.

## Installation

You can install **GraphFLA** via pip (once published on PyPI) or from source:

```bash
# via PyPI (planned)
pip install graphfla

# OR from source
git clone https://github.com/yourusername/graphfla.git
cd graphfla
pip install .
```

**Dependencies**: This framework uses `pandas`, `numpy`, `networkx`, `matplotlib`, `tqdm`, `umap-learn`, `karateclub`, and other common data-science libraries.  
For the full list, see [requirements.txt](./requirements.txt).

## Quick Start

Below is a minimal example of how to use **GraphFLA**. For more detailed examples, please refer to the [examples directory](./examples) (if available) or the [API Overview](#api-overview) section.

```python
import pandas as pd
from graphfla import Landscape

# Prepare data
X = pd.DataFrame({
    "param1": [0, 0, 1, 1, 2],
    "param2": [10, 15, 10, 15, 15],
})
f = [0.5, 0.6, 0.7, 0.55, 0.9]

data_types = {
    "param1": "ordinal",
    "param2": "ordinal"
}

# Create a Landscape object
landscape = Landscape(X, f, maximize=True, data_types=data_types)

# Print a summary
print(landscape)
landscape.describe()
```

## Usage

### 1. Create a `Landscape` object

You can create a `Landscape` from:
- A pandas DataFrame (`X`) of configurations.
- An array or list of fitness values (`f`).
- A dictionary describing variable types (`data_types`).

```python
from graphfla import Landscape

X = pd.DataFrame(...)  # your configurations
f = pd.Series(...)      # your fitness values
data_types = {
    "param1": "ordinal",
    "param2": "categorical",
    # ...
}

landscape = Landscape(X, f, maximize=True, data_types=data_types)
```

### 2. Analyze the landscape

Once the `Landscape` is created, you can:
- Compute various landscape metrics (ruggedness, neutrality, FDC, etc.).
- Identify local optima and basins of attraction.
- Construct the Local Optima Network (LON).

```python
# Basic metrics
fdc_value = landscape.fdc(method="spearman")
ruggedness_index = landscape.ruggedness()
autocorr_mean, autocorr_var = landscape.autocorrelation(walk_length=30, walk_times=500, lag=1)

# Local optima network
lon_graph = landscape.get_lon(mlon=True, min_edge_freq=2)
```

### 3. Visualize the landscape

**GraphFLA** comes with convenient visualization utilities:

```python
# Draw local neighborhood of a given configuration (node)
landscape.draw_neighborhood(node=10, radius=2)

# 2D embedding with contours
landscape.draw_landscape_2d()

# Distribution of fitness values
landscape.draw_fitness_dist(type="hist")

# Epistasis network visualization
epistasis_df = landscape.all_pairwise_epistasis(n_jobs=4)
landscape.draw_epistasis(epistasis_df=epistasis_df)
```

## Pre-collected landscape data

**GraphFLA** also provides pre-collected landscape data for optimization problems from diverse domains, which allows you to explore and analyze these landscapes without the need to generate them from scratch. Currently, **GraphFLA** includes data for the following domains:

### 1. Artificial landscapes

The `problems` module contains various classes for generating artificial landscapes with tunable properties, including the Kauffman's NK model, the Rough Mt. Fuji model, the Eggbox model, etc.

### 2. Biological landscapes

We collected more than 30 combinatorially complete landscapes in the biological domain, including protein binding sites, DNA expression, and RNA, etc. 

### 3. Combinatorial optimization landscapes

The `problems` module also includes combinatorial optimization problems such as the Number Partitioning Problem (NPP), the Quadratic Assignment Problem (QAP), etc.

### 4. Hyperparameter optimization landscapes

This dataset contains the hyperparameter optimization landscapes for various machine learning models, including XGBoost, Random Forest, CNN, etc, on a wide range of datasets, with more than 11 million configurations in total.

### 5. Software engineering landscapes

This dataset contains the performance landscape of configurable software systems, including Apache, LLVM, and SQLite. For each system, we evaluated the performance of over 1 million configurations across various distinct workloads. This then results in a total of 32 configuration landscapes and over 87 million configuration evaluations. 

## API Overview

Below is the central class from this framework:

```python
class Landscape(BaseLandscape):
    """
    Class implementing the fitness landscape object

    Notes
    -----
    In GraphFLA, a fitness landscape is represented as a genotype network in igraph, 
    where each node is a genotype (an entry in X) with fitness values (values in f) and 
    any additional information stored as node attributes. 
    ...
    """
    
    def __init__(
        self,
        X: pd.DataFrame = None,
        f: pd.Series = None,
        graph: nx.DiGraph = None,
        maximize: bool = True,
        epsilon: float = "auto",
        data_types: Dict[str, str] = None,
        verbose: bool = True,
    ) -> None:
        ...
```

The `Landscape` class offers a wide array of methods to:

- **Construct** a directed graph based on fitness comparisons.  
- **Analyze** local optima, basins of attraction, epistatic interactions, etc.  
- **Measure** landscape features such as ruggedness, neutrality, fitness distance correlation, etc.  
- **Visualize** different aspects of the landscape (2D/3D embeddings, epistasis networks, neighborhood graphs, etc.).  

For complete documentation on each method, please see the docstrings in the source code or refer to additional documentation (if provided).

## Contributing

Contributions are welcome! If you find a bug, have an idea for a new feature, or want to improve documentation, please open an issue or submit a pull request. 

### Development Setup

1. Fork the repository and clone it locally.  
2. Install in editable mode with dev dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run the tests:
   ```bash
   pytest tests
   ```

## License

This project is licensed under the terms of the [MIT License](./LICENSE).

## Citation

If you use **GraphFLA** or our pre-collected datasets in your research, please cite this repository and/or the relevant paper(s):

```bibtex
@inproceedings{HuangL23,
  author       = {Mingyu Huang and
                  Ke Li},
  title        = {Exploring Structural Similarity in Fitness Landscapes via Graph Data
                  Mining: {A} Case Study on Number Partitioning Problems},
  booktitle    = {{IJCAI}'24: Proc of the 32nd International Joint Conference on
                  Artificial Intelligence},
  pages        = {5595--5603},
  publisher    = {ijcai.org},
  year         = {2023},
  url          = {https://doi.org/10.24963/ijcai.2023/621},
  doi          = {10.24963/IJCAI.2023/621},
}

@inproceedings{HuangL25a,
  author       = {Mingyu Huang and
                  Ke Li},
  title        = {On the hyperparameter landscapes of machine learning algorithms},
  booktitle    = {{KDD}'25: Proc of the ACM SIGKDD International Conference on Knowledge 
                  Discovery & Data Mining },
  pages        = {in press},
  publisher    = {ACM},
  year         = {2025},
}

@inproceedings{HuangL25b,
  author       = {Mingyu Huang and
                  Peili Mao and
                  Ke Li},
  title        = {Rethinking Performance Analysis for Configurable Software Systems: A  
                  Case Study from a Fitness Landscape Perspective},
  booktitle    = {{ISSTA}'25: Proc. of the ACM SIGSOFT International Symposium on Software 
                  Testing and Analysis},
  pages        = {in press},
  publisher    = {ACM},
  year         = {2025},
}
```

---

**Happy analyzing!** If you have any questions or suggestions, feel free to [open an issue](https://github.com/yourusername/graphfla/issues) or [start a discussion](https://github.com/yourusername/graphfla/discussions).
