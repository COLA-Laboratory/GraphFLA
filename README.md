# GraphFLA

![GraphFLA](images/landscape.jpg)

<div align="center">
    <a href="https://colalab.ai/GraphFLA/" rel="nofollow">
        <img src="https://img.shields.io/badge/website-GraphFLA-ffd60a" alt="Website" />
    </a>
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

This is also the official code and data repository for the **NeurIPS 2025 (Spotlight)** paper "Augmenting Biological Fitness Prediction Benchmarks with Landscape Features from GraphFLA".

Full documentation, including the API reference, is at [colalab.ai/GraphFLA](https://colalab.ai/GraphFLA/). Every [tutorial](#tutorials) also runs in Google Colab.

## Key Features
- **Versatility:** works on any discrete, combinatorial sequence-fitness data, from DNA, RNA and proteins to genes and ecological communities.
- **Comprehensiveness:** 20+ metrics covering ruggedness, epistasis, navigability and neutrality.
- **Interoperability:** takes the same `X` and `f` used to train machine learning models.
- **Scalability:** handles landscapes with millions of variants.
- **Extensibility:** new metrics plug into a unified API.

## Quick Start

### 1. Install

```bash
pip install graphfla
```

### 2. Build a landscape

GraphFLA takes the same input as a machine learning model: variants `X` and their fitness `f`. `X` can be a list of sequences, or a `pandas.DataFrame` or `numpy.ndarray` with one column per position; `f` can be a list, `pandas.Series` or `numpy.ndarray`.

Choose the class that matches your data:

| Data | Class |
|---|---|
| DNA, RNA or protein sequences | `DNALandscape`, `RNALandscape`, `ProteinLandscape` |
| Sequences over another alphabet | `SequenceLandscape` |
| Binary variables (on/off, present/absent) | `BooleanLandscape` |
| Ordered levels (doses, temperatures) | `OrdinalLandscape` |
| A mix of the above | `Landscape` (see [step 4](#4-mixed-variable-types)) |

```python
from graphfla.landscape import DNALandscape

# 8 variants over 3 positions, with a single peak at GGG
X = ["AAA", "AAG", "AGA", "AGG", "GAA", "GAG", "GGA", "GGG"]
f = [0.10, 0.25, 0.25, 0.40, 0.25, 0.40, 0.40, 0.91]

landscape = DNALandscape(maximize=True)  # maximize=False if lower f is fitter
landscape.build_from_data(X, f)
```

### 3. Analyze it

`analysis.profile()` computes every landscape-level metric in one call and returns a `pandas.Series`. Given a list of landscapes, it returns a `DataFrame` with one row each. Sampling-based metrics such as `classify_epistasis` adapt to a time budget, so profiling stays tractable on large landscapes such as GB1 or DHFR.

```python
from graphfla import analysis

metrics = analysis.profile(landscape, seed=42)
metrics[["gamma", "fdc", "epistasis.magnitude"]]

analysis.profile(landscape, metrics=["ruggedness", "epistasis"])  # selected groups
analysis.local_optima_ratio(landscape)                            # a single metric
analysis.list_metrics()                                           # all available metrics
```

### 4. Mixed variable types

When columns differ in type, for example a categorical solvent alongside an ordinal temperature, use the general `Landscape` class and declare each column as `"categorical"`, `"ordinal"` or `"boolean"`.

```python
import pandas as pd
from graphfla.landscape import Landscape

df = pd.read_csv("reactions.csv")
X = df[["solvent", "catalyst", "temperature"]]  # Reaction conditions
f = df["yield"]                                 # Product yield, to maximize

landscape = Landscape(maximize=True)
landscape.build_from_data(
    X, f, data_types={"solvent": "categorical", "catalyst": "categorical", "temperature": "ordinal"}
)
```

## Tutorials

These nine notebooks include worked code and saved outputs, with their [input data](tutorials/datasets/data/) alongside them. Each one also opens in Google Colab, where it installs GraphFLA and downloads its data.

| Dataset | Application | Colab |
|---|---|---|
| [Suzuki-Miyaura](tutorials/datasets/04_suzuki_landscape.ipynb) | Chemical reaction conditions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17ef8wR3DNM8URikhJm73YeziTgoqTeVQ) |
| [Flow semihydrogenation](tutorials/datasets/05_flow_semihydrogenation_landscape.ipynb) | Electrochemical process settings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YpJbji2XZU_lNHMSkxYQj7Yp7om8tiZM) |
| [W-Re-Os alloys](tutorials/datasets/06_alloy_landscape.ipynb) | Alloy composition | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q8ieoHAUOKkCfjSyYGwFCCyvZBx9zCq-) |
| [Perovskites](tutorials/datasets/07_perovskite_landscape.ipynb) | Material constituent choices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rlJXgnu62rxqt-1840Uf8FztUhlyqOjm) |
| [BacPUS](tutorials/datasets/08_microbiome_landscape.ipynb) | Bacterial strain-substrate combinations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ujin6GvhBEpSsN6XzF8INGO0GIVdAMTc) |
| [Cyanimide](tutorials/datasets/09_cyanimide_landscape.ipynb) | Chemical building blocks and enzyme inhibition | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SY-iF_iLClT_B43OwtSlbxhX2XOkHpki) |
| [NCI-ALMANAC](tutorials/datasets/10_drug_combination_landscape.ipynb) | Drug combinations and doses | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1460Ai_tDeyFMfEoHCRs65k0NeChd3BFV) |
| [NAS-Bench-201](tutorials/datasets/11_neural_architecture_landscape.ipynb) | Neural architecture choices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11Cr2t_8ojP0T6D1jXQuCrDO-Qnb0p3lV) |
| [LLVM](tutorials/datasets/12_software_configuration_landscape.ipynb) | Compiler configuration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-pxEz0Wpg7M1xXp99RdrQ59XH-lBU37P) |

## Landscape Analysis Features

`analysis.profile()` computes every metric below except those marked †. Pass a group name (for example `metrics="ruggedness"`) to compute one group, or call any function directly.

> Functions describing individual mutations or positions (`fitness_effect_distribution`, `idiosyncratic_index`, `single_mutation_effects`) are not listed here.

<details>
<summary><b>Ruggedness</b>: multimodality and local structure · <code>metrics="ruggedness"</code></summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `local_optima_ratio` | Fraction of variants that are local optima | [0, 1] | more peaks |
| `r_s_ratio` | Roughness-to-slope ratio | [0, ∞) | more rugged |
| `autocorrelation` | Autocorrelation of fitness along random walks | [-1, 1] | less rugged |
| `gradient_intensity` | Mean absolute fitness change per edge | [0, ∞) | steeper gradients |

</details>

<details>
<summary><b>Epistasis</b>: interactions between mutations · <code>metrics="epistasis"</code></summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `gamma` | Correlation of mutation effects across genetic backgrounds | [-1, 1] | more consistent mutation effects |
| `gamma_star` | Consistency of sign epistasis (γ*) | [-1, 1] | more consistent sign epistasis |
| `classify_epistasis` | Fraction of pairwise interactions of each type: magnitude, sign, reciprocal-sign, positive, negative | [0, 1] | n/a (composition) |
| `global_idiosyncratic_index` | How context-dependent (idiosyncratic) mutation effects are | [0, ∞) | more idiosyncratic |
| `diminishing_returns_index` | Pooled background fitness vs. beneficial gains | [-1, 1] | more positive gain trend |
| `increasing_costs_index` | Pooled background fitness vs. deleterious costs | [-1, 1] | more positive cost trend |
| `extradimensional_bypass` | Reciprocal-sign motifs bypassed via extra dimensions (proportion, avg. length) | [0, 1] | more bypasses → more navigable |
| `walsh_hadamard` † | Coefficients, nested fit gains and model variance spectrum | n/a | returns coefficients and order-summary tables |

</details>

<details>
<summary><b>Navigability</b>: reachability of optima · <code>metrics="navigability"</code></summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `global_optima_accessibility` | Fraction of variants on a fitness-monotone path to the global optimum | [0, 1] | more accessible |
| `mean_path_length_to_global_optimum` | Mean shortest adaptive-walk length to the global optimum | [0, ∞) | farther to reach |
| `mean_distance_to_global_optimum` | Mean Hamming distance to the global optimum | [0, ∞) | more spread out |
| `local_optima_accessibility` † | Accessibility of one or more specified local optima | [0, 1] | more accessible |
| `mean_path_length_to_local_optima` † | Mean adaptive-walk length to specified local optima | [0, ∞) | farther to reach |
| `mean_distance_to_local_optima` † | Mean Hamming distance to specified local optima | [0, ∞) | more spread out |

</details>

<details>
<summary><b>Correlation</b>: fitness-distance and basin structure · <code>metrics="correlation"</code></summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `fdc` | Fitness-distance correlation to the global optimum | [-1, 1] | more navigable |
| `neighbor_fitness_correlation` | Correlation of a variant's fitness with its neighbors' mean | [-1, 1] | less rugged |
| `basin_fitness_correlation` | Correlation between basin size and local-optimum fitness | [-1, 1] | fitter peaks have larger basins |
| `fitness_flattening_index` | Whether fitness flattens approaching the global optimum | [-1, 1] | flatter near the peak |

</details>

<details>
<summary><b>Robustness</b>: neutrality and evolvability · <code>metrics="robustness"</code></summary>

| Function | Measures | Range | Higher value → |
|---|---|---|---|
| `neutrality` | Fraction of neutral (equal-fitness) edges | [0, 1] | more neutral |
| `evolvability_enhancing_fraction` | Fraction of directed neighbor pairs with significant evolvability enhancement | [0, 1] | more local EE changes |

</details>

<details>
<summary><b>Fitness distribution</b>: shape statistics · <code>metrics="fitness"</code></summary>

| Function | Measures | Range |
|---|---|---|
| `fitness_distribution` | Unitless shape of the fitness distribution: skewness, kurtosis, coefficient of variation, quartile coefficient, median/mean ratio, relative range, Cauchy location | various |

</details>

<sub>† Not computed by `profile()`; call directly. These need a focal optimum (`lo=...`) or return a table rather than a single value.</sub>

## Landscape Classes

<details>
<summary><b>Seven landscape classes</b>, all built with <code>build_from_data</code></summary>

| Class | Search space | Notes |
|---|---|---|
| `Landscape` | Any discrete space with categorical, ordinal or boolean columns, possibly mixed | Most general; pass `data_types=` |
| `SequenceLandscape` | Categorical sequences over a shared alphabet | General sequence data |
| `BooleanLandscape` | Boolean (binary) space | Optimized for bit-strings |
| `OrdinalLandscape` | Ordinal variables (ordered levels) | Optimized for ordinal data |
| `DNALandscape` | DNA sequences (A/C/G/T) | Optimized for DNA |
| `RNALandscape` | RNA sequences (A/C/G/U) | Optimized for RNA |
| `ProteinLandscape` | Protein sequences (20 amino acids) | Optimized for protein |

</details>

## Synthetic Problem Generators

`graphfla.problems` generates binary landscapes for benchmarking: `NK`, `RoughMountFuji`, `Additive`, `Eggbox` and `HoC` from biology, and `Max3Sat`, `Knapsack` and `NumberPartitioning` from combinatorial optimization. `get_data()` evaluates all 2^n variants, so keep n small.

```python
from graphfla.landscape import BooleanLandscape
from graphfla.problems import NK

X, f = NK(n=10, k=1).get_data()  # X holds bit strings such as "0000000001"
landscape = BooleanLandscape()
landscape.build_from_data(X, f)
```

## Development

Install `requirements-dev.txt` and run `python -m pytest`. See the [test contracts](tests/README.md), [performance benchmarks](benchmarks/README.md) and [construction optimization results](benchmarks/RESULTS.md). To preview the documentation website locally, see the [documentation guide](docs/README.md).

## License

This project is licensed under the terms of the [MIT License](./LICENSE).

## Citation

If you use GraphFLA, please cite:

```
@inproceedings{HuangZL25,
  author    = {Mingyu Huang and
               Shasha Zhou and
               Ke Li},
  title     = {Augmenting Biological Fitness Prediction Benchmarks with Landscape Features
               from GraphFLA},
  booktitle = {{NeurIPS}'25: Proc. of Advances in Neural Information Processing Systems 39},
  year      = {2025},
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

**Happy analyzing!** If you have questions or suggestions, please open an issue or start a discussion.
