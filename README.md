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

**GraphFLA** (Graph-based Fitness Landscape Analysis) is a Python framework for constructing, analyzing, manipulating and visualizing **fitness landscapes** of combinatorial optimization problems. Drawing on concepts and measures established in evolutionary biology, it represents a landscape as a graph of variants connected by single mutations and characterizes the topography of complex optimization landscapes across diverse fields.

This is also the official code and data repository for the **NeurIPS 2025 (Spotlight)** paper "Augmenting Biological Fitness Prediction Benchmarks with Landscape Features from GraphFLA".

Full documentation, including the API reference, is at [colalab.ai/GraphFLA](https://colalab.ai/GraphFLA/).

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

## Key Features
- **Versatility:** GraphFLA applies to any discrete, combinatorial mapping from variants to fitness, from biological sequences to chemical reaction conditions, material compositions, drug combinations and software configurations.
- **Comprehensiveness:** more than 20 metrics, drawn from the evolutionary biology and evolutionary computation literature, quantify ruggedness, epistasis, navigability and neutrality, together giving a detailed account of landscape topography.
- **Interoperability:** GraphFLA analyzes the same variants `X` and fitness values `f` used to train machine learning models, so landscape analysis fits into existing modeling workflows without data conversion.
- **Scalability:** GraphFLA constructs and analyzes landscapes comprising millions of variants.

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
