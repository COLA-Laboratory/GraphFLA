---
title: Quick start
description: Build and analyse your first fitness landscape from a small table of configurations and scores.
icon: material/rocket-launch-outline
---

# Quick start

In this example, we’ll explore eight configurations of three on/off choices. Each configuration has a score: higher is better. The values are illustrative, so you can run the whole example without downloading a dataset.

## 1. Install

```bash
pip install graphfla
```

## 2. Build a landscape

Put the choices in `X` and their scores in `f`. Each row is one configuration; each column is one variable.

```python
import pandas as pd
from graphfla.landscape import Landscape
from graphfla import analysis

X = pd.DataFrame({
    "a": [0, 0, 0, 0, 1, 1, 1, 1],
    "b": [0, 0, 1, 1, 0, 0, 1, 1],
    "c": [0, 1, 0, 1, 0, 1, 0, 1],
})
f = [0, 1, 2, 5, 2, 3, 6, 4]

landscape = Landscape(maximize=True)
landscape.build_from_data(
    X, f, data_types={name: "boolean" for name in X}, verbose=False
)
```

GraphFLA connects configurations that differ in one choice and directs improving moves towards higher scores. Use `maximize=False` for an objective such as cost or error, where lower is better.

## 3. Find the peaks

```python
print(analysis.local_optima_ratio(landscape))

data = landscape.get_data()
print(data.loc[data["is_lo"], ["a", "b", "c", "fitness"]])
```

```text
0.25
   a  b  c  fitness
3  0  1  1      5.0
6  1  1  0      6.0
```

Two of the eight configurations are local optima: neither has a better neighbour. Only the second has the highest score. A search that accepts only improvements can stop at the first peak.

## Use your own data

Replace `X` and `f` with columns from your table, then repeat the construction and analysis:

```python
df = pd.read_csv("experiments.csv")
X = df[["a", "b", "c"]]
f = df["score"]
```

For other variable types, declare each column as `"categorical"`, `"ordinal"` or `"boolean"` in [`Landscape`](landscape.md). Explore [worked tutorials](tutorials/index.md) for real datasets, or use [`analysis.profile`](analysis/profile.md) to calculate several landscape features together.
