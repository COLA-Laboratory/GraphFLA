---
api_grouped_classes: true
---

# Biological Models

`graphfla.problems` ships a collection of **synthetic optimization problems** that you can use to generate fitness landscapes on demand. They are useful for three reasons:

1.  As **teaching tools** — every model on this page has a well-understood theoretical profile (smooth, tunably rugged, completely random, etc.), which makes it easy to verify your understanding of an analysis or sanity-check a new pipeline.
2.  As **benchmarks** — when comparing analysis methods or evolutionary algorithms, you usually want a controlled fitness function whose properties (number of peaks, level of epistasis, dimensionality) you can dial up or down.
3.  As **building blocks** — each problem returns a `(configurations, fitness)` pair via `get_data()` that you can feed straight into a [`Landscape`](landscape.md) constructor.

The module is split into two families:

-   **Biological models** ([`NK`](#kauffmans-nk-landscape-model), [`RoughMountFuji`](#rough-mount-fuji-rmf-model), [`HoC`](#house-of-cards-hoc-model), [`Additive`](#additive-model), [`Eggbox`](#eggbox-model)) — classic tunable models from evolutionary biology and quantitative genetics.
-   **Combinatorial models** ([`Max3Sat`](problems/combinatorial.md#max-3-sat), [`Knapsack`](problems/combinatorial.md#01-knapsack), [`NumberPartitioning`](problems/combinatorial.md#number-partitioning)) — staples of the NP-hard optimization literature.

--8<-- "problem-interface.md"

## Overview

| API | Purpose |
| --- | --- |
| [`OptimizationProblem`](#base-class) | Base class — subclass to define a custom synthetic problem. |
| [`NK`](#kauffmans-nk-landscape-model) | Kauffman's tunable epistasis model ($K$ controls ruggedness). |
| [`RoughMountFuji`](#rough-mount-fuji-rmf-model) | Convex blend of additive + random landscape ($\alpha$ controls ruggedness). |
| [`HoC`](#house-of-cards-hoc-model) | House-of-Cards — maximally rugged, uncorrelated fitness. |
| [`Additive`](#additive-model) | Smooth additive landscape with independent per-locus contributions. |
| [`Eggbox`](#eggbox-model) | Deterministic periodic peaks; fitness depends only on Hamming weight. |

## Base Class

Common framework for all synthetic problems in `GraphFLA`. End users typically use one of the concrete subclasses below; this base class is exposed for those who want to define a custom problem.

**Custom problems**

Subclassing `OptimizationProblem` is straightforward — implement `evaluate(config)` (and optionally `_binary_string_to_config` if your configuration type differs from `tuple[int]`). See the source of [`Eggbox`](#eggbox-model) for a minimal example.

::: graphfla.problems.OptimizationProblem

## Kauffman's *NK* Landscape Model

The Kauffman *NK* model is a tunable fitness-landscape model where $N$ loci interact with $K$ other randomly chosen loci. The model interpolates between simple, smooth landscapes and complex, rugged ones — when $K=0$ each locus contributes to fitness independently (an [additive model](#additive-model)); when $K=N-1$ every locus interacts with all others and the landscape approaches a [House-of-Cards model](#house-of-cards-hoc-model).

The fitness of a configuration is computed as the average of $N$ per-locus contributions; each contribution is a deterministic function of the locus's state *and* the states of its $K$ epistatic partners. Contributions are sampled lazily from $U(0, 1)$ via the instance's seeded RNG, then cached, so repeated `evaluate(config)` calls are consistent.

!!! note "Statistical signatures"
    For an *NK* landscape with $K = N - 1$, the expected number of local optima under best-improvement adaptive walks is approximately $\frac{2^N}{N + 1}$. With $N = 20$ that yields about 50,000 local optima — a useful sanity-check reference when developing ruggedness measures.

::: graphfla.problems.NK

**Example — generate a synthetic landscape, then analyze it:**

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import local_optima_ratio, autocorrelation

problem = NK(n=10, k=4, seed=42)
X, f = problem.get_data()

landscape = BooleanLandscape().build_from_data(X, f, verbose=False)
print(f"Ruggedness (LO ratio): {local_optima_ratio(landscape):.3f}")
print(f"Autocorrelation:       {autocorrelation(landscape, seed=0):.3f}")
```

## Rough Mount Fuji (RMF) Model

The Rough Mount Fuji (RMF) model interpolates between a perfectly smooth, additive landscape and a completely rugged, random one. The fitness is a convex combination of:

1.  An **additive (smooth) component** — each locus makes an independent contribution drawn from $U(-1, 1)$ at construction time. The smooth value for a configuration is the sum of contributions from active loci.
2.  A **random (rugged) component** — each configuration is assigned an independent value drawn from $U(0, 1)$ when first evaluated (House-of-Cards-style).

$$
F(\sigma) = (1 - \alpha) \, F_{\text{smooth}}(\sigma) \;+\; \alpha \, F_{\text{rugged}}(\sigma)
$$

::: graphfla.problems.RoughMountFuji

## House-of-Cards (HoC) Model

The House-of-Cards (HoC) model represents a maximally rugged, completely uncorrelated landscape: every configuration is assigned an independent fitness drawn from $U(0, 1)$. There is no structure to exploit and adaptive walks halt at the first local optimum encountered.

In `GraphFLA`, HoC is implemented as the limiting case `RoughMountFuji(n, alpha=1.0)` — so it inherits `get_data` and `evaluate` from RMF.

::: graphfla.problems.HoC

## Additive Model

The Additive model is the smoothest possible landscape: per-locus contributions are independently drawn and the fitness is just their sum. There are no epistatic interactions, so the landscape is unimodal and adaptive walks always reach the global optimum.

At construction, each locus is assigned two contributions (one for state `0`, one for state `1`) drawn from $U(0, 1)$ via the instance's RNG.

::: graphfla.problems.Additive

## Eggbox Model

The Eggbox model is a deterministic landscape whose fitness depends *only* on the sum of bits in the configuration. Concretely,

$$
F(\sigma) = \sin^2\!\bigl(\,\text{frequency} \cdot \textstyle\sum_i \sigma_i \cdot \pi\bigr).
$$

This creates a periodic peak-and-valley pattern: regularly spaced peaks and many redundant local optima. Useful as a stress-test for ruggedness and basin-size analyses because every configuration with the same Hamming weight has the same fitness.

::: graphfla.problems.Eggbox

## Example — Full Pipeline

A typical experiment generates data from a synthetic problem, builds a landscape, and feeds it into the analysis stack:

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import (
    local_optima_ratio,
    autocorrelation,
    classify_epistasis,
    fdc,
)

# Synthetic landscape with moderate epistasis
problem = NK(n=10, k=5, seed=42)
X, f = problem.get_data()

landscape = BooleanLandscape().build_from_data(
    X, f,
    verbose=False,
)

print("Ruggedness:")
print(f"  LO ratio:        {local_optima_ratio(landscape):.3f}")
print(f"  Autocorrelation: {autocorrelation(landscape, seed=0):.3f}")
print()
print(f"FDC: {fdc(landscape):.3f}")
print()
print("Epistasis:")
print(classify_epistasis(landscape, sample_cut_prob=0.1, seed=0))
```

## References

-   Stuart A. Kauffman, "The Origins of Order: Self-Organization and Selection in Evolution", *Oxford University Press* (1993).
-   Stuart A. Kauffman and Simon Levin, "Towards a general theory of adaptive walks on rugged landscapes", *J. Theor. Biol.* (1987).
-   Takuyo Aita *et al.*, "Analysis of a local fitness landscape with a model of the rough Mount Fuji-type landscape", *Biophys. Chem.* (2000).
