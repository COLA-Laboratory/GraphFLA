---
api_grouped_classes: true
---

# Biological Models

`graphfla.problems` provides synthetic fitness landscapes from evolutionary biology. Their behavior is well characterized, and each has a parameter that controls ruggedness. This makes them suitable for checking an analysis pipeline before applying it to experimental data, and for comparing methods under controlled conditions.

All models on this page define fitness over $n$ binary variables. Classic NP-hard optimization problems are documented under [Combinatorial Problems](problems/combinatorial.md).

--8<-- "problem-interface.md"

## Overview

| API | Purpose |
| --- | --- |
| [`OptimizationProblem`](#base-class) | Base class for defining a custom problem. |
| [`NK`](#nk-model) | Kauffman's model with tunable epistasis; $k$ controls ruggedness. |
| [`RoughMountFuji`](#rough-mount-fuji-model) | Weighted sum of an additive and a random component; $\alpha$ controls ruggedness. |
| [`HoC`](#house-of-cards-model) | Independent random fitness for every configuration. |
| [`Additive`](#additive-model) | Independent contributions from each variable, with no epistasis. |
| [`Eggbox`](#eggbox-model) | Periodic fitness that depends only on the number of ones. |

## Base Class

To define a custom problem, subclass `OptimizationProblem` and implement `evaluate(config)`. The base class provides `get_data()` and `iter_data()`. The example in the class documentation below is a complete implementation.

::: graphfla.problems.OptimizationProblem

## NK Model

In Kauffman's NK model, each of the $n$ variables contributes to fitness according to its own state and the states of $k$ other variables chosen at random. Fitness is the mean of these $n$ contributions. Each contribution is drawn from $U(0, 1)$ the first time its combination of states is evaluated, and then cached.

The parameter $k$ sets the degree of epistasis. With $k = 0$, the model is additive and has a single peak. With $k = n - 1$, the fitness values of different configurations are independent, as in the [House of Cards model](#house-of-cards-model). Such a landscape has on average $2^n / (n + 1)$ local optima, or about 50,000 for $n = 20$.

::: graphfla.problems.NK

The following example generates an NK landscape and computes two ruggedness measures:

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import local_optima_ratio, autocorrelation

problem = NK(n=10, k=4, seed=42)
X, f = problem.get_data()

landscape = BooleanLandscape().build_from_data(X, f, verbose=False)
print(f"Local optima ratio: {local_optima_ratio(landscape):.3f}")
print(f"Autocorrelation:    {autocorrelation(landscape, seed=0):.3f}")
```

## Rough Mount Fuji Model

The Rough Mount Fuji model adds random fluctuations to an additive landscape. Each variable has a coefficient $w_i$ drawn from $U(-1, 1)$, and each configuration has a random value $u(\sigma)$ drawn from $U(0, 1)$ when it is first evaluated:

$$
F(\sigma) = (1 - \alpha) \sum_{i=1}^{n} w_i \sigma_i + \alpha \, u(\sigma).
$$

With $\alpha = 0$ the landscape is additive, and with $\alpha = 1$ it is a House of Cards landscape. The additive term is not normalized by $n$, so $\alpha$ is a mixing weight rather than the fraction of fitness variance due to the random component.

::: graphfla.problems.RoughMountFuji

## House of Cards Model

The House of Cards model assigns every configuration an independent fitness value from $U(0, 1)$. Fitness is uncorrelated between neighbors, so the landscape is maximally rugged. `HoC(n)` produces the same landscape as `RoughMountFuji(n, alpha=1.0)` with the same seed.

::: graphfla.problems.HoC

## Additive Model

In the additive model, each variable has two contributions, one for each state, drawn from $U(0, 1)$. Fitness is the sum of the contributions selected by the configuration. Without epistasis, the landscape has a single peak, and every path of improving single mutations leads to it.

::: graphfla.problems.Additive

## Eggbox Model

The Eggbox model is deterministic. Fitness depends only on the number of ones in the configuration:

$$
F(\sigma) = \sin^2\!\Bigl(\pi \cdot \text{frequency} \cdot \sum_{i=1}^{n} \sigma_i\Bigr).
$$

All configurations with the same number of ones have the same fitness. With the default `frequency=0.5`, configurations with an odd number of ones have fitness 1 and those with an even number have fitness 0. Every single mutation then moves between the two groups, so half of all configurations are global optima. This makes the model a useful edge case for ruggedness and basin analyses.

::: graphfla.problems.Eggbox

## Full Example

The following example builds an NK landscape and computes measures of ruggedness, fitness-distance correlation and epistasis:

```python
from graphfla.problems import NK
from graphfla.landscape import BooleanLandscape
from graphfla.analysis import (
    autocorrelation,
    classify_epistasis,
    fdc,
    local_optima_ratio,
)

problem = NK(n=10, k=5, seed=42)
X, f = problem.get_data()
landscape = BooleanLandscape().build_from_data(X, f, verbose=False)

print(f"Local optima ratio: {local_optima_ratio(landscape):.3f}")
print(f"Autocorrelation:    {autocorrelation(landscape, seed=0):.3f}")
print(f"FDC:                {fdc(landscape):.3f}")
print(classify_epistasis(landscape, seed=0))
```

## References

-   Stuart A. Kauffman, "The Origins of Order: Self-Organization and Selection in Evolution", *Oxford University Press* (1993).
-   Stuart A. Kauffman and Simon Levin, "Towards a general theory of adaptive walks on rugged landscapes", *J. Theor. Biol.* (1987).
-   Takuyo Aita *et al.*, "Analysis of a local fitness landscape with a model of the rough Mount Fuji-type landscape", *Biophys. Chem.* (2000).
