---
icon: material/cube-outline
---

Robustness in fitness landscapes is the flip side of evolvability: it asks not "how can a population improve?" but "how *little* does fitness change when a population is perturbed?". A landscape with many neutral mutations and large flat regions is *robust*; one where every mutation is consequential is *fragile*. Mutational robustness shapes the speed and direction of adaptation: under high mutation rates, selection often favors variants that occupy flat regions of the landscape (a phenomenon termed "survival of the flattest").

This page documents the robustness diagnostics in `GraphFLA`. They fall into three groups:

1.  **Global neutrality** — `neutrality`, summarizing how widespread near-neutral mutations are across the entire landscape.
2.  **Per-mutation effect distributions** — `single_mutation_effects` and `all_mutation_effects`, which test the significance of specific or all mutations across genetic backgrounds.
3.  **Evolvability-enhancing mutations** — `evol_enhance_mutations` (formerly `calculate_evol_enhance`), which quantifies the fraction of beneficial mutations that also raise the mean fitness of the neighborhood.

## Overview

| API | Purpose |
| --- | --- |
| [`neutrality`](#neutrality) | Proportion of single-mutation edges with `\|Δf\| ≤ threshold`. |
| [`single_mutation_effects`](#per-position-significance-test) | Per-position binomial test of all allele-pair fitness effects. |
| [`all_mutation_effects`](#all-position-summary) | `single_mutation_effects` aggregated across every position. |
| [`evol_enhance_mutations`](#evolvability-enhancing-mutations) | Fraction of improving edges that also raise mean neighbor fitness (Wagner 2023). |

## Neutrality

```api
def graphfla.analysis.neutrality(landscape, threshold: float = 0.01) -> float
```

Calculates the overall neutrality of the landscape.

Landscape neutrality describes the presence of regions where distinct genotypes exhibit equivalent or very similar fitness levels. Mutations that result in negligible or no change in fitness are known as neutral mutations, and interconnected sets of such genotypes form neutral networks (or "effectively neutral" regions). These neutral pathways enable populations to explore a wide array of genetic variations without incurring significant fitness penalties.

This function measures the overall neutrality of the landscape by calculating the proportion of single-point mutations whose absolute fitness effect falls below the specified `threshold`. A higher neutrality value suggests larger neutral networks or areas where mutations have minimal impact on fitness.

!!! api-parameters "Parameters"

    **landscape : *object***
    :   The fitness landscape object.

    **threshold : *float, default=0.01***
    :   The noise tolerance threshold. Pairs with `|f_a - f_b| <= threshold` are considered neutral.


!!! api-returns "Returns"

    **neutrality : *float***
    :   The neutrality index, a value between 0.0 and 1.0.
    
        - A value closer to 1.0 indicates a highly neutral landscape, in which most mutations have negligible fitness effects (most genotypes have similar fitness).
    
        - A value closer to 0.0 indicates a landscape where neutral mutations are rare.


!!! note "Plateau-aware behavior"
    When the landscape was built with `epsilon > 0`, neutral neighbor pairs stored during construction (`landscape._neutral_neighbors`) are included in the calculation alongside the standard graph-based neighbors. This ensures that equal-fitness pairs — which carry no directed edge in the improving graph — are still counted toward the neutrality metric.

## Mutational Robustness

On the individual genotype level, mutational robustness refers to the ability of a genotype to preserve its phenotype (fitness) when subjected to mutations. Genotypes exhibiting high mutational robustness can endure a greater proportion of mutational changes with minimal or no adverse effects. Such genotypes often occupy "flatter" areas of the fitness landscape, and under conditions of high mutation rates, selection might favor these robust genotypes through "survival of the flattest".

### Per-position significance test

```api
def graphfla.analysis.single_mutation_effects(landscape, position: str, test_type: str = "positive", n_jobs: int = 1) -> pandas.DataFrame
```

Assess the fitness effects of all possible mutations at a single position across all genetic backgrounds.

For every pair of distinct alleles $(A, B)$ at the given position, the function pairs up genotypes sharing the same background and reports the median absolute effect size (normalized by the landscape's fitness standard deviation), the mean effect, and the $p$-value of a one-sided binomial test of whether the effect is *positive* (i.e., $f(\text{B-background}) > f(\text{A-background})$) or *negative*.

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object.

    **position : *str***
    :   The column name (variable / site) for which to assess all pairwise allele mutations.

    **test_type : *{'positive', 'negative'}, default='positive'***
    :   Direction of the one-sided binomial test against $H_0: \Pr(\text{effect} > 0) = 0.5$.

    **n_jobs : *int, default=1***
    :   Number of parallel workers (passed to `joblib`).


!!! api-returns "Returns"

    **pandas.DataFrame**
    :   One row per ordered allele pair $(A, B)$ at the position, with columns:
    
        -   `mutation_from`, `mutation_to`: the allele pair.
    
        -   `median_abs_effect`: median absolute fitness difference, normalized by `f.std()`.
    
        -   `mean_effect`: mean fitness difference.
    
        -   `p_value`: $p$-value of the binomial test.
    
        -   `significant`: whether `p_value < 0.05`.


### All-position summary

```api
def graphfla.analysis.all_mutation_effects(landscape, test_type: str = "positive", n_jobs: int = 1) -> pandas.DataFrame
```

Apply `single_mutation_effects` across **all positions** of the landscape, returning a concatenated DataFrame indexed by position and allele pair.

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object.

    **test_type : *{'positive', 'negative'}, default='positive'***
    :   Forwarded to `single_mutation_effects`.

    **n_jobs : *int, default=1***
    :   Number of parallel workers (positions are parallelized).


!!! api-returns "Returns"

    **pandas.DataFrame**
    :   A long-format DataFrame containing the per-position outputs of `single_mutation_effects` stacked together. Useful for identifying mutations whose effects are robust (consistently insignificant) or universally beneficial (consistently positive and significant).


## Evolvability-enhancing Mutations

```api
def graphfla.analysis.evol_enhance_mutations(landscape, epsilon: float = 0, auto_calculate: bool = True) -> float
```

Calculates the proportion of fitness-increasing edges whose target also has a *higher mean neighbor fitness* than the source.

Wagner ([2023](https://doi.org/10.1038/s41576-023-00559-0)) defines an evolvability-enhancing (EE) mutation as one that creates a genetic background in which subsequent mutations are more likely to be adaptive. Equivalently, an EE mutation moves the population to a node whose neighborhood is on average fitter than the original neighborhood. This function returns the fraction of improving edges that satisfy that criterion.

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object.

    **epsilon : *float, default=0***
    :   Tolerance threshold. An edge counts as EE if its `delta_mean_neighbor_fit` strictly exceeds `epsilon`.

    **auto_calculate : *bool, default=True***
    :   If `True`, automatically computes neighbour fitness (via the `landscape.neighbor_fitness` property) when the required edge attribute is missing. If `False`, raises `RuntimeError`.


!!! api-returns "Returns"

    **float**
    :   The proportion of edges classified as EE, in `[0.0, 1.0]`.


!!! note "Deprecated alias"
    `graphfla.analysis.calculate_evol_enhance` is a deprecated alias for `evol_enhance_mutations` (issues a `FutureWarning`). New code should use the canonical name.

**References**

- Andreas Wagner, "Evolvability-enhancing mutations in the fitness landscapes of an RNA and a protein." *Nat. Commun.* (2023).
