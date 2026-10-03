---
api_narrative: true
---

Robustness in fitness landscapes is the flip side of evolvability: it asks not "how can a population improve?" but "how *little* does fitness change when a population is perturbed?". A landscape with many neutral mutations and large flat regions is *robust*; one where every mutation is consequential is *fragile*. Mutational robustness shapes the speed and direction of adaptation: under high mutation rates, selection often favors variants that occupy flat regions of the landscape (a phenomenon termed "survival of the flattest").

This page documents the robustness diagnostics in `GraphFLA`. They fall into three groups:

1.  **Global neutrality** — `neutrality`, summarizing how widespread near-neutral mutations are across the entire landscape.
2.  **Per-mutation effect distributions** — `single_mutation_effects` and `all_mutation_effects`, which test the significance of specific or all mutations across genetic backgrounds.
3.  **Evolvability-enhancing mutations** — `evolvability_enhancing_fraction` and `evolvability_effects`, which summarize statistically supported changes in the opportunities for subsequent adaptation.

## Overview

| API | Purpose |
| --- | --- |
| [`neutrality`](#neutrality) | Fraction of represented neighbor pairs with `\|Δf\| ≤ threshold`. |
| [`single_mutation_effects`](#per-position-significance-test) | Per-position binomial test of all allele-pair fitness effects. |
| [`all_mutation_effects`](#all-position-summary) | `single_mutation_effects` aggregated across every position. |
| [`evolvability_enhancing_fraction`](#ee-fraction) | Fraction of observed directed mutations with a statistically supported EE effect. |
| [`evolvability_effects`](#per-mutation-ee-results) | Per-mutation EE statistics and significance. |

## Neutrality

Calculates the overall neutrality of the landscape.

Landscape neutrality describes the presence of regions where distinct genotypes exhibit equivalent or very similar fitness levels. Mutations that result in negligible or no change in fitness are known as neutral mutations, and interconnected sets of such genotypes form neutral networks (or "effectively neutral" regions). These neutral pathways enable populations to explore a wide array of genetic variations without incurring significant fitness penalties.

This function measures the overall neutrality of the landscape by calculating the proportion of single-point mutations whose absolute fitness effect falls below the specified `threshold`. A higher neutrality value suggests larger neutral networks or areas where mutations have minimal impact on fitness.

::: graphfla.analysis.neutrality

## Mutational Robustness

On the individual genotype level, mutational robustness refers to the ability of a genotype to preserve its phenotype (fitness) when subjected to mutations. Genotypes exhibiting high mutational robustness can endure a greater proportion of mutational changes with minimal or no adverse effects. Such genotypes often occupy "flatter" areas of the fitness landscape, and under conditions of high mutation rates, selection might favor these robust genotypes through "survival of the flattest".

### Per-position significance test

Assess the fitness effects of all possible mutations at a single position across all genetic backgrounds.

For every pair of distinct alleles $(A, B)$ at the given position, the function pairs up genotypes sharing the same background and reports the median absolute effect size (normalized by the landscape's fitness standard deviation), the mean effect, and the $p$-value of a one-sided binomial test of whether the effect is *positive* (i.e., $f(\text{B-background}) > f(\text{A-background})$) or *negative*.

::: graphfla.analysis.single_mutation_effects

### All-position summary

Apply `single_mutation_effects` across **all positions** of the landscape, returning a concatenated DataFrame indexed by position and allele pair.

::: graphfla.analysis.all_mutation_effects

--8<-- "ee-mutations.md"

### EE fraction

::: graphfla.analysis.evolvability_enhancing_fraction

### Per-mutation EE results

::: graphfla.analysis.evolvability_effects
