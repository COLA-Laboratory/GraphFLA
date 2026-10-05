---
api_narrative: true
---

Ruggedness is arguably the most widely studied topograhical aspect of fitness landscapes. Although biologists have offered a staggering number of definitions of ruggedness, they agree broadly on the essence of the idea: the lack of correlation in fitness between genotypes. This can either manifests as the presence of multiple fitness peaks (local optima) or significant fitness fluctuation.

**Relationship to landscape navigability:** Ruggedness can pose a fundamental challenge to evolution’s ability to find a landscape’s highest peaks, i.e., landscape *navigability* or peak *accessibility*. This is because a population evolving under the influence of natural selection can only travel on *accessible paths* through the landscape, that is, paths in which each mutational step increases fitness. The reason is that natural selection favors high fitness genotypes and does not allow a population to traverse low fitness valleys between a local peak of intermediate fitness and nearby higher fitness peaks.

- In a smooth landscpae, all mutational paths to the single peak are accessible. 
- When landscape becomes rugged, many peaks exist, yet high-fitness peaks are reachable by abundant short accessible paths, especially in high-dimensional sequence spaces. 
- For a maximumally rugged landscape, selectively accessible paths become rare, and evolution may stall on suboptimal peaks.

**Relationship to epistasis:** Reciprocal sign epistasis is a necessary yet not sufficient condition for landscape ruggedness-landscape can be rugged even when it is rare, while its mere presence does not guarantee ruggedness. Yet, for a nonepistatic (i.e., purely additive) landscape, there is often only a single, global fitness peak. In contrast, pervasive sign epistasis could create a rugged landscape with numerous sub-optimal peaks.

Following the key idea of ruggedness, various established quantitative measures have been developed.

## Overview

| API | Purpose |
| --- | --- |
| [`local_optima_ratio`](#local-optima) | Independent local optima divided by retained configurations. |
| [`r_s_ratio`](#roughness-to-slope-rs-ratio) | Roughness-to-slope ratio — deviation from a purely additive fit. |
| [`autocorrelation`](#autocorrelation) | Fitness autocorrelation along random walks (Weinberger 1990). |
| [`gradient_intensity`](#gradient-intensity) | Mean absolute fitness step per edge, normalized by mean fitness. |
| [`neighbor_fitness_correlation`](#neighbor-fitness-correlation) | Correlation between node fitness and mean neighbor fitness. |

## Local Optima

Calculates the proportion of local optima variants in the landscape. 

The number of local optima (peaks) is a principal indicator of fitness landscape ruggedness. To make it a unitless measure, we divide it by the number of total variants in the landscape. To provide a sense of its magnitude, a mostly rugged *NK* landscape with dimension $n$ and degree of interaction $k=n-1$ would have $\frac{2^n}{n+1}$ local optima. When $n=20$, the ratio of local optima would be around $4.76\%$.

::: graphfla.analysis.local_optima_ratio

## Roughness-to-Slope (r/s) Ratio

The roughness-to-slope (r/s) ratio measures how well a landscape can be described by an additive model, where individual variables contribute independently to the objective value. It is calculated by fitting a linear model to the encoded configurations using least squares. The slope is the mean absolute additive coefficient, while the roughness is the root mean square of the residuals.

A smaller ratio indicates less variation around the additive trend, reaching zero for an exact additive fit with nonzero slope. A larger ratio indicates greater residual variation relative to that trend. Comparisons require consistent variable encoding: categorical reference states can affect the ratio, and nonlinear effects of an ordinal variable can contribute to roughness even without interactions between variables.

::: graphfla.analysis.r_s_ratio

## Autocorrelation

Calculates the autocorrelation of a fitness landscape (Weinberger 1990).

Autocorrelation measures landscape ruggedness by simulating random walks across the landscape. For a smooth landscape, fitness values for adjacent variants encountered during the same walk would be highly correlated (autocorrelation close to 1). In contrasts, this correlation diminishes in rugged landscapes wherein fitness values fluctuates dramatically even across a single mutation (autocorrelation close to 0).

::: graphfla.analysis.autocorrelation

## Gradient Intensity

Calculates the gradient intensity of the landscape: the average absolute fitness difference (`delta_fit`) along all edges of the landscape graph, normalized by the mean fitness so that the result is scale-invariant.

A landscape with steep "uphill" edges relative to the typical fitness magnitude will have a high gradient intensity. A landscape whose fitness values are nearly flat (or whose typical fitness step is small compared to the overall fitness scale) will have a low gradient intensity. Together with `autocorrelation` and `r_s_ratio`, this provides a complementary numeric signature of how locally varied the landscape is.

::: graphfla.analysis.gradient_intensity

## Neighbor Fitness Correlation

Calculates the correlation between a configuration's fitness and the mean fitness of its neighbors across the fitness landscape.

Neighbor fitness correlation (NFC) reflects how fitness values are distributed spatially. A strong positive correlation indicates that high-fitness configurations tend to be surrounded by other high-fitness neighbors—and similarly for low-fitness ones—suggesting clustering, separability, and an underlying gradient in the landscape. In contrast, a weak or negative correlation implies that high- and low-fitness configurations are intermixed, signaling high fitness variability and a rugged, unstructured landscape.

::: graphfla.analysis.neighbor_fitness_correlation

