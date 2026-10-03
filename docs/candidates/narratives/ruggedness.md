---
icon: material/cube-outline
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
| [`lo_ratio`](#local-optima) | Fraction of variants that are local optima — most intuitive ruggedness index. |
| [`r_s_ratio`](#roughness-to-slope-rs-ratio) | Roughness-to-slope ratio — deviation from a purely additive fit. |
| [`autocorrelation`](#autocorrelation) | Fitness autocorrelation along random walks (Weinberger 1990). |
| [`gradient_intensity`](#gradient-intensity) | Mean absolute fitness step per edge, normalized by mean fitness. |
| [`neighbor_fit_corr`](#neighbor-fitness-correlation) | Correlation between node fitness and mean neighbor fitness. |

## Local Optima

```api
def graphfla.analysis.lo_ratio(landscape) -> float
```

Calculates the proportion of local optima variants in the landscape. 

The number of local optima (peaks) is a principal indicator of fitness landscape ruggedness. To make it a unitless measure, we divide it by the number of total variants in the landscape. To provide a sense of its magnitude, a mostly rugged *NK* landscape with dimension $n$ and degree of interaction $k=n-1$ would have $\frac{2^n}{n+1}$ local optima. When $n=20$, the ratio of local optima would be around $4.76\%$.


!!! api-parameters "Parameters"

    **landscape** : ***Landscape***
    :   The fitness landscape object.


!!! api-returns "Returns"

    **float**
    :   The ratio of local optima, ranging from 0 to 1. Larger values indicate more rugged landscapes.


## Roughness-to-Slope (r/s) Ratio

```api
def graphfla.analysis.r_s_ratio(landscape) -> float
```

Calculates the roughness-to-slope (r/s) ratio of a fitness landscape.

The Roughness-to-Slope (r/s) ratio is an overall measure of the ruggedness of a fitness landscape that quantifies how well the landscape can be described by a purely additive model, where the effects of individual mutations sum linearly. It is calculated by fitting the fitness landscape to a multidimensional linear model using the least-squares method. The slope of the linear model corresponds to the average additive fitness effect, whereas the roughness is given by the variance of the residuals. 

Generally, the better the linear model fit, the smaller the variance in residuals such that the roughness-to-slope ratio approaches 0 in a perfectly additive model (i.e., smooth landscape). Conversely, a very rugged fitness landscape would have a large residual variance and, thus, a very large roughness-to-slope ratio.

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object.


!!! api-returns "Returns"

    **float**
    :   The calculated roughness-to-slope ($r/s$) ratio.


!!! warning "Underdetermined or Unstable Regression"
    If the number of samples (genotypes) in the landscape data is less than or equal to the number of features (after encoding categorical/boolean variables) used for the linear regression, a warning is issued. In such cases, the linear regression model might be underdetermined or unstable, potentially affecting the reliability of the $r/s$ ratio.

!!! warning "Zero or Near-Zero Slope"
    If the calculated slope ($s$) is zero or extremely close to zero, it suggests that the landscape is either flat or that the additive model explains very little of the fitness variation (implying strong epistasis or noise dominating any additive signal). In this scenario, the function returns `numpy.inf` for the $r/s$ ratio and issues a warning.

## Autocorrelation

```api
def graphfla.analysis.autocorrelation(landscape, walk_length: int = 20, walk_times: int = 1000, lag: int = 1) -> float
```

Calculates the autocorrelation of a fitness landscape (Weinberger 1990).

Autocorrelation measures landscape ruggedness by simulating random walks across the landscape. For a smooth landscape, fitness values for adjacent variants encountered during the same walk would be highly correlated (autocorrelation close to 1). In contrasts, this correlation diminishes in rugged landscapes wherein fitness values fluctuates dramatically even across a single mutation (autocorrelation close to 0). 

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object.

    **walk_length : *int, default=20***
    :   The number of steps taken in each individual random walk. Recommend to be slightly shorter than (or comparable to) the dimension of the landscape (i.e., length of each sequence).

    **walk_times : *int, default=1000***
    :   The total number of independent random walks to perform on the landscape. It controls the noise in the measure, where more repeatations leads to in more reliable measurement. 

    **lag : *int, default=1***
    :   The distance (number of steps) lag used for calculating the autocorrelation within each walk. For example, a lag of 1 compares fitness at step $i$ with fitness at step $i+1$.


!!! api-returns "Returns"

    **autocorr : *float***
    :   The mean of the autocorrelation values calculated across all random walks. Values close to 0 indicates a fairly rugged landscape.


**References**

-   E. Weinberger, "Correlated and Uncorrelated Fitness Landscapes and How to Tell the Difference," *Biol. Cybern.* 63, 325-336 (1990).


## Gradient Intensity

```api
def graphfla.analysis.gradient_intensity(landscape) -> float
```

Calculates the gradient intensity of the landscape: the average absolute fitness difference (`delta_fit`) along all edges of the landscape graph, normalized by the mean fitness so that the result is scale-invariant.

A landscape with steep "uphill" edges relative to the typical fitness magnitude will have a high gradient intensity. A landscape whose fitness values are nearly flat (or whose typical fitness step is small compared to the overall fitness scale) will have a low gradient intensity. Together with `autocorrelation` and `r_s_ratio`, this provides a complementary numeric signature of how locally varied the landscape is.

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object.


!!! api-returns "Returns"

    **float**
    :   The gradient intensity. Returns `0.0` if the landscape has no edges.


!!! note
    Requires the `delta_fit` edge attribute, which is automatically populated during `Landscape.build_from_data`.


## Neighbor Fitness Correlation

```api
def graphfla.analysis.neighbor_fit_corr(landscape: BaseLandscape, auto_calculate: bool = True, method: Literal["pearson", "spearman", "kendall"] = "pearson") -> float
```

Calculates the correlation between a configuration's fitness and the mean fitness of its neighbors across the fitness landscape.

Neighbor fitness correlation (NFC) reflects how fitness values are distributed spatially. A strong positive correlation indicates that high-fitness configurations tend to be surrounded by other high-fitness neighbors—and similarly for low-fitness ones—suggesting clustering, separability, and an underlying gradient in the landscape. In contrast, a weak or negative correlation implies that high- and low-fitness configurations are intermixed, signaling high fitness variability and a rugged, unstructured landscape.



!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object. 

    **auto_calculate : *bool, default=True***
    :   If `True`, the function will automatically compute the mean neighbor fitness (via the `landscape.neighbor_fitness` property) if it hasn't been pre-calculated. If `False`, it will raise a `RuntimeError` if these metrics are missing.

    **method : *Literal["pearson", "spearman", "kendall"], default='pearson'***
    :   The statistical method to use for calculating the correlation:
    
        -   `'pearson'`: Computes Pearson's product-moment correlation coefficient. Assumes data is normally distributed.
    
        -   `'spearman'`: Computes Spearman's rank correlation coefficient. Based on ranked values, suitable for non-normally distributed data and monotonic relationships.
    
        -   `'kendall'`: Computes Kendall's Tau rank correlation coefficient. Also based on ranks, often used for smaller datasets or when there are many tied ranks.


!!! api-returns "Returns"

    **NFC:** ***float***
    :   The neighbor fitness correlation (NFC) of the landscape. Lower and negative value indicates a less structured, rugged landscape with large fitness fluctuations.
