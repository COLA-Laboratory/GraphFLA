---
icon: material/cube-outline
---

Landscape navigability, often closely related to peak accessibility, concerns the ease with which evolving populations can traverse a fitness landscape to discover and ascend fitness peaks, particularly those representing high or optimal fitness genotypes. It is a critical property influencing the rate and predictability of adaptation, as it determines whether populations can efficiently find beneficial evolutionary trajectories or become trapped on suboptimal solutions. The fundamental basis of navigability lies in the existence and nature of accessible paths—sequences of mutations where each step confers a non-decreasing, and typically increasing, fitness effect, which natural selection can favor.

In simple terms, it describes whether there are viable mutational routes that an evolving population can follow to reach higher fitness states without having to cross significant fitness valleys, which selection would typically prevent.

## Overview

| API | Purpose |
| --- | --- |
| [`local_optima_accessibility`](#peak-accessibility) | Fraction of genotypes that can reach a given local optimum via monotonic paths. |
| [`global_optima_accessibility`](#peak-accessibility) | Same, but specifically for the global optimum. |
| [`fitness_distance_corr`](#fitness-distance-correlation) | Spearman / Pearson correlation between fitness and distance-to-GO (FDC). |
| [`basin_fit_corr`](#basin-size-fitness-correlation) | Correlation between basin size and the fitness of its local optimum. |
| [`evol_enhance_mutations`](#evolvability-enhancing-mutations) | Fraction of improving edges that also raise mean neighbor fitness (Wagner 2023). |
| [`mean_path_lengths`](#length-of-accessible-paths) | Mean/variance of shortest path lengths from variants to specified peaks. |
| [`mean_path_lengths_go`](#length-of-accessible-paths) | Same, targeting the global optimum. |
| [`mean_dist_lo`](#length-of-accessible-paths) | Mean Hamming/edit distance from all variants to specified peaks. |
| [`mean_dist_go`](#length-of-accessible-paths) | Same, targeting the global optimum. |

## Peak Accessibility

```api
def graphfla.analysis.local_optima_accessibility(landscape, lo: Union[int, List[int]]) -> Union[float, List[float]]
```

Calculates the accessibility of one or more specified peak(s) in the fitness landscape.

This metric quantifies the proportion of all genotypes in the landscape that can reach a specified peak (or set of peaks) by following any path of monotonically increasing fitness (i.e., adaptive walks). It is equivalent to the size of the peak’s basin of attraction divided by the total number of genotypes in the landscape. In other words, it reflects how many variants fall within the basin of attraction of the given peak(s).

A higher accessibility indicates that more variants can access the peak(s) during evolution, whereas a low value indicates that the peak(s) is hardly accessible. 

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object.

    **lo : *Union[int, List[int]]***
    :   The index of the peak to analyze, or a list of indices if analyzing multiple peaks. Each index must correspond to a genotype that is a true peak (i.e., `is_lo=True`).


!!! api-returns "Returns"

    **Union[float, List[float]]**
    :   -   If `lo` is a single integer: Returns a single float value (between 0.0 and 1.0) representing the accessibility.
    
        -   If `lo` is a list of integers: Returns a list of floats, where each float corresponds to the accessibility of the peak at the respective index in the input list `lo`.


```api
def graphfla.analysis.global_optima_accessibility(landscape) -> float
```

Calculates the accessibility of the global peak in the fitness landscape.

This metric quantifies the proportion of all genotypes in the landscape that can reach the global peak via any path of monotonically increasing fitness (i.e., adaptive walks). It is equivalent to the size of the gloal peak’s basin of attraction divided by the total number of genotypes in the landscape. In other words, it reflects how many variants fall within the basin of attraction of the global peak.

Since the global peak is the utmost goal of evolution, its accessibility can reflect the navigability of whole landscape. 

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object. 


!!! api-returns "Returns"

    **float**
    :   The fraction of genotypes (a value between 0.0 and 1.0) from which the global peak can be reached through monotonic, fitness-improving paths.


## Fitness Distance Correlation

```api
def graphfla.analysis.fitness_distance_corr(landscape, method: str = "spearman") -> float
```

Calculates the Fitness Distance Correlation (FDC) of a landscape. 

This metric measures the navigability of a fitness landscape by quantifying the correlation between the fitness values of variants and their respective distances to the global peak. It assesses how informative the fitness landscape is in guiding the evolution towards prominent regions. A landscape where fitness reliably increases as evolution approaches the global peak (indicated by a strong negative FDC) is generally considered easier to navigate than one where the relationship is weak, random, or misleading (indicated by an FDC near zero or positive).

!!! note "Automatic Distance Calculation"
    The function automatically attempts to calculate the distance to the global peak for each genotype if it's not already present in the landscape data . This will add new attributes for genotypes (`"dist_go"`) that can be accessed via `Landscape.graph` or `Landscape.get_data()`.
!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object. 

    **method : *{'spearman', 'pearson'}, default='spearman'***
    :   The statistical correlation measure to be used for assessing FDC:
    
        -   `'spearman'`: Calculates Spearman's rank correlation coefficient.
    
        -   `'pearson'`: Calculates Pearson correlation coefficient.


!!! api-returns "Returns"

    **float**
    :   The calculated Fitness Distance Correlation coefficient. This value ranges from -1 to 1.
    
        -   A value close to -1 indicates that fitness tends to increase as the distance to the global peak *decreases*, corresponding to a highly navigable landscape.
    
        -   A value close to 0 implies a more rugged, neutral, or unpredictable relationship between fitness and distance to the global peak, resulting in reduced navigability. 
    
        -   A value close to +1 indicates fitness tends to decrease as the distance to the global peak *decreases*. This suggests a "deceptive" landscape where regions of higher fitness are generally further away from the actual global peak, making it extremely difficult for evolution to navigate.


**References**

-   Terry Jones and Stephanie Forrest, "Fitness Distance Correlation as a Measure of Problem Difficulty for Genetic Algorithms." In *Proceedings of the Sixth International Conference on Genetic Algorithms (ICGA'95)* (1995).

## Basin Size-Fitness Correlation

```api
def graphfla.analysis.basin_fit_corr(landscape, method: str = "spearman") -> float
```

Calculates the correlation between the size of the basin of attraction and the fitness of peaks.

The size of the basin of attraction of a peak is defined as the total number of variants in the landscape from which the peak is accessible. The size of these basins can reveal important information about the landscape's structure and navigability. A common question is whether larger basins tend to be associated with fitter peaks, which could imply that fitter peaks are easier to be accessed. This function quantifies such a relationship by calculating the correlation between basin sizes and the fitness values of their corresponding peaks.

!!! api-parameters "Parameters"

    **landscape** : ***Landscape***
    :   The fitness landscape object. 

    **method** : ***str, one of {"spearman", "pearson"}, default='spearman'***
    :   The correlation measure to use:
    
        -   `'spearman'`: Computes Spearman's rank correlation coefficient. This method assesses monotonic relationships and is robust to outliers.
    
        -   `'pearson'`: Computes Pearson's product-moment correlation coefficient. This method assesses linear relationships and assumes data is approximately normally distributed.


!!! api-returns "Returns"

    **float**
    :   The correlation coefficient between basin size and the fitness of its local optimum. The correlation coefficient is a value between -1 and 1. A larger value (e.g., close to 1) implies suggests that fitter peaks tend to have larger basin of attraction, indicating a more navigable landscape. 


!!! note "Automatic Basin Calculation"
    If basin sizes are not already computed and stored in the `landscape` object's graph vertex attributes, this function will attempt to calculate them by accessing the `landscape.basins` property.

## Evolvability-enhancing Mutations

```api
def graphfla.analysis.evol_enhance_mutations(landscape: Landscape, epsilon: float = 0, auto_calculate: bool = True) -> float
```

Calculates the prevalence of evolvability-enhancing (EE) mutations within the fitness landscape.

Wagner defines an evolvability-enhancing (EE) mutation as a mutation that creates a genetic background in which subsequent mutations are more likely to be adaptive. Equivalently, an EE mutation increases the average fitness of neighboring genotypes compared to the original background. This function returns the proportion of all fitness-increasing mutations in the landscape that are also EE.

!!! note "Automatic Calculation of Neighbor Fitness"
    If the mean neighbor fitness has not yet been calculated for the landscape, this function will automatically compute it by accessing the `landscape.neighbor_fitness` property. This adds a `'mean_neighbor_fit'` node attribute and a `'delta_mean_neighbor_fit'` edge attribute, both accessible via `Landscape.graph` or `Landscape.get_data()`.

!!! api-parameters "Parameters"

    **landscape : *Landscape***
    :   The fitness landscape object.

    **epsilon : *float, default=0***
    :   A tolerance threshold for detecting significant evolvability enhancement. An edge counts as EE if its `delta_mean_neighbor_fit` strictly exceeds this value.

    **auto_calculate : *bool, default=True***
    :   If `True`, automatically computes neighbour fitness (via the `landscape.neighbor_fitness` property) when the required edge attribute is missing. If `False`, raises `RuntimeError` instead.


!!! api-returns "Returns"

    **float**
    :   The proportion of edges in the landscape that are classified as EE — a value between 0.0 and 1.0.


!!! note "Deprecated alias"
    `graphfla.analysis.calculate_evol_enhance` is a deprecated alias for `evol_enhance_mutations` (issues a `FutureWarning`). New code should use the canonical name.

**References**

- Andreas Wagner, "Evolvability-enhancing mutations in the fitness landscapes of an RNA and a protein." *Nat Commun.* (2023).


## Length of Accessible Paths

```api
def graphfla.analysis.mean_path_lengths(landscape, lo: Union[int, List[int]] = None, accessible: bool = True, n_samples: Optional[Union[int, float]] = None) -> Union[dict, List[dict]]
```

Calculates the mean and variance of the shortest path lengths from variants to specified peaks.

In a perfectly smooth landscape, the length of the shortest accessible path from any one variant to a high fitness peak equals the genetic distance between the variant and the peak. In contrast, in a rugged landscape, even the shortest accessible path may meander through the landscape and thus be much longer than this genetic distance. 

This function quantifies these path lengths, providing insights into the landscape's navigability by measuring the expected adaptive walk steps required to reach peaks. It computes the shortest path length from each variant (or a sample thereof) to one or more target peaks.

!!! api-parameters "Parameters"

    **landscape** : ***Landscape***
    :   The fitness landscape object.

    **lo** : ***Union[int, List[int]], optional***
    :   The index (or list of indices) of the peak(s) to analyze. Each index must correspond to a node that is a valid peak. If `None` (default), the function will use the global peak of the landscape.

    **n_samples** : ***Optional[Union[int, float]], default=None***
    :   Specifies whether to use sampling to approximate the results, which is recommended for large landscapes:
    
        -   If a float between 0.0 (exclusive) and 1.0 (inclusive): Samples this fraction of the total variants.
    
        -   If an int greater than 1: Samples this specific number of variants.
    
        -   If `None`: Computes path lengths for all variants. A warning is issued if the number of variants exceeds 10,000.


!!! api-returns "Returns"

    **Union[dict, List[dict]]**
    :   -   If `lo` is a single integer (or `None`, targeting the global peak): A dictionary containing the `"mean"` and `"variance"` of the shortest path lengths to that optimum.
    
        -   If `lo` is a list of integers: A list of dictionaries, where each dictionary corresponds to a peak in the input list `lo` and contains its respective `"mean"` and `"variance"`.
        Path lengths that are infinite (i.e., the optimum is unreachable from a variant under the given `accessible` constraint) are excluded from the mean and variance calculations. If no finite paths exist to an optimum, `np.nan` will be returned for mean and variance.


!!! warning "Computational Cost on Large Landscapes"
    For landscapes with a large number of variants (e.g., >10,000), calculating all-pairs shortest paths can be very time-consuming and memory-intensive. It is highly recommended to use the `n_samples` parameter to analyze a subset of variants in such cases.

```api
def graphfla.analysis.mean_path_lengths_go(landscape, accessible: bool = True, n_samples: Optional[Union[int, float]] = None) -> float
```

Calculates the mean of the shortest path lengths from variants to the global peak.

This function computes the shortest path length from each variant (or a sample thereof) to the global peak of the landscape. It serves as a convenience wrapper around the more general `mean_path_lengths` function, specifically targeting the global peak. The path lengths provide insights into the landscape's navigability by measuring the expected number of adaptive steps required to reach the global peak.

!!! api-parameters "Parameters"

    **landscape** : ***Landscape***
    :   The fitness landscape object.

    **n_samples** : ***Optional[Union[int, float]]***, default=`None`
    :   Specifies whether to use sampling to approximate the results, recommended for large landscapes:
    
        -   If a float between 0.0 (exclusive) and 1.0 (inclusive): Samples this fraction of the total variants.
    
        -   If an int greater than 1: Samples this specific number of variants.
    
        -   If `None`: Computes path lengths for all variants. A warning similar to that in `mean_path_lengths` applies if the number of variants is large.


!!! api-returns "Returns"

    ***float***
    :   The mean of the shortest path lengths to the global peak. Path lengths that are infinite (i.e., the global peak is unreachable from a variant) are excluded from the mean calculation. If no finite paths exist to the global peak, `np.nan` will be returned.


!!! warning "Computational Cost on Large Landscapes"
    For landscapes with a large number of variants (e.g., >10,000), calculating shortest paths can be very time-consuming and memory-intensive. It is highly recommended to use the `n_samples` parameter to analyze a subset of variants in such cases.


---
```api
def graphfla.analysis.mean_dist_lo(landscape, lo: Union[int, List[int]], distance_func: Optional[Callable] = None) -> Union[float, List[float]]
```

Calculates the mean distance from all variants to one or more specified peak(s).

This function provides a measure of how "far" on average other points in the landscape are from specific peak(s), using a defined distance metric (e.g., Hamming distance or Edit distance). This can be useful for understanding the global structure of the landscape in relation to its peaks.

!!! api-parameters "Parameters"

    **landscape** : ***Landscape***
    :   The fitness landscape object.

    **lo** : ***Union[int, List[int]]***
    :   The index (or a list of indices) of the peak(s) to analyze. Each provided index must correspond to a node in the landscape graph that is a valid peak.

    **distance_func** : ***Optional[Callable]***, default=`None`
    :   A callable function used to calculate the distance between variants. The function should typically accept two variants (or arrays of variants) and landscape data types as input and return a numerical distance or an array of distances. If `None`, the landscape's default distance metric (e.g., Hamming distance) will be used.


!!! api-returns "Returns"

    ***Union[float, List[float]]***
    :   -   If `lo` is a single integer: Returns a float representing the mean distance from all variants in the landscape to the specified peak.
    
        -   If `lo` is a list of integers: Returns a list of floats, where each float is the mean distance to the corresponding peak in the input list `lo`.


---
```api
def graphfla.analysis.mean_dist_go(landscape, distance_func: Optional[Callable] = None) -> float
```

Calculates the mean distance from all variants to the global peak.

This function determines the average distance from every variant in the landscape to its global peak. It can utilize a pre-calculated `'dist_go'` vertex attribute if available; otherwise, it computes these distances using the specified or default distance function. This metric helps characterize the overall spread or compactness of the landscape relative to its highest peak.

!!! api-parameters "Parameters"

    **landscape** : ***Landscape***
    :   The fitness landscape object. 

    **distance_func** : ***Optional[Callable]***, default=`None`
    :   A callable function to calculate distances between variants. If `None`, the landscape's default distance metric is used.


!!! api-returns "Returns"

    ***float***
    :   The mean distance from all variants in the landscape to the global peak.


!!! note "Automatic Distance Calculation"
    The function automatically attempts to calculate the distance to the global peak for each genotype if it's not already present in the landscape data . This will add new attributes for genotypes (`"dist_go"`) that can be accessed via `Landscape.graph` or `Landscape.get_data()`.
