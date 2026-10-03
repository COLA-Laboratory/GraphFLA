---
api_narrative: true
---

Landscape navigability, often closely related to peak accessibility, concerns the ease with which evolving populations can traverse a fitness landscape to discover and ascend fitness peaks, particularly those representing high or optimal fitness genotypes. It is a critical property influencing the rate and predictability of adaptation, as it determines whether populations can efficiently find beneficial evolutionary trajectories or become trapped on suboptimal solutions. The fundamental basis of navigability lies in the existence and nature of accessible paths—sequences of mutations where each step confers a non-decreasing, and typically increasing, fitness effect, which natural selection can favor.

In simple terms, it describes whether there are viable mutational routes that an evolving population can follow to reach higher fitness states without having to cross significant fitness valleys, which selection would typically prevent.

## Overview

| API | Purpose |
| --- | --- |
| [`local_optima_accessibility`](#peak-accessibility) | Fraction of genotypes that can reach a given local optimum via monotonic paths. |
| [`global_optima_accessibility`](#peak-accessibility) | Same, but specifically for the global optimum. |
| [`fdc`](#fitness-distance-correlation) | Spearman / Pearson correlation between fitness and distance-to-GO (FDC). |
| [`basin_fitness_correlation`](#basin-size-fitness-correlation) | Correlation between basin size and the fitness of its local optimum. |
| [`evolvability_enhancing_fraction`](#evolvability-enhancing-mutations) | Fraction of observed directed mutations with a statistically supported EE effect. |
| [`mean_path_length_to_local_optima`](#length-of-accessible-paths) | Mean/variance of shortest path lengths from variants to specified peaks. |
| [`mean_path_length_to_global_optimum`](#length-of-accessible-paths) | Same, targeting the global optimum. |
| [`mean_distance_to_local_optima`](#length-of-accessible-paths) | Mean Hamming/edit distance from all variants to specified peaks. |
| [`mean_distance_to_global_optimum`](#length-of-accessible-paths) | Same, targeting the global optimum. |

## Peak Accessibility

Calculates the accessibility of one or more specified peak(s) in the fitness landscape.

This metric quantifies the proportion of all genotypes in the landscape that can reach a specified peak (or set of peaks) by following any path of monotonically increasing fitness (i.e., adaptive walks). It is equivalent to the size of the peak’s basin of attraction divided by the total number of genotypes in the landscape. In other words, it reflects how many variants fall within the basin of attraction of the given peak(s).

A higher accessibility indicates that more variants can access the peak(s) during evolution, whereas a low value indicates that the peak(s) is hardly accessible.

::: graphfla.analysis.local_optima_accessibility

Calculates the accessibility of the global peak in the fitness landscape.

This metric quantifies the proportion of all genotypes in the landscape that can reach the global peak via any path of monotonically increasing fitness (i.e., adaptive walks). It is equivalent to the size of the selected global peak’s basin of attraction divided by the total number of genotypes in the landscape. In other words, it reflects how many variants fall within the basin of attraction of the global peak.

Since the global peak is the utmost goal of evolution, its accessibility can reflect the navigability of whole landscape.

::: graphfla.analysis.global_optima_accessibility

## Fitness Distance Correlation

Calculates the Fitness Distance Correlation (FDC) of a landscape. 

This metric measures the navigability of a fitness landscape by quantifying the correlation between the fitness values of variants and their respective distances to the global peak. It assesses how informative the fitness landscape is in guiding the evolution towards prominent regions. A landscape where fitness reliably increases as evolution approaches the global peak (indicated by a strong negative FDC) is generally considered easier to navigate than one where the relationship is weak, random, or misleading (indicated by an FDC near zero or positive).

!!! note "Automatic Distance Calculation"
    The function automatically attempts to calculate the distance to the global peak for each genotype if it's not already present in the landscape data . This will add new attributes for genotypes (`"dist_go"`) that can be accessed via `Landscape.graph` or `Landscape.get_data()`.

::: graphfla.analysis.fdc

## Basin Size-Fitness Correlation

Calculates the correlation between the size of the basin of attraction and the fitness of peaks.

The size of the basin of attraction of a peak is defined as the total number of variants in the landscape from which the peak is accessible. The size of these basins can reveal important information about the landscape's structure and navigability. A common question is whether larger basins tend to be associated with fitter peaks, which could imply that fitter peaks are easier to be accessed. This function quantifies such a relationship by calculating the correlation between basin sizes and the fitness values of their corresponding peaks.

::: graphfla.analysis.basin_fitness_correlation

--8<-- "ee-mutations.md"

::: graphfla.analysis.evolvability_enhancing_fraction
    options:
      skip_local_inventory: true

See [per-mutation EE results](robustness.md#per-mutation-ee-results) for the detailed output.

## Length of Accessible Paths

Calculates the mean and variance of the shortest path lengths from variants to specified peaks.

In a perfectly smooth landscape, the length of the shortest accessible path from any one variant to a high fitness peak equals the genetic distance between the variant and the peak. In contrast, in a rugged landscape, even the shortest accessible path may meander through the landscape and thus be much longer than this genetic distance. 

This function quantifies these path lengths, providing insights into the landscape's navigability by measuring the expected adaptive walk steps required to reach peaks. It computes the shortest path length from each variant (or a sample thereof) to one or more target peaks.

::: graphfla.analysis.mean_path_length_to_local_optima

Calculates the mean of the shortest path lengths from variants to the global peak.

This function computes the shortest path length from each variant (or a sample thereof) to the global peak of the landscape. It serves as a convenience wrapper around the more general `mean_path_length_to_local_optima` function, specifically targeting the global peak. The path lengths provide insights into the landscape's navigability by measuring the expected number of adaptive steps required to reach the global peak.

::: graphfla.analysis.mean_path_length_to_global_optimum

Calculates the mean distance from all variants to one or more specified peak(s).

This function provides a measure of how "far" on average other points in the landscape are from specific peak(s), using a defined distance metric (e.g., Hamming distance or Edit distance). This can be useful for understanding the global structure of the landscape in relation to its peaks.

::: graphfla.analysis.mean_distance_to_local_optima

Calculates the mean distance from all variants to the global peak.

This function determines the average distance from every variant in the landscape to its global peak. It can utilize a pre-calculated `'dist_go'` vertex attribute if available; otherwise, it computes these distances using the specified or default distance function. This metric helps characterize the overall spread or compactness of the landscape relative to its highest peak.

::: graphfla.analysis.mean_distance_to_global_optimum
