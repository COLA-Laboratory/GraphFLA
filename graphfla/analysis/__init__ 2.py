# graphfla/analysis/__init__.py

"""
Methods for fitness landscape analysis.

TODO:
1. (Finished) Fitness distribution: skwenness, kurtosis, coefficient of variation (CV), etc.
2. (Doubting) Evolvability index.
3. (Now!) Fraction of accessible shortest paths.
4. (Finished) Compare mean path length to hamming distance.
5. (Later) NC paper on calculating fitness.
7. (Now!) DFE.
8. [Optional] Across environments: fitness (rank) correlation, change in landscape structure (e.g., all those metrics), and specifically, epistasis (e.g., sign epistasis).
9. (Computationally hard) The gamma statistics. See Supp of "On the (un)predictability of a large intragenic fitness landscape".
10. (Now!) DR and IC can also be calculated via slope.
11. (Plotting) Global epistasis.
12. Epistasis decomposition
13. (Idea) Correlation of fraction/number of beneficial mutations available with the current fitness.

Pairwise Epistasis (ε, ωij, βij, Interaction Score, E)
Measures: The extent to which the fitness effect of two mutations together deviates from their expected combined effect (non-additivity/non-multiplicativity). Key determinant of landscape ruggedness.
Calculation:
Additive model deviation: ε_A = f_AB - (f_A + f_B) (relative to WT=0) or ε_A = f_AB - f_A - f_B + f_WT (general). Used when fitness is on an additive scale (e.g., log fitness, growth rate). (Phillips, Lunzer, Tekin/Dowell analysis)
Multiplicative model deviation (log scale): log(ε_M) = log(f_AB) - (log(f_A) + log(f_B)) (relative to WT=1) or log(ε_M) = log(f_AB) + log(f_WT) - log(f_A) - log(f_B) (general). Used when fitness is on a multiplicative scale (e.g., relative frequencies, raw fitness). (Li, Bonhoeffer, Hinkley, Phillips)
Chimeric model deviation: ε_C = f_AB - f_A * f_B (relative to WT=1). Note: Mathematically inconsistent for higher orders (Skwara et al. 2025). (Khan, Costanzo x2, Kuzmin/Tekin cited in Skwara).
Regression coefficients: The coefficient (βij) for the interaction term (x_i * x_j) in a linear or log-linear regression model fitted to fitness data. (Bakerlee, Puchta, Hinkley, Otwinowski, Tonner, Skwara, Buda, Johnston)
Background-averaged: Using Walsh-Hadamard transforms or averaging pairwise deviations across multiple backgrounds. (Poelwijk 2019, Domingo, Buda)
"""

"""
TODO:
1. Add support for neutrality (thus neutral networks)
2. Add native support for multi-objective 
3. Add support for additional attributes
4. Try speed up neighbor generation using C
5. Try speed up with parallelization
6. Add support for automatically imputing to combinatorially complete
7. Support Papkou-like graphfilter. 
"""

from .correlation import fitness_distance_corr, ffi, basin_fit_corr, neighbor_fit_corr
from .fitness import fitness_distribution, distribution_fit_effects
from .ruggedness import (
    lo_ratio,
    autocorrelation,
    r_s_ratio,
    gradient_intensity,
)
from .navigability import (
    global_optima_accessibility,
    local_optima_accessibility,
    mean_path_lengths,
    mean_path_lengths_go,
    mean_dist_lo,
)
from .robustness import (
    neutrality,
    single_mutation_effects,
    all_mutation_effects,
    calculate_evol_enhance,
)
from .epistasis import (
    higher_order_epistasis,
    classify_epistasis,
    idiosyncratic_index,
    global_idiosyncratic_index,
    diminishing_returns_index,
    increasing_costs_index,
    gamma_statistic,
)

__all__ = [
    "higher_order_epistasis",
    "fitness_distance_corr",
    "ffi",
    "basin_fit_corr",
    "neighbor_fit_corr",
    "fitness_distribution",
    "distribution_fit_effects",
    "lo_ratio",
    "autocorrelation",
    "r_s_ratio",
    "gradient_intensity",
    "calculate_evol_enhance",
    "classify_epistasis",
    "idiosyncratic_index",
    "global_idiosyncratic_index",
    "diminishing_returns_index",
    "increasing_costs_index",
    "gamma_statistic",
    "global_optima_accessibility",
    "local_optima_accessibility",
    "mean_dist_lo",
    "mean_path_lengths",
    "mean_path_lengths_go",
    "neutrality",
    "single_mutation_effects",
    "all_mutation_effects",
]
