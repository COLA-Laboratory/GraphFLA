# graphfla/analysis/__init__.py

"""
Methods for fitness landscape analysis.

TODO:
1. (Now!) Fitness distribution: skwenness, kurtosis, coefficient of variation (CV), etc.
2. (Doubting) Evolvability index.
3. (Now!) Fraction of accessible shortest paths.
4. (Finished) Compare mean path length to hamming distance.
5. (Later) NC paper on calculating fitness.
7. (Now!) DFE.
8. [Optional] Across environments: fitness (rank) correlation, change in landscape structure (e.g., all those metrics), and specifically, epistasis (e.g., sign epistasis).
9. (Computationally hard) The gamma statistics. See Supp of "On the (un)predictability of a large intragenic fitness landscape".
10. (Now!) DR and IC can also be calculated via slope.

We detected the presence of global epistasis as a nonlinear dependence between fitness values
and the sum of linear predictors of the first order (72). To this end, we estimated first-order
additive effects of each allele for each position using a linear regression model. We sum the first
order effects and mapped them to fitness values of corresponding variants via a nonlinear
monotonically increasing function. In particular, we used I-splines basis functions (72).

Fitness Effect Correlation (FEC): This term is less a distinct biological phenomenon and more a quantitative measure or observation used to assess DR and IC. It refers specifically to the statistical correlation between the fitness effect of a mutation (or a set of mutations) and the fitness of the genetic background(s) in which those effects are measured.
A negative correlation between the effect of beneficial mutations and background fitness indicates Diminishing Returns.
A negative correlation between the effect of deleterious mutations and background fitness indicates Increasing Costs (because the effect becomes more negative as background fitness increases).
This is implicitly or explicitly the measure used in Papkou et al., Bakerlee et al., Johnson et al., Kryazhimskiy et al., and Khan et al. when plotting mutational effects against background fitness.

"""

from .correlation import fdc, ffi, basin_fit_corr, neighbor_fit_corr
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
    classify_epistasis,
    idiosyncratic_index,
    global_idiosyncratic_index,
    diminishing_returns_index,
    increasing_costs_index,
)

__all__ = [
    "fdc",
    "ffi",
    "basin_fit_corr",
    "neighbor_fit_corr",
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
    "global_optima_accessibility",
    "local_optima_accessibility",
    "mean_dist_lo",
    "mean_path_lengths",
    "mean_path_lengths_go",
    "neutrality",
    "single_mutation_effects",
    "all_mutation_effects",
]
