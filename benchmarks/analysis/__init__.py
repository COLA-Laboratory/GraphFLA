"""Analysis benchmarks, selectable by metric; no all-metric default runner."""

from ._shared import prepare_call

METRIC_MODULES = {
    "all_mutation_effects": "all_mutation_effects",
    "autocorrelation": "autocorrelation",
    "basin_fitness_correlation": "basin_fitness_correlation",
    "classify_epistasis": "classify_epistasis",
    "diminishing_returns_index": "diminishing_returns_index",
    "evolvability_effects": "ee",
    "evolvability_enhancing_fraction": "ee",
    "extradimensional_bypass": "extradimensional_bypass",
    "fdc": "fdc",
    "fitness_distribution": "fitness_distribution",
    "fitness_effect_distribution": "fitness_effect_distribution",
    "fitness_flattening_index": "fitness_flattening_index",
    "gamma": "gamma",
    "gamma_star": "gamma_star",
    "global_idiosyncratic_index": "global_idiosyncratic_index",
    "global_optima_accessibility": "global_optima_accessibility",
    "gradient_intensity": "gradient_intensity",
    "higher_order_epistasis": "higher_order_epistasis",
    "idiosyncratic_index": "idiosyncratic_index",
    "increasing_costs_index": "increasing_costs_index",
    "list_metrics": "list_metrics",
    "local_optima_accessibility": "local_optima_accessibility",
    "local_optima_ratio": "local_optima_ratio",
    "mean_distance_to_global_optimum": "mean_distance_to_global_optimum",
    "mean_distance_to_local_optima": "mean_distance_to_local_optima",
    "mean_path_length_to_global_optimum": "mean_path_length_to_global_optimum",
    "mean_path_length_to_local_optima": "mean_path_length_to_local_optima",
    "neighbor_fitness_correlation": "neighbor_fitness_correlation",
    "neutrality": "neutrality",
    "profile": "profile",
    "r_s_ratio": "r_s_ratio",
    "single_mutation_effects": "single_mutation_effects",
    "walsh_hadamard": "walsh_hadamard",
}
EXCLUDED_METHODS = {
    "evolvability_enhancing_mutations": "Deprecated EE compatibility wrapper; canonical EE benchmarks cover the shared calculation."
}
METHODS = tuple(METRIC_MODULES)
