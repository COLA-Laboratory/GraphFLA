"""Public analysis functions on fixed inputs with prepared landscape caches."""

from functools import partial

from graphfla import analysis as A

from ._datasets import build_dataset

# Every exported function is represented; result dataclasses are not workloads.
METHODS = [
    name
    for name in A.__all__
    if name
    not in {
        "EpistasisClassification",
        "ExtradimensionalBypass",
    }
]


def prepare_call(landscape, method):
    """Bind bounded, deterministic parameters outside the timed region."""
    if method == "list_metrics":
        return A.list_metrics
    kwargs = {}
    if method in {
        "gamma",
        "gamma_star",
        "higher_order_epistasis",
        "global_idiosyncratic_index",
        "profile",
    }:
        kwargs["n_jobs"] = 1
    if method in {
        "autocorrelation",
        "global_idiosyncratic_index",
        "classify_epistasis",
        "extradimensional_bypass",
        "mean_path_length_to_global_optimum",
        "mean_path_length_to_local_optima",
        "profile",
    }:
        kwargs["seed"] = 0
    if method in {"classify_epistasis", "extradimensional_bypass"}:
        kwargs["sample_cut_prob"] = 0.5
    if method in {
        "local_optima_accessibility",
        "mean_distance_to_local_optima",
        "mean_path_length_to_local_optima",
    }:
        kwargs["lo"] = landscape.lo_index[:4]
    if method in {
        "mean_path_length_to_local_optima",
        "mean_path_length_to_global_optimum",
    }:
        kwargs["n_samples"] = 64
    if method in {
        "fitness_effect_distribution",
        "idiosyncratic_index",
        "single_mutation_effects",
    }:
        position = next(iter(landscape.data_types))
        alleles = sorted(set(landscape.graph.vs[position]))
        if method == "single_mutation_effects":
            kwargs["position"] = position
        else:
            kwargs["mutation"] = (alleles[0], position, alleles[1])
    if method == "profile":
        kwargs.update(
            include=["local_optima_ratio", "gradient_intensity", "fdc"],
            on_error="raise",
        )
    return partial(getattr(A, method), landscape, **kwargs)


class Analysis:
    params = (["CR6261", "TrpB3I", "synthetic-rna", "synthetic-hpo"], METHODS)
    param_names = ["dataset", "method"]
    timeout = 120
    repeat = 5

    def setup(self, dataset, method):
        landscape = build_dataset(dataset)
        for name in ("basins", "dist_to_go", "neighbor_fitness", "accessible_paths"):
            getattr(landscape, name)
        self.call = prepare_call(landscape, method)

    def time_method(self, dataset, method):
        self.call()

    def peakmem_method(self, dataset, method):
        self.call()
