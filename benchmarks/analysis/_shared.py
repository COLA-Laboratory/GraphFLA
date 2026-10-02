"""Public analysis functions on bounded synthetic landscapes."""

from functools import partial

from graphfla import analysis as A

from ._workloads import SMALL_CASES, build_case


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


class _MetricBenchmark:
    """Shared harness; each concrete metric has its own importable module."""

    params = SMALL_CASES
    param_names = ["dataset"]
    timeout = 30
    number = 1
    repeat = 5

    def setup(self, dataset):
        landscape = build_case(dataset)
        # Only prepare dependencies needed by the selected metric.
        self.call = prepare_call(landscape, self.method)

    def time_metric(self, dataset):
        self.call()

    def peakmem_metric(self, dataset):
        self.call()
