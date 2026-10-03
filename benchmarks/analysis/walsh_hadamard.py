"""Isolated benchmark for graphfla.analysis.walsh_hadamard."""

from functools import partial

from graphfla.analysis import walsh_hadamard
from ._shared import _MetricBenchmark
from ._workloads import WALSH_CASES, build_case


class Benchmark(_MetricBenchmark):
    params = [WALSH_CASES, ["ols", "lasso", "lasso_cv"]]
    param_names = ["dataset", "estimator"]
    method = "walsh_hadamard"

    def setup(self, dataset, estimator):
        self.call = partial(
            walsh_hadamard, build_case(dataset), **fit_options(estimator)
        )

    def time_metric(self, dataset, estimator):
        self.call()

    def peakmem_metric(self, dataset, estimator):
        self.call()


def fit_options(estimator):
    return dict(
        max_order=2,
        max_cells=1e6,
        method="ols" if estimator == "ols" else "lasso",
        alpha="cv" if estimator == "lasso_cv" else 0.01,
        cv=3,
        random_state=0,
    )
