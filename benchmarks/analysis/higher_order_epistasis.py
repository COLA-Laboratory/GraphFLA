"""Isolated benchmark for graphfla.analysis.higher_order_epistasis."""

from functools import partial

from graphfla.analysis import higher_order_epistasis, walsh_hadamard
from ._shared import _MetricBenchmark
from ._workloads import WALSH_CASES, build_case


class Benchmark(_MetricBenchmark):
    params = [WALSH_CASES, ["fit", "reuse"]]
    param_names = ["dataset", "source"]
    method = "higher_order_epistasis"

    def setup(self, dataset, source):
        landscape = build_case(dataset)
        if source == "reuse":
            landscape = walsh_hadamard(landscape)
        self.call = partial(higher_order_epistasis, landscape)

    def time_metric(self, dataset, source):
        self.call()

    def peakmem_metric(self, dataset, source):
        self.call()
