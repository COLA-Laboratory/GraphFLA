"""Bounded edge-trend timing and memory by statistic."""

from graphfla import analysis
from ._workloads import TREND_CASES, build_case


class Benchmark:
    params = [TREND_CASES, ["pearson", "spearman", "regression"]]
    param_names = ["dataset", "statistic"]
    timeout = 30
    number = 1
    repeat = 5

    def setup(self, dataset, statistic):
        self.landscape = build_case(dataset)

    def time_metric(self, dataset, statistic):
        analysis.increasing_costs_index(self.landscape, method=statistic)

    def peakmem_metric(self, dataset, statistic):
        analysis.increasing_costs_index(self.landscape, method=statistic)
