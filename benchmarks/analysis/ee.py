"""EE only: 64--1024 vertices, no construction or other metrics in timing."""

from graphfla.analysis import evolvability_effects, evolvability_enhancing_fraction
from ._workloads import EE_CASES, build_case


class EE:
    params = (EE_CASES, ["fraction", "effects"])
    param_names = ["dataset", "output"]
    timeout = 30
    number = 1
    repeat = 5

    def setup(self, dataset, output):
        self.landscape = build_case(dataset)
        self.function = (evolvability_enhancing_fraction if output == "fraction"
                         else evolvability_effects)

    def time_metric(self, dataset, output):
        self.function(self.landscape)

    def peakmem_metric(self, dataset, output):
        self.function(self.landscape)
