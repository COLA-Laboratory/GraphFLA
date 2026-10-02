"""Isolated benchmark for graphfla.analysis.gamma_star."""

from ._shared import _MetricBenchmark
from ._workloads import GAMMA_CASES


class Benchmark(_MetricBenchmark):
    params = GAMMA_CASES
    method = "gamma_star"
