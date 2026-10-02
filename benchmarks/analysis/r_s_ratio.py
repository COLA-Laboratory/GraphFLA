"""Isolated benchmark for graphfla.analysis.r_s_ratio."""

from ._shared import _MetricBenchmark
from ._workloads import RS_CASES


class Benchmark(_MetricBenchmark):
    params = RS_CASES
    method = "r_s_ratio"
