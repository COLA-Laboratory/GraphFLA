"""Isolated benchmark for graphfla.analysis.global_idiosyncratic_index."""

from ._shared import _MetricBenchmark
from ._workloads import IDIOSYNCRASY_CASES


class Benchmark(_MetricBenchmark):
    method = "global_idiosyncratic_index"
    params = IDIOSYNCRASY_CASES
