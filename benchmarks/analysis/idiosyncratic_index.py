"""Isolated benchmark for graphfla.analysis.idiosyncratic_index."""

from ._shared import _MetricBenchmark
from ._workloads import IDIOSYNCRASY_CASES


class Benchmark(_MetricBenchmark):
    method = "idiosyncratic_index"
    params = IDIOSYNCRASY_CASES
