"""Isolated benchmark for graphfla.analysis.local_optima_ratio."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "local_optima_ratio"
