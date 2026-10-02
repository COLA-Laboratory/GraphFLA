"""Isolated benchmark for graphfla.analysis.diminishing_returns_index."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "diminishing_returns_index"
