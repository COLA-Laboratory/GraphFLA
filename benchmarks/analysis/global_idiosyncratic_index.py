"""Isolated benchmark for graphfla.analysis.global_idiosyncratic_index."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "global_idiosyncratic_index"
