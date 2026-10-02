"""Isolated benchmark for graphfla.analysis.idiosyncratic_index."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "idiosyncratic_index"
