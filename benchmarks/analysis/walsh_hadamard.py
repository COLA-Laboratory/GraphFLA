"""Isolated benchmark for graphfla.analysis.walsh_hadamard."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "walsh_hadamard"
