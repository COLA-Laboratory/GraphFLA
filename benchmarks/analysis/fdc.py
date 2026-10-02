"""Isolated benchmark for graphfla.analysis.fdc."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "fdc"
