"""Isolated benchmark for graphfla.analysis.gamma."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "gamma"
