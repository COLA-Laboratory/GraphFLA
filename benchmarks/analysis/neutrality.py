"""Isolated benchmark for graphfla.analysis.neutrality."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "neutrality"
