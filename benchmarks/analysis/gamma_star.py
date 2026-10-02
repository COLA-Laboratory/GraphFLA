"""Isolated benchmark for graphfla.analysis.gamma_star."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "gamma_star"
