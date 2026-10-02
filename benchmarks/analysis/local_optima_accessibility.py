"""Isolated benchmark for graphfla.analysis.local_optima_accessibility."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "local_optima_accessibility"
