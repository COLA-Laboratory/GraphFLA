"""Isolated benchmark for graphfla.analysis.global_optima_accessibility."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "global_optima_accessibility"
