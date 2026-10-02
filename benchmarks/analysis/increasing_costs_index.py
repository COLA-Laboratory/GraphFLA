"""Isolated benchmark for graphfla.analysis.increasing_costs_index."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "increasing_costs_index"
