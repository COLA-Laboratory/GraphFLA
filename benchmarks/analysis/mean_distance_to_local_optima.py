"""Isolated benchmark for graphfla.analysis.mean_distance_to_local_optima."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "mean_distance_to_local_optima"
