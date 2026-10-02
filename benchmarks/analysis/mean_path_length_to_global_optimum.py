"""Isolated benchmark for graphfla.analysis.mean_path_length_to_global_optimum."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "mean_path_length_to_global_optimum"
