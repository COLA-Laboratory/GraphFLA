"""Isolated benchmark for graphfla.analysis.neighbor_fitness_correlation."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "neighbor_fitness_correlation"
