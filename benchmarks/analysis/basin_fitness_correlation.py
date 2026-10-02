"""Isolated benchmark for graphfla.analysis.basin_fitness_correlation."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "basin_fitness_correlation"
