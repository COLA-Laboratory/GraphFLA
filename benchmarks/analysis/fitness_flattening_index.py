"""Isolated benchmark for graphfla.analysis.fitness_flattening_index."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "fitness_flattening_index"
