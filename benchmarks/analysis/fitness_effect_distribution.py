"""Isolated benchmark for graphfla.analysis.fitness_effect_distribution."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "fitness_effect_distribution"
