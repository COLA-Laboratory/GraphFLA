"""Isolated benchmark for graphfla.analysis.single_mutation_effects."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "single_mutation_effects"
