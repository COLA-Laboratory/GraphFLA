"""Isolated benchmark for graphfla.analysis.all_mutation_effects."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "all_mutation_effects"
