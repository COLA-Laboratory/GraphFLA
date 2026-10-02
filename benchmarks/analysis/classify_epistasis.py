"""Isolated benchmark for graphfla.analysis.classify_epistasis."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "classify_epistasis"
