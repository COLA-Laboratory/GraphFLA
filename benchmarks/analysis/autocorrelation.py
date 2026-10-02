"""Isolated benchmark for graphfla.analysis.autocorrelation."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "autocorrelation"
