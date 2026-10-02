"""Isolated benchmark for graphfla.analysis.profile."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "profile"
