"""Isolated benchmark for graphfla.analysis.list_metrics."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "list_metrics"
