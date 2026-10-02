"""Isolated benchmark for graphfla.analysis.gradient_intensity."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "gradient_intensity"
