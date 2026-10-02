"""Isolated benchmark for graphfla.analysis.r_s_ratio."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "r_s_ratio"
