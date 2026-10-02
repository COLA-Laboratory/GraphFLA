"""Isolated benchmark for graphfla.analysis.higher_order_epistasis."""

from ._shared import _MetricBenchmark


class Benchmark(_MetricBenchmark):
    method = "higher_order_epistasis"
