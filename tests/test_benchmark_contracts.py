"""Keep benchmark inputs reproducible and the public analysis inventory complete."""

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from benchmarks._datasets import DATASETS, DMS, load_dataset
from benchmarks.analysis import METHODS, METRIC_MODULES, EXCLUDED_METHODS, prepare_call
from benchmarks.analysis._workloads import (
    SMALL_CASES,
    IDIOSYNCRASY_CASES,
    build_case,
    cases_for_metric,
)
from importlib import import_module
from graphfla import analysis


@pytest.mark.parametrize("dataset", DATASETS)
def test_benchmark_dataset_contract(dataset):
    cls, X, fitness, kwargs = load_dataset(dataset)
    assert len(X) == len(fitness) > 1
    assert np.isfinite(fitness).all()
    if dataset in DMS:
        spec = DMS[dataset]
        assert len(X) == spec["n_variants"]
        assert set(X.str.len()) == {spec["sequence_length"]}
        path = Path(__file__).resolve().parents[1] / "benchmarks/data" / spec["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
        assert X.is_unique


def test_benchmark_analysis_inventory():
    functions = {
        name for name in analysis.__all__ if inspect.isfunction(getattr(analysis, name))
    }
    assert set(METHODS).isdisjoint(EXCLUDED_METHODS)
    assert set(METHODS) | set(EXCLUDED_METHODS) == functions
    assert len(METHODS) == len(set(METHODS))
    for method, module in METRIC_MODULES.items():
        assert import_module(f"benchmarks.analysis.{module}")
    assert all(EXCLUDED_METHODS.values())


@pytest.fixture(scope="module", params=SMALL_CASES)
def benchmark_landscape(request):
    return build_case(request.param)


@pytest.mark.parametrize("method", METHODS)
def test_analysis_benchmark_calls_public_api(benchmark_landscape, method):
    prepare_call(benchmark_landscape, method)()


def test_asv_discovers_only_concrete_metric_benchmarks():
    from asv_runner.discovery import disc_benchmarks

    root = Path(__file__).resolve().parents[1] / "benchmarks"
    names = [benchmark.name for benchmark in disc_benchmarks(str(root))]
    assert len(names) == len(set(names))
    assert not any("MetricBenchmark" in name for name in names)
    for module in set(METRIC_MODULES.values()):
        prefix = f"analysis.{module}."
        assert len([name for name in names if name.startswith(prefix)]) == 2
    assert any(name.startswith("construction.build.") for name in names)


@pytest.mark.parametrize("case", IDIOSYNCRASY_CASES)
def test_idiosyncrasy_workload_bounds_and_shapes(case):
    ls = build_case(case)
    assert 2 <= ls.n_configs <= 1024 and 1 <= len(ls.data_types) <= 72
    assert np.isfinite(ls.graph.vs["fitness"]).all()
    if case == "long-boolean-72":
        assert (ls.n_configs, len(ls.data_types)) == (560, 72)
    for method in ("idiosyncratic_index", "global_idiosyncratic_index"):
        assert cases_for_metric(method) == IDIOSYNCRASY_CASES
        benchmark = import_module(f"benchmarks.analysis.{method}").Benchmark
        assert benchmark.params == IDIOSYNCRASY_CASES
