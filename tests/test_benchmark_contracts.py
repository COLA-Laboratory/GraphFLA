"""Keep benchmark inputs reproducible and the public analysis inventory complete."""

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from benchmarks._datasets import DATASETS, DMS, build_dataset, load_dataset
from benchmarks.analysis import METHODS, prepare_call
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
    assert set(METHODS) == functions
    assert len(METHODS) == len(functions)


@pytest.fixture(scope="module")
def benchmark_landscape():
    return build_dataset("synthetic-rna")


@pytest.mark.parametrize("method", METHODS)
def test_analysis_benchmark_calls_public_api(benchmark_landscape, method):
    prepare_call(benchmark_landscape, method)()
