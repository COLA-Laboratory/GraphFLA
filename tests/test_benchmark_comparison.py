"""Comparison gates must reject incomparable runs and tolerate timing outliers."""

from copy import deepcopy

import pytest

from tools.compare_benchmark_rounds import compare


def result(samples):
    return {
        "environment": {"python": "test"},
        "protocol": {"processes": 3},
        "datasets": {
            "small": {
                "input_sha256": "same-input",
                "shape": [4, 4],
                "variable_sites": 2,
                "workers": [{"seconds": worker} for worker in samples],
                "mad_seconds": 0,
                "median_peak_rss_bytes": 1024,
            }
        },
    }


def test_process_medians_resist_one_timing_outlier():
    before = result([[1, 1, 100], [1, 1, 1], [1, 1, 1]])
    after = result([[0.5] * 3] * 3)
    row = compare(before, after)[0]
    assert row["status"] == "faster"
    assert row["speedup"] == 2


def test_small_effect_is_inconclusive():
    assert (
        compare(result([[1] * 3] * 3), result([[0.98] * 3] * 3))[0]["status"]
        == "inconclusive"
    )


@pytest.mark.parametrize(
    "key,value",
    [("input_sha256", "different"), ("shape", [4, 3]), ("variable_sites", 3)],
)
def test_incompatible_graph_or_input_is_rejected(key, value):
    before = result([[1] * 3] * 3)
    after = deepcopy(before)
    after["datasets"]["small"][key] = value
    with pytest.raises(ValueError, match="mismatch"):
        compare(before, after)


def test_analysis_snapshots_check_labels_shapes_dtypes_and_float_values(tmp_path):
    import numpy as np
    from tools.benchmark_analysis import compare_snapshots

    baseline, candidate = tmp_path / "baseline.npz", tmp_path / "candidate.npz"
    source = {"p": np.array([0.1, np.nan]), "is_ee": np.array([1, -1], dtype=np.int8)}
    np.savez(baseline, **source)
    np.savez(candidate, **{**source, "p": np.array([0.1 + 1e-13, np.nan])})
    assert set(compare_snapshots(baseline, candidate)) == set(source)
    for changed in [
        {**source, "p": np.array([0.11, np.nan])},
        {**source, "p": np.array([0.1, 0.2])},
        {**source, "p": np.array([[0.1, np.nan]])},
        {**source, "is_ee": np.array([0, -1], dtype=np.int8)},
        {**source, "is_ee": np.array([1, -1], dtype=np.int64)},
        {"p": source["p"]},
    ]:
        np.savez(candidate, **changed)
        with pytest.raises(AssertionError):
            compare_snapshots(baseline, candidate)
