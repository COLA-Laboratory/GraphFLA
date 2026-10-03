"""Analysis API contracts for reproducibility, result identity and overrides."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from graphfla import analysis
from graphfla.landscape import BooleanLandscape, Landscape


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_single_mutation_seed_matches_an_independent_control(seed):
    rows = [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    fitness = np.array([0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 6.0])
    landscape = BooleanLandscape().build_from_data(rows, fitness, verbose=False)
    controls = np.random.RandomState(seed).choice(fitness, (4, 2), replace=True)
    expected = np.std([1, 1, 2, 3]) / np.std(controls[:, 1] - controls[:, 0])
    before = np.random.get_state()
    actual = analysis.idiosyncratic_index(landscape, (0, "bit_0", 1), 3, seed=seed)
    after = np.random.get_state()
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected)
    assert actual == analysis.idiosyncratic_index(
        landscape, (0, "bit_0", 1), min_pairs=3, seed=seed
    )
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_single_mutation_rejects_invalid_seed(seed):
    landscape = BooleanLandscape().build_from_data(
        ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False
    )
    with pytest.raises(ValueError):
        analysis.idiosyncratic_index(landscape, (0, "bit_0", 1), seed=seed)


def test_mutation_summary_rows_retain_their_position_and_existing_statistics():
    data = pd.DataFrame({"first": [0, 0, 1, 1], "second": [0, 1, 0, 1]})
    landscape = Landscape().build_from_data(
        data, [0, 1, 2, 4], data_types=dict.fromkeys(data, "categorical"),
        verbose=False,
    )
    actual = analysis.all_mutation_effects(landscape)
    assert actual[["position", "mutation_from", "mutation_to", "mean_effect"]].values.tolist() == [
        ["first", 0, 1, 2.5], ["second", 0, 1, 1.5],
    ]
    assert actual.p_value.tolist() == [0.25, 0.25]
    assert actual.significant.tolist() == [False, False]
    np.testing.assert_allclose(
        actual.median_abs_effect, np.array([2.5, 1.5]) / np.std([0, 1, 2, 4], ddof=1)
    )
    for position in data:
        pd.testing.assert_frame_equal(
            analysis.single_mutation_effects(landscape, position),
            actual.loc[actual.position == position].reset_index(drop=True),
        )


@pytest.mark.parametrize("with_position", [True, False])
def test_empty_mutation_summary_keeps_its_schema(with_position):
    # A retained-data view can have no variable positions or no allele pairs.
    data = pd.DataFrame({"fitness": [1.0, 2.0], "fixed": [0, 0]})
    landscape = SimpleNamespace(
        get_data=lambda: data,
        data_types={"fixed": "categorical"} if with_position else {},
    )
    actual = analysis.all_mutation_effects(landscape)
    assert actual.empty
    assert list(actual) == [
        "mutation_from", "mutation_to", "median_abs_effect", "mean_effect",
        "p_value", "significant", "position",
    ]
    for column in ("median_abs_effect", "mean_effect", "p_value"):
        assert actual[column].dtype == np.dtype(float)
    assert actual.significant.dtype == np.dtype(bool)
    if with_position:
        pd.testing.assert_frame_equal(
            actual, analysis.single_mutation_effects(landscape, "fixed")
        )


@pytest.mark.parametrize("cached", [True, False])
def test_explicit_distance_function_is_respected_with_or_without_cache(cached):
    landscape = BooleanLandscape().build_from_data(
        ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False
    )
    if cached:
        _ = landscape.dist_to_go
    attributes_before = set(landscape.graph.vs.attributes())

    def weighted_distance(configs, target, data_types):
        assert len(data_types) == 2
        return np.sum((configs != target) * [2, 4], axis=1)

    # Distances to 11 are 6, 2, 4 and 0; their mean is 3.
    assert analysis.mean_distance_to_global_optimum(
        landscape, distance_func=weighted_distance
    ) == 3.0
    assert analysis.mean_distance_to_local_optima(
        landscape, lo=3, distance_func=weighted_distance
    ).mean_distance.tolist() == [3.0]
    assert set(landscape.graph.vs.attributes()) == attributes_before
    if cached:
        assert landscape.graph.vs["dist_go"] == [2, 1, 1, 0]
        assert analysis.mean_distance_to_global_optimum(landscape) == 1.0
