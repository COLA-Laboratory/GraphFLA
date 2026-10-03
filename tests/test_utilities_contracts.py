"""Small exact checks for public filters, distances, sampling and walks."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from graphfla.algorithms import HillClimb, RandomWalk, SearchCache
from graphfla.distances import hamming_distance, manhattan_distance, mixed_distance
from graphfla.filters import LandscapeFilter
from graphfla.landscape import BooleanLandscape
from graphfla.sampling import grid_search


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int64, np.float64])
def test_distances_preserve_signed_differences(dtype):
    X = np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=dtype)
    x = np.array([2, 2, 0], dtype=dtype)
    np.testing.assert_array_equal(hamming_distance(X, x), [2, 2, 2])
    np.testing.assert_array_equal(manhattan_distance(X, x), [3, 3, 2])
    np.testing.assert_array_equal(
        mixed_distance(X, x, {"a": "ordinal", "b": "categorical", "c": "boolean"}),
        [3, 2, 2],
    )


@pytest.mark.parametrize(
    "operation,value,expected",
    [
        (">", 1, [2, 3]),
        (">=", 1, [1, 2, 3]),
        ("<", 2, [0, 1]),
        ("<=", 2, [0, 1, 2]),
        ("==", 2, [2]),
        ("!=", 2, [0, 1, 3]),
        ("in", [0, 3], [0, 3]),
        ("not_in", [0, 3], [1, 2]),
    ],
)
def test_filter_comparisons_preserve_index(operation, value, expected):
    data = pd.DataFrame({"x": range(4)}, index=[9, 7, 5, 3])
    filt = LandscapeFilter([dict(column="x", operation=operation, value=value)])
    pd.testing.assert_frame_equal(filt.apply(data), data.iloc[expected])


@pytest.mark.parametrize("combine,expected", [("and", [1]), ("or", [0, 1, 3])])
def test_filter_combination(combine, expected):
    data = pd.Series([0.0, 1.0, 2.0, 3.0], name="fitness")
    rules = [
        dict(column="fitness", operation="<", value=2),
        dict(column="fitness", operation="custom", function=lambda s: s % 2 == 1),
    ]
    pd.testing.assert_series_equal(
        LandscapeFilter(rules, combine).apply(data), data.iloc[expected]
    )


def test_grid_search_enumerates_cartesian_product_once():
    result = grid_search({"x": [0, 1, 2], "y": [1, 3]}, lambda row: row["x"] * row["y"])
    assert list(result[["x", "y"]].itertuples(index=False, name=None)) == list(
        product([0, 1, 2], [1, 3])
    )
    assert result.fitness.tolist() == [0, 0, 1, 3, 2, 6]


@pytest.mark.parametrize("strategy", ["best-improvement", "first-improvement"])
def test_hillclimb_reaches_onemax_peak_via_only_improving_edges(strategy):
    X = np.array(list(product([0, 1], repeat=4)))
    ls = BooleanLandscape().build_from_data(X, X.sum(axis=1), verbose=False)
    walker = HillClimb(SearchCache(ls.graph), strategy=strategy, seed=8)
    for start in range(16):
        result = walker.run(start)
        assert result['final'] == 15
        assert result['path'][0] == start
        assert result['n_steps'] == 4 - X[start].sum()
        for a, b in zip(result['path'][:-1], result['path'][1:]):
            assert ls.graph.get_eid(int(a), int(b), error=False) >= 0


def test_random_walk_stays_on_mutational_neighbors():
    X = np.array(list(product([0, 1], repeat=3)))
    ls = BooleanLandscape().build_from_data(X, X.sum(axis=1), verbose=False)
    result = RandomWalk(SearchCache(ls.graph), length=20, seed=0).run(0)
    for a, b in zip(result['path'][:-1], result['path'][1:]):
        assert np.count_nonzero(X[a] != X[b]) == 1


@pytest.mark.parametrize('walker_name', ['RandomWalk', 'HillClimb'])
def test_walk_dictionary_includes_isolated_start(walker_name):
    import igraph as ig
    from graphfla import algorithms

    graph = ig.Graph(n=1, directed=True)
    graph.vs['fitness'] = [1.0]
    cache = algorithms.SearchCache(graph)
    result = getattr(algorithms, walker_name)(cache).run(0)
    assert type(result) is dict
    assert set(result) == {'path', 'final', 'n_steps'}
    np.testing.assert_array_equal(result['path'], [0])
    assert result['final'] == 0 and result['n_steps'] == 0


@pytest.mark.parametrize('length', [0, -1, 1.5, True])
def test_random_walk_requires_positive_integer_length(length):
    from _landscapes import onemax
    from graphfla.algorithms import SearchCache, RandomWalk
    cache = SearchCache(onemax(2).graph)
    with pytest.raises(ValueError, match='positive integer'):
        RandomWalk(cache, length=length)
