"""Edge-population contracts, independent statistics and numeric robustness."""

from itertools import product
from types import SimpleNamespace

import igraph as ig
import numpy as np
import pytest
from scipy.stats import linregress, pearsonr, spearmanr

from graphfla import analysis as A
from graphfla.analysis.epistasis import _fitness_trends as kernel
from graphfla.landscape import BooleanLandscape, Landscape, OrdinalLandscape

FUNCTIONS = [A.diminishing_returns_index, A.increasing_costs_index]
METHODS = ["pearson", "spearman", "regression"]


def graph_input(fitness, edges, maximize=True):
    graph = ig.Graph(n=len(fitness), edges=edges, directed=True)
    graph.vs["fitness"] = fitness
    return SimpleNamespace(graph=graph, maximize=maximize, _check_built=lambda: None)


def oracle(landscape, method, costs):
    """Enumerate every retained edge independently; never average by vertex."""
    q = np.asarray(landscape.graph.vs["fitness"]) * (1 if landscape.maximize else -1)
    rows = [
        (q[v if costs else u], q[v] - q[u])
        for u, v in landscape.graph.get_edgelist()
        if q[v] > q[u]
    ]
    x, y = np.asarray(rows).T
    if method == "regression":
        return linregress(x, y).slope
    return (pearsonr if method == "pearson" else spearmanr)(x, y).statistic


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("costs,fn", list(enumerate(FUNCTIONS)))
def test_unequal_degrees_use_each_edge_and_correct_background(method, costs, fn):
    ls = graph_input(
        [0.0, 1.0, 3.0, 7.0, 9.0], [(0, 1), (0, 2), (0, 4), (1, 2), (2, 3)]
    )
    expected = oracle(ls, method, costs)
    assert fn(ls, method) == pytest.approx(expected, abs=1e-14)
    q = ls.graph.vs["fitness"]
    grouped = {}
    for u, v in ls.graph.get_edgelist():
        grouped.setdefault(v if costs else u, []).append(q[v] - q[u])
    x, y = zip(*[(q[u], np.mean(v)) for u, v in grouped.items()])
    old = (
        linregress(x, y).slope
        if method == "regression"
        else (pearsonr if method == "pearson" else spearmanr)(x, y).statistic
    )
    assert expected != pytest.approx(old)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fn", FUNCTIONS)
@pytest.mark.parametrize(
    "scale,offset", [(1.0, 2**48), (1e-250, 0), (1e250, 0), (-3.0, 21.0)]
)
def test_units_translation_and_minimization(method, fn, scale, offset):
    f = np.array([0.0, 1.0, 3.0, 7.0, 9.0])
    edges = [(0, 1), (0, 2), (0, 4), (1, 2), (2, 3)]
    ref = fn(graph_input(f, edges), method)
    ls = graph_input(f * scale + offset, edges, maximize=scale > 0)
    assert fn(ls, method) == pytest.approx(ref, rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fn", FUNCTIONS)
def test_extreme_opposite_sign_fitness_does_not_overflow(method, fn):
    edges = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
    f = np.array([-1.0, -0.25, 0.25, 1.0])
    assert fn(graph_input(f * 1.7e308, edges), method) == pytest.approx(
        fn(graph_input(f, edges), method), abs=2e-14
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("costs,fn", list(enumerate(FUNCTIONS)))
@pytest.mark.parametrize("kind", ["boolean", "categorical", "ordinal"])
def test_built_neighborhood_filters_and_missing_data(method, costs, fn, kind):
    X = np.asarray(list(product(range(2 if kind == "boolean" else 3), repeat=3)))[:-2]
    f = X @ [1.0, 2.0, 3.0] + X[:, 0] * X[:, 2]
    cls = dict(
        boolean=BooleanLandscape, categorical=Landscape, ordinal=OrdinalLandscape
    )[kind]
    kwargs = (
        {"data_types": {i: "categorical" for i in range(3)}}
        if kind == "categorical"
        else {}
    )
    ls = cls().build_from_data(X, f, epsilon=1.1, verbose=False, **kwargs)
    assert fn(ls, method) == pytest.approx(oracle(ls, method, costs), abs=2e-14)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fn", FUNCTIONS)
def test_edge_attributes_do_not_override_node_fitness(method, fn):
    ls = graph_input([0.0, 1.0, 3.0, 7.0], [(0, 1), (0, 2), (1, 3), (2, 3)])
    expected = fn(ls, method)
    ls.graph.es["delta_fit"] = [float("nan"), 0, 1e300, -99]
    assert fn(ls, method) == expected


@pytest.mark.parametrize("fn", FUNCTIONS)
@pytest.mark.parametrize("method", METHODS)
def test_neutral_edges_are_excluded(method, fn):
    ls = graph_input([0.0, 0.0, 1.0, 3.0], [(0, 2), (0, 3), (2, 3)])
    expected = fn(ls, method)
    ls.graph.add_edges([(0, 1), (2, 2)])
    assert fn(ls, method) == expected


@pytest.mark.parametrize("fn", FUNCTIONS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("edges", [[], [(0, 1)]])
def test_insufficient_edges(method, fn, edges):
    with pytest.warns(UserWarning, match="undefined"):
        assert np.isnan(fn(graph_input([0.0, 1.0], edges), method))


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_constant_effect_slope_is_zero_but_correlation_undefined(fn):
    ls = graph_input([0.0, 1.0, 2.0, 3.0], [(0, 1), (1, 2), (2, 3)])
    assert fn(ls, "regression") == 0
    for method in ["pearson", "spearman"]:
        with pytest.warns(UserWarning, match="undefined"):
            assert np.isnan(fn(ls, method))


@pytest.mark.parametrize(
    "fn,edges", [(FUNCTIONS[0], [(0, 1), (0, 2)]), (FUNCTIONS[1], [(0, 2), (1, 2)])]
)
def test_constant_background_is_undefined_even_for_regression(fn, edges):
    with pytest.warns(UserWarning, match="undefined"):
        assert np.isnan(fn(graph_input([0.0, 1.0, 3.0], edges), "regression"))


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_invalid_method_checked_even_on_edgeless_input(fn):
    with pytest.raises(ValueError, match="Method"):
        fn(graph_input([], []), "not-a-method")


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_invalid_inputs(fn):
    with pytest.raises(RuntimeError):
        fn(BooleanLandscape())
    ls = graph_input([0.0, 1.0], [(0, 1)])
    for invalid in [np.nan, np.inf, -np.inf]:
        ls.graph.vs["fitness"] = [0.0, invalid]
        with pytest.raises(ValueError, match="finite"):
            fn(ls)
    del ls.graph.vs["fitness"]
    with pytest.raises(ValueError, match="fitness"):
        fn(ls)
    ls.graph = None
    with pytest.raises(ValueError, match="graph"):
        fn(ls)
    ls = graph_input([0.0, 1.0], [(1, 0)])
    with pytest.raises(ValueError, match="improving"):
        fn(ls)
    ls.graph.to_undirected()
    with pytest.raises(ValueError, match="directed"):
        fn(ls)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fn", FUNCTIONS)
def test_block_size_and_edge_order_do_not_change_statistic(monkeypatch, method, fn):
    rng = np.random.default_rng(11)
    f = rng.normal(size=80)
    pairs = list(product(range(40), range(40, 80)))
    edges = [(u, v) if f[u] < f[v] else (v, u) for u, v in pairs]
    ls = graph_input(f, edges)
    expected = fn(ls, method)
    monkeypatch.setattr(kernel, "_EDGE_BLOCK", 7)
    rng.shuffle(edges)
    assert fn(graph_input(f, edges), method) == pytest.approx(expected, abs=2e-14)


def test_profile_matches_direct_calls():
    ls = BooleanLandscape().build_from_data(
        ["00", "01", "10", "11"], [0.0, 2.0, 3.0, 4.0], verbose=False
    )
    names = [f.__name__ for f in FUNCTIONS]
    result = A.profile(ls, metrics=names)
    for fn in FUNCTIONS:
        assert result[fn.__name__] == fn(ls)


def test_additive_landscape_can_have_a_pooled_trend_without_epistasis():
    # Unequal fixed mutation effects change the mixture available at each
    # background. A nonzero pooled index is not evidence of an interaction.
    ls = BooleanLandscape().build_from_data(
        ["00", "01", "10", "11"], [0.0, 2.0, 1.0, 3.0], epsilon=0, verbose=False
    )
    assert A.diminishing_returns_index(ls) == pytest.approx(-1 / np.sqrt(11))
    assert A.increasing_costs_index(ls) == pytest.approx(1 / np.sqrt(11))


def test_single_background_multiple_edges_is_still_undefined():
    # Edge count alone does not make constant-background correlation defined.
    ls = graph_input([0.0, 1.0, 3.0], [(0, 1), (0, 2)])
    with pytest.warns(UserWarning, match="undefined"):
        assert np.isnan(A.diminishing_returns_index(ls))
    assert A.increasing_costs_index(ls) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", FUNCTIONS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "fitness,edges", [([], []), ([0.0, 0.0, 0.0], [(0, 1), (1, 2)])]
)
def test_empty_or_only_neutral_population_warns(fn, method, fitness, edges):
    with pytest.warns(UserWarning, match="undefined"):
        assert np.isnan(fn(graph_input(fitness, edges), method))


@pytest.mark.parametrize("fn", FUNCTIONS)
@pytest.mark.parametrize("method", METHODS)
def test_isolated_extreme_fitness_cannot_change_edge_statistic(fn, method):
    f = [0.0, 1.0, 3.0, 7.0]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    expected = fn(graph_input(f, edges), method)
    shifted = [(u + 1, v + 1) for u, v in edges]
    assert fn(graph_input([1e308] + f, shifted), method) == pytest.approx(expected)
