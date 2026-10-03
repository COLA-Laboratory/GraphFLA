"""Definition and statistical edge cases for Wagner EE mutations."""

from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from graphfla.analysis import (
    evolvability_enhancing_fraction,
    evolvability_effects,
)
from graphfla.analysis._evolvability import (
    _benjamini_hochberg, _bh_adjusted_pvalues, _ee_pvalues, _ee_statistics,
    _landscape_ee_statistics, _validate_fdr,
)
from graphfla.landscape import BooleanLandscape, Landscape
from validation.oracles.ee import reference_statistics


def hamming_pairs(configs):
    return [(u, v) for u, v in combinations(range(len(configs)), 2)
            if sum(configs[u] != configs[v]) == 1]


def two_stars(effect=1.0, q=4):
    # Two focal backgrounds each have q non-focal neighbors. All these
    # neighbors have fitness 0 in background 0 and 100 in background 1.
    configs, fitness = [], []
    for focal in [0, 1]:
        configs.append([focal] + [0]*q)
        fitness.append(focal*effect)
        for pos in range(q):
            configs.append([focal] + [int(i == pos) for i in range(q)])
            fitness.append(100*focal)
    configs = np.asarray(configs)
    return configs, np.asarray(fitness, dtype=float), hamming_pairs(configs)


@pytest.mark.parametrize("effect", [1.0, 0.0, -1.0])
def test_beneficial_neutral_deleterious_and_ordered_denominator(effect):
    X, f, pairs = two_stars(effect)
    table = _ee_statistics(X, f, pairs)
    assert len(table) == 26  # 13 undirected pairs, both directions
    assert table.testable.sum() == 2
    assert table.ee.sum() == 1
    row = table[table.ee].iloc[0]
    assert row.source == 0 and row.target == 5
    assert row.delta_fitness == effect and row.delta_neighbor_fitness == 100
    # Neutral construction edges must be retained for a complete denominator.
    ls = BooleanLandscape().build_from_data(X, f, epsilon=.001, verbose=False)
    assert evolvability_enhancing_fraction(ls) == pytest.approx(1/26)
    expected_type = {1.: "beneficial", 0.: "neutral", -1.: "deleterious"}[effect]
    for effect_type in ["beneficial", "deleterious", "neutral"]:
        expected_fraction = 1/26 if effect_type == expected_type else 0.
        assert evolvability_enhancing_fraction(ls, effect_type=effect_type) == expected_fraction
    public = evolvability_effects(ls)
    assert str(public.is_ee.dtype) == "boolean"
    assert public.is_ee.isna().sum() == 24
    assert (public.status == "insufficient_neighbors").sum() == 24
    assert public.loc[public.is_ee.fillna(False), "effect_type"].tolist() == [expected_type]


def test_strict_excess_and_epsilon_are_not_pvalues():
    X, f, pairs = two_stars(effect=100)
    assert not _ee_statistics(X, f, pairs).ee.any()
    X, f, pairs = two_stars(effect=1)


def test_exclude_entire_multiallelic_site():
    X = np.asarray(list(product(range(3), range(3), range(3))))
    f = X[:, 0] + 2*X[:, 1] + 4*X[:, 2]
    table = _ee_statistics(X, f, hamming_pairs(X))
    assert set(table.n_source) == {4}
    np.testing.assert_allclose(table.delta_neighbor_fitness, table.delta_fitness)
    assert not table.ee.any()


@pytest.mark.parametrize("with_errors", [False, True])
def test_punctured_multiallelic_landscape_matches_independent_oracle(with_errors):
    rng = np.random.RandomState(3)
    X = np.asarray(list(product(range(3), range(2), range(3), range(2))))
    X = np.delete(X, [1, 4, 7, 12, 16, 19], axis=0)
    f = rng.normal(size=len(X))
    var = rng.uniform(.001, .1, len(X)) if with_errors else None
    pairs = hamming_pairs(X)
    reference = reference_statistics(X, f, pairs, var)
    actual = _ee_statistics(X, f, pairs, fitness_variance=var)
    for key in ["p_effect", "p_zero"]:
        np.testing.assert_allclose(actual[key], reference[key], atol=1e-14)
    expected = np.where(reference["effect"] > 0, reference["flag_effect"] == 1,
                        reference["flag_zero"] == 1)
    np.testing.assert_array_equal(actual.ee, expected)
    # Both variances matter. Reverse tests have equal p-values for these
    # symmetric nulls; the author's duplicated-target bug violates this.
    m = len(pairs)
    np.testing.assert_allclose(actual.p_effect[:m], actual.p_effect[m:])
    np.testing.assert_allclose(actual.p_zero[:m], actual.p_zero[m:])


def test_zero_variance_and_insufficient_neighbors():
    result = _ee_pvalues(np.array([0., 1., 1., 1.]),
                        np.array([0., 0., 1., 0.]), np.array([3, 3, 1, 1]))
    np.testing.assert_allclose(result, [1, 0, np.nan, np.nan], equal_nan=True)
    X = np.asarray(list(product([0, 1], repeat=2)))
    ls = BooleanLandscape().build_from_data(X, [0, 1, 2, 8], verbose=False)
    with pytest.warns(RuntimeWarning, match="No testable EE"):
        assert np.isnan(evolvability_enhancing_fraction(ls))
    table = evolvability_effects(ls)
    assert table.is_ee.isna().all()
    assert table[["p_effect", "p_zero", "q_effect", "q_zero"]].isna().all().all()


def test_bh_keeps_untestable_hypotheses_and_handles_boundaries():
    np.testing.assert_array_equal(
        _benjamini_hochberg([.0025, .005, .02, np.nan]), [True, True, False, False]
    )
    assert not _benjamini_hochberg([.006, np.nan]).any()
    assert not _benjamini_hochberg([]).any()
    # Four hypotheses remain in the correction, although one cannot be tested.
    np.testing.assert_allclose(
        _bh_adjusted_pvalues([.03, .004, np.nan, .01]),
        [.04, .016, np.nan, .02], equal_nan=True,
    )


def test_scale_offset_minimize_permutation_and_cache_independence():
    X, f, pairs = two_stars()
    for values, maximize in [(f, True), (-f, False), (8*f+1024, True)]:
        ls = BooleanLandscape(maximize=maximize).build_from_data(
            X, values, epsilon=.001, verbose=False)
        assert "delta_mean_neighbor_fit" not in ls.graph.es.attributes()
        assert evolvability_enhancing_fraction(ls) == pytest.approx(1/26)
        assert "delta_mean_neighbor_fit" not in ls.graph.es.attributes()
        ls.graph.es["delta_mean_neighbor_fit"] = [1e100]*ls.graph.ecount()
        assert evolvability_enhancing_fraction(ls) == pytest.approx(1/26)
        # Reciprocal graph storage must not double-count pairs.
        ls.graph.add_edges([(v, u) for u, v in ls.graph.get_edgelist()])
        assert evolvability_enhancing_fraction(ls) == pytest.approx(1/26)
    permutation = np.random.RandomState(8).permutation(len(X))
    table = _ee_statistics(X[permutation, ::-1], f[permutation],
                           hamming_pairs(X[permutation]))
    assert table.ee.sum() == 1 and len(table) == 26



def test_invalid_inputs():
    X, f, pairs = two_stars()
    with pytest.raises(ValueError, match="one-site"):
        _ee_statistics(X, f, [(1, 7)])
    for malformed in [X.ravel(), X[:-1]]:
        with pytest.raises(ValueError, match="align"):
            _ee_statistics(malformed, f, pairs)
    bad = f.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _ee_statistics(X, bad, pairs)
    for variance in [[1], [-1]*len(f), [np.inf]*len(f)]:
        with pytest.raises(ValueError, match="variances"):
            _ee_statistics(X, f, pairs, fitness_variance=variance)
    ls = BooleanLandscape().build_from_data(X, f, epsilon=.001, verbose=False)
    ls.data_types = None
    with pytest.raises(ValueError, match="Configuration columns"):
        _landscape_ee_statistics(ls)


def test_no_neighbor_pairs():
    table = _ee_statistics(np.zeros((1, 1)), [0], [])
    assert table.empty


def test_constant_landscape_has_no_ee_mutations():
    X = np.asarray(list(product([0, 1], repeat=3)))
    ls = BooleanLandscape().build_from_data(X, [.1]*len(X), epsilon=.001, verbose=False)
    assert evolvability_enhancing_fraction(ls) == 0.0


@pytest.mark.parametrize("fdr", [0, 1, -.1, np.inf, np.nan, True, "0.01", None, [0.01]])
def test_invalid_fdr(fdr):
    with pytest.raises(ValueError, match="fdr"):
        _validate_fdr(fdr)


@pytest.mark.parametrize("effect_type", ["positive", "", None, ["all"], 1])
def test_invalid_effect_type(effect_type):
    X, f, _ = two_stars()
    ls = BooleanLandscape().build_from_data(X, f, epsilon=.001, verbose=False)
    with pytest.raises(ValueError, match="effect_type"):
        evolvability_enhancing_fraction(ls, effect_type=effect_type)


def test_fdr_changes_decisions_but_not_pvalues_or_denominators():
    X = np.asarray(list(product([0, 1], repeat=4)))
    f = np.random.RandomState(2).normal(size=len(X))
    ls = BooleanLandscape().build_from_data(X, f, verbose=False)
    tables = []
    for fdr in [.01, .5, .9]:
        table = evolvability_effects(ls, fdr=fdr)
        tables.append(table)
        oracle = reference_statistics(X, f, hamming_pairs(X), fdr=fdr)
        expected = np.where(oracle["effect"] > 0, oracle["flag_effect"] == 1,
                            oracle["flag_zero"] == 1)
        actual_by_pair = table.set_index(["source_id", "target_id"])
        # Construction retains input row order for this complete landscape.
        index = list(zip(oracle["source"], oracle["target"]))
        np.testing.assert_array_equal(actual_by_pair.loc[index, "is_ee"], expected)
        fractions = []
        for effect_type in ["all", "beneficial", "deleterious", "neutral"]:
            mask = (table.effect_type == effect_type) if effect_type != "all" else True
            fraction = evolvability_enhancing_fraction(ls, fdr=fdr, effect_type=effect_type)
            assert fraction == (table.is_ee & mask).sum()/len(table)
            fractions.append(fraction)
        assert fractions[0] == sum(fractions[1:])
    assert [t.is_ee.sum() for t in tables] == [0, 3, 18]
    for other in tables[1:]:
        pd.testing.assert_frame_equal(
            tables[0].drop(columns="is_ee"), other.drop(columns="is_ee")
        )


def test_labels_join_to_nonbiological_configuration_data_and_empty_schema():
    X, f, _ = two_stars()
    data = pd.DataFrame(X, columns=["solver"] + [f"option_{i}" for i in range(4)])
    data["solver"] = data["solver"].map({0: "first", 1: "second"})
    ls = Landscape().build_from_data(
        data, f, data_types={name: "categorical" for name in data},
        epsilon=.001, neighborhood_strategy="active", verbose=False,
    )
    table = evolvability_effects(ls)
    retained = ls.get_data()
    assert isinstance(table.index, pd.RangeIndex)
    assert table.equals(table.sort_values(["source_id", "target_id"]))
    for row in table.itertuples():
        assert row.source_allele == retained.loc[row.source_id, row.position]
        assert row.target_allele == retained.loc[row.target_id, row.position]
    ee = table[table.is_ee.fillna(False)].iloc[0]
    assert (ee.position, ee.source_allele, ee.target_allele) == ("solver", "first", "second")
    ls.graph.delete_edges(ls.graph.es)
    ls._neutral_neighbors = None
    empty = evolvability_effects(ls)
    assert empty.empty and empty.columns.tolist() == table.columns.tolist()
    assert str(empty.is_ee.dtype) == "boolean"
    with pytest.warns(RuntimeWarning, match="No testable EE"):
        assert np.isnan(evolvability_enhancing_fraction(ls))



@pytest.mark.parametrize("function", [evolvability_enhancing_fraction, evolvability_effects])
def test_public_input_validation_and_unbuilt_guard(function):
    with pytest.raises(RuntimeError, match="built"):
        function(BooleanLandscape())
    X, f, _ = two_stars()
    ls = BooleanLandscape().build_from_data(X, f, epsilon=.001, verbose=False)
    with pytest.raises(ValueError, match="fdr"):
        function(ls, fdr=0)
    # Each malformed landscape is rejected before producing classifications.
    column = next(iter(ls.data_types))
    original = ls.graph.vs[column]
    ls.graph.vs[0][column] = None
    with pytest.raises(ValueError, match="missing"):
        function(ls)
    ls.graph.vs[column] = original
    for name in ls.data_types:
        ls.graph.vs[1][name] = ls.graph.vs[0][name]
    with pytest.raises(ValueError, match="unique"):
        function(ls)


def test_blocking_wide_inputs_and_edge_order(monkeypatch):
    import graphfla.analysis._evolvability as implementation
    X = np.asarray(list(product(range(3), range(3), range(2), range(2))))
    f = .1 + X[:, 1]*1e-4 + X[:, 2]*.02 + X[:, 3]*.03
    f += np.where(X[:, 0] == 2, 2.0, X[:, 0]*(1+X[:, 1]*.01))
    X = np.column_stack([np.zeros((len(X), 37)), X, np.ones((len(X), 41))])
    X = np.delete(X, [3, 8, 19], axis=0)
    f = np.delete(f, [3, 8, 19])
    pairs = hamming_pairs(X)[::-1]
    pairs[::2] = [(v, u) for u, v in pairs[::2]]
    oracle = reference_statistics(X, f, pairs)
    monkeypatch.setattr(implementation, "_MOMENT_CELLS", 17)
    actual = _ee_statistics(X, f, pairs)
    for name in ["p_effect", "p_zero"]:
        np.testing.assert_allclose(actual[name], oracle[name], rtol=1e-10, atol=2e-12)
    expected = np.where(oracle["effect"] > 0, oracle["flag_effect"] == 1,
                        oracle["flag_zero"] == 1)
    np.testing.assert_array_equal(actual.ee, expected)


def test_empty_focal_neighborhood_is_untestable():
    # All changes are at a single categorical position, so exclusion removes
    # every neighbor even though the graph has many edges.
    X = np.arange(5)[:, None]
    actual = _ee_statistics(X, np.arange(5), hamming_pairs(X))
    assert len(actual) == 20 and (actual.n_source == 0).all()
    assert actual.p_effect.isna().all() and not actual.testable.any()


def test_large_excluded_effect_preserves_small_nonfocal_variance():
    # Do not subtract huge focal second moments from total second moments.
    # Check well-conditioned local moments directly; tiny differences between
    # means near 1e10 cannot provide a precise p-value oracle in float64.
    X = np.asarray(list(product(range(3), range(3), range(2))))
    f = X[:, 1]*1e-4 + X[:, 2]*.02 + np.where(X[:, 0] == 2, 1e10, X[:, 0])
    table = _ee_statistics(X, f, hamming_pairs(X))
    selected = table[(table.position == 0) & (X[table.source, 0] < 2)]
    assert len(selected) > 0
    for row in selected.itertuples():
        u = row.source
        neighbors = (np.sum(X != X[u], axis=1) == 1) & (X[:, 0] == X[u, 0])
        np.testing.assert_allclose(row.variance_source, np.var(f[neighbors]),
                                   rtol=1e-12, atol=1e-16)
