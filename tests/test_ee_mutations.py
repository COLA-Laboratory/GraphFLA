"""Definition and statistical edge cases for Wagner EE mutations."""

from itertools import combinations, product

import numpy as np
import pytest

from graphfla.analysis import evolvability_enhancing_mutations
from graphfla.analysis._evolvability import (
    _benjamini_hochberg, _ee_pvalues, _ee_statistics, _landscape_ee_statistics,
)
from graphfla.landscape import BooleanLandscape
from validation.ee_mutations import reference_statistics


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
    assert evolvability_enhancing_mutations(ls) == pytest.approx(1/26)


def test_strict_excess_and_epsilon_are_not_pvalues():
    X, f, pairs = two_stars(effect=100)
    assert not _ee_statistics(X, f, pairs).ee.any()
    X, f, pairs = two_stars(effect=1)
    assert _ee_statistics(X, f, pairs, epsilon=98).ee.sum() == 1
    assert _ee_statistics(X, f, pairs, epsilon=99).ee.sum() == 0


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
        assert np.isnan(evolvability_enhancing_mutations(ls))


def test_bh_keeps_untestable_hypotheses_and_handles_boundaries():
    np.testing.assert_array_equal(
        _benjamini_hochberg([.0025, .005, .02, np.nan]), [True, True, False, False]
    )
    assert not _benjamini_hochberg([.006, np.nan]).any()
    assert not _benjamini_hochberg([]).any()


def test_scale_offset_minimize_permutation_and_cache_independence():
    X, f, pairs = two_stars()
    for values, maximize in [(f, True), (-f, False), (8*f+1024, True)]:
        ls = BooleanLandscape(maximize=maximize).build_from_data(
            X, values, epsilon=.001, verbose=False)
        with pytest.raises(RuntimeError, match="haven't been calculated"):
            evolvability_enhancing_mutations(ls, auto_calculate=False)
        assert evolvability_enhancing_mutations(ls) == pytest.approx(1/26)
        ls.graph.es["delta_mean_neighbor_fit"] = [1e100]*ls.graph.ecount()
        assert evolvability_enhancing_mutations(ls, auto_calculate=False) == pytest.approx(1/26)
        # Reciprocal graph storage must not double-count pairs.
        ls.graph.add_edges([(v, u) for u, v in ls.graph.get_edgelist()])
        assert evolvability_enhancing_mutations(ls, auto_calculate=False) == pytest.approx(1/26)
    permutation = np.random.RandomState(8).permutation(len(X))
    table = _ee_statistics(X[permutation, ::-1], f[permutation],
                           hamming_pairs(X[permutation]))
    assert table.ee.sum() == 1 and len(table) == 26


@pytest.mark.parametrize("epsilon", [-1, np.inf, np.nan, "bad", 1j, True, [0]])
def test_invalid_epsilon(epsilon):
    X, f, _ = two_stars()
    ls = BooleanLandscape().build_from_data(X, f, epsilon=.001, verbose=False)
    with pytest.raises(ValueError, match="epsilon"):
        evolvability_enhancing_mutations(ls, epsilon=epsilon)


def test_invalid_inputs():
    X, f, pairs = two_stars()
    with pytest.raises(ValueError, match="one-site"):
        _ee_statistics(X, f, [(1, 7)])
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
        _landscape_ee_statistics(ls, 0)


def test_no_neighbor_pairs():
    table = _ee_statistics(np.zeros((1, 1)), [0], [])
    assert table.empty


def test_constant_landscape_has_no_ee_mutations():
    X = np.asarray(list(product([0, 1], repeat=3)))
    ls = BooleanLandscape().build_from_data(X, [.1]*len(X), epsilon=.001, verbose=False)
    assert evolvability_enhancing_mutations(ls) == 0.0
