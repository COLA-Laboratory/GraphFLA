"""Definition and statistical edge cases for Wagner EE mutations."""

from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from graphfla.analysis import (
    evolvability_enhancing_mutations, evolvability_enhancing_fraction,
    evolvability_effects,
)
from graphfla.analysis._evolvability import (
    _benjamini_hochberg, _bh_adjusted_pvalues, _ee_pvalues, _ee_statistics,
    _landscape_ee_statistics,
)
from graphfla.landscape import BooleanLandscape, Landscape
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


@pytest.mark.parametrize("epsilon", [-1, np.inf, np.nan, "bad", 1j, True, [0]])
def test_invalid_epsilon(epsilon):
    X, f, _ = two_stars()
    ls = BooleanLandscape().build_from_data(X, f, epsilon=.001, verbose=False)
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="epsilon"):
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
    assert evolvability_enhancing_fraction(ls) == 0.0


@pytest.mark.parametrize("fdr", [0, 1, -.1, np.inf, np.nan, True, "0.01", None, [0.01]])
@pytest.mark.parametrize("function", [evolvability_enhancing_fraction, evolvability_effects])
def test_invalid_fdr(function, fdr):
    X, f, _ = two_stars()
    ls = BooleanLandscape().build_from_data(X, f, epsilon=.001, verbose=False)
    with pytest.raises(ValueError, match="fdr"):
        function(ls, fdr=fdr)


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


def test_legacy_parameters_keep_their_behavior_with_migration_warning():
    X, f, _ = two_stars()
    ls = BooleanLandscape().build_from_data(X, f, epsilon=.001, verbose=False)
    with pytest.warns(FutureWarning), pytest.raises(RuntimeError, match="haven't been calculated"):
        evolvability_enhancing_mutations(ls, auto_calculate=False)
    with pytest.warns(FutureWarning, match="evolvability_enhancing_fraction"):
        old = evolvability_enhancing_mutations(ls)
    assert old == evolvability_enhancing_fraction(ls)
    assert "delta_mean_neighbor_fit" in ls.graph.es.attributes()
    with pytest.warns(FutureWarning):
        assert evolvability_enhancing_mutations(ls, epsilon=99, auto_calculate=False) == 0
