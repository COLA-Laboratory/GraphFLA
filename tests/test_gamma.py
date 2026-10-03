"""Equation and numerical contracts for gamma, separate from empirical tests."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from graphfla import analysis as A
from graphfla.landscape import (
    BooleanLandscape,
    Landscape,
    ProteinLandscape,
    OrdinalLandscape,
)
from graphfla.exceptions import NotBuiltError
from validation.oracles.gamma import directed_gamma


def build(values, variants=None, **kwargs):
    if variants is None:
        variants = list(product([0, 1], repeat=2))
    return BooleanLandscape().build_from_data(
        variants, np.asarray(values), verbose=False, **kwargs
    )


@pytest.mark.parametrize("scale", [1.0, 1e200, 1e-200, -1e200, -1e-200, 1e-320, 5e-324])
def test_gamma_scale_invariance(scale):
    # Appendix C.1: numerator 16, denominator 18, independent of fitness units.
    landscape = build(np.array([0.0, 1.0, 2.0, 4.0]) * scale)
    assert A.gamma(landscape, n_jobs=1) == pytest.approx(8 / 9)
    assert A.gamma_star(landscape, n_jobs=1) == 1.0


def test_overflowing_fitness_difference():
    extreme = np.finfo(float).max
    # Construction's edge orientation also subtracts these endpoints. This
    # test targets analysis; retain the values without changing construction.
    with np.errstate(over="ignore"):
        landscape = build([-extreme, extreme, extreme, -extreme])
    with np.errstate(over="raise", invalid="raise"):
        assert A.gamma(landscape, n_jobs=1) == -1.0
        assert A.gamma_star(landscape, n_jobs=1) == -1.0


def test_nonparticipating_outlier_cannot_erase_small_square():
    # The final genotype has a neighbor but belongs to no complete square.
    variants = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1)]
    landscape = build([0, 1e-200, 2e-200, 4e-200, 1e200], variants)
    assert A.gamma(landscape, n_jobs=1) == pytest.approx(8 / 9)
    assert A.gamma_star(landscape, n_jobs=1) == 1.0


def test_neutral_large_square_cannot_erase_small_effects():
    # Two separate squares. The large constant offset has no effect and must
    # not set the scale used to square the tiny effects on the other square.
    variants = list(product([0, 1], repeat=3))
    # Separate squares by two correlated background columns; no cross squares.
    variants = [(*g[:2], g[2], g[2]) for g in variants]
    values = [
        1e200 if g[2] else [0, 1e-200, 2e-200, 4e-200][2 * g[0] + g[1]]
        for g in variants
    ]
    landscape = build(values, variants)
    assert A.gamma(landscape, n_jobs=1) == pytest.approx(8 / 9)
    assert A.gamma_star(landscape, n_jobs=1) == 1.0


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0, 0, 1, 2], 2 / 3),
        ([0, 0, 0, 1], 0.0),
        ([0, 1, 1, 0], -1.0),
        ([0, 1, 2, 3], 1.0),
    ],
)
def test_gamma_star_neutral_denominator(values, expected):
    landscape = build(values)
    assert A.gamma_star(landscape, n_jobs=1) == pytest.approx(expected)
    reference = directed_gamma(list(product([0, 1], repeat=2)), values, signs=True)
    assert reference["value"] == pytest.approx(expected)
    assert reference["directed_quadruples"] == 8


@pytest.mark.parametrize("values", [[1, 1, 1, 1], [1, 2, 3]])
def test_undefined_without_nonzero_effects_or_complete_squares(values):
    variants = list(product([0, 1], repeat=2))[: len(values)]
    landscape = build(values, variants)
    assert np.isnan(A.gamma(landscape, n_jobs=1))
    assert np.isnan(A.gamma_star(landscape, n_jobs=1))


def test_graph_epsilon_does_not_redefine_signs():
    for epsilon in [0, 0.5, 5]:
        landscape = build([0, 0.25, 1, 2], epsilon=epsilon)
        assert A.gamma_star(landscape, n_jobs=1) == 1.0


def test_motif_identity_only_for_same_tie_free_population():
    tied = build([0, 0, 1, 2])
    motifs = A.classify_epistasis(tied, sample_cut_prob=0)
    assert A.gamma_star(tied, n_jobs=1) == pytest.approx(2 / 3)
    assert 1 - motifs['sign'] - 2 * motifs['reciprocal_sign'] == 1
    variants = list(product([0, 1], repeat=4))
    fitness = np.random.default_rng(1).permutation(16)
    landscape = build(fitness, variants)
    motifs = A.classify_epistasis(landscape, sample_cut_prob=0)
    assert A.gamma_star(landscape, n_jobs=1) == pytest.approx(
        1 - motifs['sign'] - 2 * motifs['reciprocal_sign']
    )


def test_heterogeneous_allele_counts_and_sparse_pooling():
    # Equal weights per directed substitution pair, not per position or square
    # ratio. Unequal allele counts are absent from the older oracle tests.
    variants = list(product(range(2), range(3), range(4)))[:-3]
    fitness = np.random.default_rng(4).integers(-3, 8, len(variants))
    frame = pd.DataFrame(variants, columns=["a", "b", "c"])
    landscape = Landscape().build_from_data(
        frame, fitness, data_types={c: "categorical" for c in frame}, verbose=False
    )
    for signs, metric in [(False, A.gamma), (True, A.gamma_star)]:
        expected = directed_gamma(variants, fitness, signs=signs)["value"]
        assert metric(landscape, n_jobs=1) == pytest.approx(expected)
        assert metric(landscape, n_jobs=2) == pytest.approx(expected)


def test_pure_walsh_order_formula():
    # Ferretti Eq. (26), also Ghafari et al. (2026) Eq. (3): an order-k
    # component alone has gamma = 1 - 2*(k-1)/(L-1), including eggbox = -1.
    variants = np.array(list(product([0, 1], repeat=5)))
    for order in range(1, 6):
        fitness = np.prod(2 * variants[:, :order] - 1, axis=1)
        landscape = build(fitness, variants)
        assert A.gamma(landscape, n_jobs=1) == pytest.approx(1 - (order - 1) / 2)


def test_extreme_scales_still_pool_effect_size_weights():
    variants = [(*g[:2], g[2], g[2]) for g in product([0, 1], repeat=3)]
    values = [
        [1e200, -1e200, -1e200, 1e200][2 * g[0] + g[1]]
        if g[2]
        else [0, 1e-200, 2e-200, 4e-200][2 * g[0] + g[1]]
        for g in variants
    ]
    landscape = build(values, variants)
    assert A.gamma(landscape, n_jobs=1) == -1.0
    assert A.gamma_star(landscape, n_jobs=1) == 0.0


@pytest.mark.parametrize("scale", [1, 1e-200])
def test_dict_fallback_at_tiny_scale(scale):
    variants = np.zeros((8, 66), dtype=int)
    variants[:, :3] = list(product([0, 1], repeat=3))
    variants[:, 3:] = variants[:, 2, None]
    fitness = np.array([0, 1, 2, 4, 3, 5, 7, 8])
    landscape = build(fitness * scale, variants)
    for signs, metric in [(False, A.gamma), (True, A.gamma_star)]:
        assert metric(landscape, n_jobs=1) == pytest.approx(
            directed_gamma(variants, fitness, signs=signs)["value"]
        )


def test_all_three_level_squares_match_directed_equations():
    variants = list(product([0, 1], repeat=2))
    for values in product([-1, 0, 1], repeat=4):
        landscape = build(values)
        for signs, metric in [(False, A.gamma), (True, A.gamma_star)]:
            expected = directed_gamma(variants, values, signs=signs)["value"]
            actual = metric(landscape, n_jobs=1)
            if np.isnan(expected):
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(expected)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("n_alleles", [2, 3])
def test_gamma_matches_directed_substitution_equation(sparse, maximize, n_alleles):
    variants = list(product(range(n_alleles), repeat=3))
    if sparse:
        variants = variants[:-2]
    fitness = np.random.default_rng(27).integers(-3, 7, len(variants)).astype(float)
    sequences = ["".join("ACW"[v] for v in variant) for variant in variants]
    landscape = ProteinLandscape(maximize=maximize).build_from_data(
        sequences, fitness, verbose=False
    )
    assert A.gamma(landscape, n_jobs=1) == pytest.approx(
        directed_gamma(variants, fitness)["value"]
    )
    assert A.gamma_star(landscape, n_jobs=1) == pytest.approx(
        directed_gamma(variants, fitness, signs=True)["value"]
    )


@pytest.mark.parametrize("metric", [A.gamma, A.gamma_star])
def test_public_input_errors_and_one_variable(metric):
    with pytest.raises(NotBuiltError, match="not been built"):
        metric(BooleanLandscape(), n_jobs=1)
    single = BooleanLandscape().build_from_data(["0", "1"], [0, 1], verbose=False)
    with pytest.warns(UserWarning, match="at least 2 variables"):
        assert np.isnan(metric(single, n_jobs=1))
    landscape = build([0, 1, 2, 4])
    with pytest.raises(ValueError, match="n_jobs"):
        metric(landscape, n_jobs=0)
    del landscape.graph.vs["fitness"]
    with pytest.raises(ValueError, match="fitness"):
        metric(landscape, n_jobs=1)


@pytest.mark.parametrize("metric", [A.gamma, A.gamma_star])
def test_relabeling_reordering_affine_units_and_profile(metric):
    X = pd.DataFrame(
        list(product(["red", "blue"], ["a", "b", "c"], [0, 1])),
        columns=["color", "type", "flag"],
    )
    values = np.random.default_rng(19).integers(-20, 20, len(X))
    types = {c: "categorical" for c in X}
    before = Landscape().build_from_data(X, values, data_types=types, verbose=False)
    expected = directed_gamma(X.to_numpy(), values, signs=metric is A.gamma_star)[
        "value"
    ]
    assert metric(before, n_jobs=1) == pytest.approx(expected)
    renamed = X.replace({"red": "Z", "blue": "A", "a": "C", "b": "B", "c": "A"})
    after = Landscape().build_from_data(
        renamed.iloc[::-1, ::-1],
        (-8 * values + 17)[::-1],
        data_types={c: types[c] for c in renamed.columns[::-1]},
        verbose=False,
    )
    actual = metric(after, n_jobs=1)
    assert type(actual) is float
    assert actual == pytest.approx(expected)
    result = A.profile(
        after, metrics=[metric.__name__], params={metric.__name__: {"n_jobs": 1}}
    )
    assert list(result.index) == [metric.__name__]
    assert result.iloc[0] == actual


def test_ordinal_metric_uses_all_observed_allele_pairs():
    variants = np.array(list(product(range(3), repeat=2)))
    values = np.array([0, 1, 4, 2, 3, 8, 5, 7, 6])
    landscape = OrdinalLandscape().build_from_data(variants, values, verbose=False)
    for signs, metric in [(False, A.gamma), (True, A.gamma_star)]:
        assert metric(landscape, n_jobs=1) == pytest.approx(
            directed_gamma(variants, values, signs=signs)["value"]
        )


@pytest.mark.parametrize("dense_limit", [1, 1 << 20])
def test_sparse_fallback_preserves_complete_square_weights(monkeypatch, dense_limit):
    from importlib import import_module

    module = import_module("graphfla.analysis.epistasis.gamma")
    monkeypatch.setattr(module, "_MAX_GRID_CELLS", dense_limit)
    # A sparse ring with one added corner creates one genuine rectangle among
    # 24 alleles per variable. Both the byte cap and occupancy route are tested.
    variants = [(a, b) for a in range(24) for b in (a, (a + 1) % 24)] + [(0, 2)]
    values = np.random.default_rng(8).integers(-2, 10, len(variants))
    frame = pd.DataFrame(variants, columns=["x", "y"])
    landscape = Landscape().build_from_data(
        frame, values, data_types={c: "categorical" for c in frame}, verbose=False
    )
    for signs, metric in [(False, A.gamma), (True, A.gamma_star)]:
        assert metric(landscape, n_jobs=1) == pytest.approx(
            directed_gamma(variants, values, signs=signs)["value"]
        )


def test_background_pair_products_avoid_squared_sum_cancellation():
    variants = list(product(range(2), range(3)))
    frame = pd.DataFrame(variants, columns=["x", "y"])
    landscape = Landscape().build_from_data(
        frame,
        [0, 0, 0, 1e200, 1, 1],
        data_types={c: "categorical" for c in frame},
        verbose=False,
    )
    # Analytically (2*H+1)/(2*H**2-2*H+3), indistinguishable from 1/H here.
    # Computing ((sum s)**2 - sum(s**2))/2 would erase the small numerator.
    assert A.gamma(landscape, n_jobs=1) == pytest.approx(1e-200, rel=1e-12, abs=0)
    assert A.gamma_star(landscape, n_jobs=1) == 0.75


def test_four_observations_can_still_have_no_complete_square():
    frame = pd.DataFrame([(0, 0), (0, 1), (1, 2), (1, 3)], columns=["x", "y"])
    landscape = Landscape().build_from_data(
        frame, [0, 1, 3, 2], data_types={c: "categorical" for c in frame}, verbose=False
    )
    assert landscape.n_configs == 4
    assert np.isnan(A.gamma(landscape, n_jobs=1))
    assert np.isnan(A.gamma_star(landscape, n_jobs=1))


def test_multiallelic_overflowing_differences():
    variants = list(product(range(2), range(3)))
    frame = pd.DataFrame(variants, columns=["x", "y"])
    units = np.array([-1, 1, -1, 1, -1, 1])
    with np.errstate(over="ignore"):
        landscape = Landscape().build_from_data(
            frame, units * np.finfo(float).max,
            data_types={c: "categorical" for c in frame}, verbose=False
        )
    for signs, metric in [(False, A.gamma), (True, A.gamma_star)]:
        with np.errstate(over="raise", invalid="raise"):
            assert metric(landscape, n_jobs=1) == pytest.approx(
                directed_gamma(variants, units, signs=signs)["value"]
            )
