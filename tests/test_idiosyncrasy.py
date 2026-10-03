"""Definition-level controls for Lyons' finite-sample SD ratio."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from graphfla.analysis import idiosyncratic_index, global_idiosyncratic_index
from graphfla.analysis.epistasis.idiosyncrasy import (
    _idiosyncratic_ratio,
)
from graphfla.landscape import BooleanLandscape, Landscape
from validation.oracles.idiosyncrasy import landscape_mean


def literal_reference(landscape, seed, min_pairs=3):
    """Independent tuple lookup and literal control enumeration, no metric helpers."""
    data = landscape.get_data()
    return landscape_mean(
        data[list(landscape.data_types)].itertuples(index=False, name=None),
        data.fitness.to_numpy(),
        seed,
        min_pairs,
    )


@pytest.fixture
def categorical():
    X = pd.DataFrame(
        list(product(["A", "C", "G"], [2, 7], [False, True])),
        columns=["site_9", "dose", "switch"],
    )
    f = [0, 1, 3, 2, 2, 8, 4, 7, 1, 5, 9, 3]
    return Landscape().build_from_data(
        X,
        f,
        data_types=dict.fromkeys(X.columns, "categorical"),
        verbose=False,
    )


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_directed_equal_weight_mean_matches_literal_definition(categorical, seed):
    expected = literal_reference(categorical, seed)
    assert global_idiosyncratic_index(
        categorical, n_jobs=1, seed=seed
    ) == pytest.approx(expected)


def test_seed_is_effective_parallel_stable_and_local(categorical):
    np.random.seed(872)
    before = np.random.get_state()
    a = global_idiosyncratic_index(categorical, n_jobs=1, seed=17)
    assert a == global_idiosyncratic_index(categorical, n_jobs=2, seed=17)
    assert a != global_idiosyncratic_index(categorical, n_jobs=1, seed=18)
    idiosyncratic_index(categorical, ("A", "site_9", "C"))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_single_mutation_labels_and_reverse_use_correct_backgrounds(categorical):
    rng_class = np.random.RandomState
    fitness = categorical.get_data().fitness.to_numpy()
    pairs = rng_class(7).choice(fitness, (4, 2), replace=True)
    expected = np.std([2, 7, 1, 5]) / np.std(pairs[:, 1] - pairs[:, 0])
    assert idiosyncratic_index(categorical, ("A", "site_9", "C"), seed=7) == pytest.approx(
        expected
    )
    assert idiosyncratic_index(categorical, ("C", "site_9", "A"), seed=7) == pytest.approx(
        expected
    )


def test_control_has_exactly_matched_size_and_replacement():
    class RecordedDraw:
        def choice(self, pool, size, replace):
            np.testing.assert_array_equal(pool, [0, 1, 2, 3])
            assert size == (3, 2) and replace is True
            # Includes a self-pair and repeats an endpoint, both permitted.
            return np.asarray([[0, 0], [0, 2], [0, 3]])

    expected = np.std([1, 2, 5]) / np.std([0, 2, 3])
    assert (
        _idiosyncratic_ratio(np.std([1, 2, 5]), 3, [0, 1, 2, 3], RecordedDraw())
        == expected
    )
    assert expected != pytest.approx(
        np.std([1, 2, 5]) / (np.sqrt(2) * np.std([0, 1, 2, 3]))
    )


@pytest.mark.parametrize("value", [1, 0, -1, 2.5, True, None])
def test_invalid_min_pairs(categorical, value):
    with pytest.raises(ValueError, match="min_pairs"):
        global_idiosyncratic_index(categorical, n_jobs=1, min_pairs=value)
    with pytest.raises(ValueError, match="min_pairs"):
        idiosyncratic_index(categorical, ("A", "site_9", "C"), min_pairs=value)


@pytest.mark.parametrize(
    "mutation",
    [
        ("A", "missing", "C"),
        ("T", "site_9", "C"),
        ("A", "site_9", "T"),
        ("A", "site_9", "A"),
    ],
)
def test_invalid_mutation(categorical, mutation):
    with pytest.raises(ValueError):
        idiosyncratic_index(categorical, mutation)


def test_flat_and_too_few_backgrounds_are_undefined():
    seqs = ["".join(x) for x in product("01", repeat=3)]
    flat = BooleanLandscape().build_from_data(seqs, [1] * 8, epsilon=0, verbose=False)
    assert np.isnan(global_idiosyncratic_index(flat, n_jobs=1, seed=0))
    col = next(iter(flat.data_types))
    alleles = flat.get_data()[col].unique()
    assert np.isnan(idiosyncratic_index(flat, (alleles[0], col, alleles[1])))
    square = BooleanLandscape().build_from_data(
        ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False
    )
    assert np.isnan(global_idiosyncratic_index(square, n_jobs=1, seed=0))
    # A matched sample of size two can be degenerate even on a nonflat landscape.
    assert np.isnan(literal_reference(square, 0, 2))
    with pytest.warns(RuntimeWarning, match="zero standard deviation"):
        assert np.isnan(
            global_idiosyncratic_index(square, n_jobs=1, seed=0, min_pairs=2)
        )


def test_failed_control_warns_instead_of_resampling_or_returning_zero():
    class ConstantDraw:
        def choice(self, pool, size, replace):
            return np.ones(size)

    with pytest.warns(RuntimeWarning, match="zero standard deviation"):
        assert np.isnan(_idiosyncratic_ratio(0, 3, [0, 1], ConstantDraw()))


def test_missing_backgrounds_do_not_dilute_mean():
    X = pd.DataFrame(
        [(a, b) for a in ["A", "B"] for b in range(4)] + [("C", 0)],
        columns=["focal", "background"],
    )
    ls = Landscape().build_from_data(
        X,
        [0, 1, 2, 3, 1, 4, 2, 7, 6],
        data_types=dict.fromkeys(X.columns, "categorical"),
        verbose=False,
    )
    assert global_idiosyncratic_index(ls, n_jobs=1, seed=0) == pytest.approx(
        literal_reference(ls, 0)
    )
    assert np.isnan(idiosyncratic_index(ls, ("A", "focal", "C")))


def test_long_sequence_fallback_matches_independent_oracle():
    short = np.asarray(list(product(range(2), repeat=4)), dtype=np.int32)
    long = np.column_stack([short[:, 0], np.tile(short[:, 1:], (1, 24))])
    f = np.random.RandomState(0).normal(size=len(short))
    ls = table_landscape(long, f)
    actual = global_idiosyncratic_index(ls, n_jobs=1, seed=23)
    assert actual == pytest.approx(literal_reference(ls, 23), rel=1e-12)


def test_nonlinear_global_map_can_have_positive_lyons_index():
    X = ["".join(s) for s in product("01", repeat=4)]
    ls = BooleanLandscape().build_from_data(
        X, [s.count("1") ** 2 for s in X], verbose=False
    )
    # This is a global transformation of a purely additive trait. I_id is not
    # a classifier distinguishing specific interactions from global epistasis.
    assert global_idiosyncratic_index(ls, n_jobs=1, seed=0) > 0


@pytest.mark.parametrize("bad", ["duplicate", "nonfinite", "missing", "no_columns"])
def test_ambiguous_imported_data_is_rejected(bad):
    frame = pd.DataFrame({"x": [0, 1, 2], "fitness": [1.0, 2.0, 3.0]})
    if bad == "duplicate":
        frame.loc[2, "x"] = 1
    elif bad == "nonfinite":
        frame.loc[2, "fitness"] = np.inf
    elif bad == "missing":
        frame.loc[2, "x"] = np.nan
    ls = SimpleNamespace(
        get_data=lambda: frame,
        data_types=None if bad == "no_columns" else {"x": "categorical"},
    )
    with pytest.raises(ValueError):
        global_idiosyncratic_index(ls, n_jobs=1, seed=0)


def table_landscape(configs, fitness):
    """A retained-data view; tests metric semantics without graph pruning."""
    X = pd.DataFrame(configs).rename(columns=lambda i: f"site_{i}")
    frame = X.assign(fitness=np.asarray(fitness, dtype=float))
    return SimpleNamespace(
        get_data=lambda: frame.copy(), data_types=dict.fromkeys(X, "categorical")
    )


def test_integer_additive_landscape_is_exactly_zero():
    X = np.asarray(list(product(range(2), repeat=4)))
    ls = table_landscape(X, X @ [1, 2, 4, 8])
    assert global_idiosyncratic_index(ls, n_jobs=1, seed=0) == 0.0


def test_large_indices_are_not_clipped():
    X = np.asarray(list(product(range(2), repeat=5)))
    ls = table_landscape(X, X.sum(axis=1) % 2)
    expected = literal_reference(ls, 0)
    assert expected > 1
    assert global_idiosyncratic_index(ls, n_jobs=1, seed=0) == pytest.approx(expected)


@pytest.mark.parametrize("scale,offset", [(8, 32), (-8, 32)])
def test_affine_fitness_invariance(categorical, scale, offset):
    frame = categorical.get_data()
    X = frame[list(categorical.data_types)]
    changed = Landscape(maximize=scale > 0).build_from_data(
        X,
        scale * frame.fitness + offset,
        data_types=categorical.data_types,
        verbose=False,
    )
    assert global_idiosyncratic_index(changed, n_jobs=1, seed=7) == pytest.approx(
        global_idiosyncratic_index(categorical, n_jobs=1, seed=7),
        rel=1e-12,
    )


@pytest.mark.parametrize("minimum", [2, np.int64(3), 4])
def test_background_threshold_boundary(minimum):
    X = [[a, b] for a in (0, 1) for b in range(3)]
    ls = table_landscape(X, [0, 1, 4, 2, 6, 7])
    expected = literal_reference(ls, 23, minimum)
    actual = global_idiosyncratic_index(ls, n_jobs=1, seed=23, min_pairs=minimum)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


def test_two_backgrounds_can_give_a_defined_single_mutation_index(monkeypatch):
    import graphfla.analysis.epistasis.idiosyncrasy as module

    class FixedControl:
        def __init__(self, seed=None):
            pass

        def choice(self, pool, size, replace):
            assert size == (2, 2) and replace
            return pool[[[0, 1], [0, 3]]]

    ls = table_landscape([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 2, 4])
    monkeypatch.setattr(module.np.random, "RandomState", FixedControl)
    assert idiosyncratic_index(ls, (0, "site_0", 1), min_pairs=2) == pytest.approx(
        1 / 3
    )
    assert np.isnan(idiosyncratic_index(ls, (0, "site_0", 1), min_pairs=3))


def test_no_backgrounds_or_no_mutations_are_undefined():
    for X, f in [
        ([], []),
        ([[0]], [1]),
        ([[0], [1]], [0, 1]),
        ([[0, 0], [1, 1]], [0, 1]),
    ]:
        ls = table_landscape(X, f)
        assert np.isnan(global_idiosyncratic_index(ls, n_jobs=1, seed=0, min_pairs=2))
    ls = table_landscape([[0, 0], [1, 1]], [0, 1])
    assert np.isnan(idiosyncratic_index(ls, (0, "site_0", 1), min_pairs=2))


def test_invariant_features_do_not_add_mutations_or_consume_control_draws():
    X = np.asarray(list(product(range(2), repeat=4)))
    f = np.random.RandomState(11).normal(size=len(X))
    base = table_landscape(X, f)
    padded = table_landscape(np.column_stack([np.zeros(len(X)), X]), f)
    assert global_idiosyncratic_index(
        base, n_jobs=1, seed=7
    ) == global_idiosyncratic_index(padded, n_jobs=1, seed=7)


def test_isolated_genotype_still_belongs_to_the_control_pool():
    X = np.asarray(list(product(range(2), repeat=3)))
    f = [0, 1, 2, 3, 1, 2, 4, 6]
    base = table_landscape(X, f)
    full = table_landscape(np.vstack([X, [2, 2, 2]]), f + [50])
    expected = literal_reference(full, 7)
    actual = global_idiosyncratic_index(full, n_jobs=1, seed=7)
    assert actual == pytest.approx(expected, rel=1e-12)
    assert actual != pytest.approx(global_idiosyncratic_index(base, n_jobs=1, seed=7))


def test_unbuilt_landscape_is_rejected():
    with pytest.raises(RuntimeError):
        global_idiosyncratic_index(BooleanLandscape(), n_jobs=1, seed=0)
    with pytest.raises(RuntimeError):
        idiosyncratic_index(BooleanLandscape(), (0, "bit_0", 1))
