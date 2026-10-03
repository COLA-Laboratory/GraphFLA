"""Analytic transforms and fitting contracts on small, general landscapes."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from graphfla.analysis import walsh_hadamard
from graphfla.landscape import BooleanLandscape, DNALandscape, Landscape
from validation.oracles.walsh import regression_design, transform


def table_landscape(X, y, kind="categorical", kinds=None):
    frame = pd.DataFrame(X).copy()
    kinds = kinds or dict.fromkeys(frame.columns, "categorical")
    frame["fitness"] = y
    return SimpleNamespace(
        get_data=lambda: frame.copy(),
        data_types=kinds,
        kind=kind,
        _check_built=lambda: None,
        graph=SimpleNamespace(vs=SimpleNamespace(attributes=lambda: ["fitness"])),
        n_vars=len(kinds),
        configs=None,
    )


def keyed_coefficients(table):
    return dict(zip(table.term, table.coefficient))


def term_label(term):
    return "-".join(f"0_{j + 1}_{a}" for j, a in enumerate(term) if a) or "WT"


@pytest.mark.parametrize("arities", [(2, 2), (3, 3), (2, 3, 4)])
def test_complete_transform_matches_background_differences(arities):
    X, T = transform(arities)
    y = np.random.default_rng(42).normal(size=len(X))
    actual = keyed_coefficients(walsh_hadamard(table_landscape(X, y), len(arities)))
    expected = dict(zip(map(term_label, X), T @ y))
    assert actual == pytest.approx(expected, abs=2e-13)


def test_original_sequence_positions_survive_invariant_removal():
    land = DNALandscape().build_from_data(
        ["GAC", "GCC"], [0, 3], epsilon=0, verbose=False
    )
    table = walsh_hadamard(land)
    assert table.positions.tolist() == [(), (2,)]
    assert keyed_coefficients(table) == pytest.approx({"WT": 1.5, "A_2_C": 3})


def test_original_categorical_labels_and_invariant_columns():
    X = pd.DataFrame({"fixed": ["x"] * 3, "material": ["iron", "copper", "gold"]})
    land = Landscape().build_from_data(
        X, [0, 2, 7], data_types=dict.fromkeys(X, "categorical"), verbose=False
    )
    actual = walsh_hadamard(land)
    assert actual.positions.tolist() == [(), (2,), (2,)]
    assert keyed_coefficients(actual) == pytest.approx(
        {"WT": 3, "iron_2_copper": 2, "iron_2_gold": 7}
    )


def test_large_alphabet_and_delimiters_do_not_control_computation():
    labels = [f"state_{i}-x%" for i in range(64)]
    land = table_landscape({"component": labels}, np.arange(64) ** 2)
    actual = walsh_hadamard(land, max_order=1)
    assert len(actual) == 64 and actual.term.is_unique
    assert actual.positions.tolist().count((1,)) == 63
    assert actual.set_index("term").loc[
        "state%5F0%2Dx%25_1_state%5F47%2Dx%25", "coefficient"
    ] == pytest.approx(47**2)


@pytest.mark.parametrize("max_order", [0, 1, 2])
def test_incomplete_identifiable_fit_matches_inverse_oracle(max_order):
    X = np.array(list(product(range(2), range(3), range(2))))
    X = X[:-1]
    y = np.random.default_rng(9).normal(size=len(X))
    terms, design = regression_design([2, 3, 2], X, max_order)
    coefs = np.linalg.lstsq(design, y, rcond=None)[0]
    actual = keyed_coefficients(walsh_hadamard(table_landscape(X, y), max_order))
    assert actual == pytest.approx(dict(zip(map(term_label, terms), coefs)), abs=2e-13)


def test_underdetermined_fit_raises_with_counts():
    land = table_landscape([[0, 0], [0, 1], [1, 0]], [0, 1, 2])
    with pytest.raises(ValueError, match="n_samples=3.*n_terms=4"):
        walsh_hadamard(land)


def test_rank_deficiency_is_detected_even_with_enough_observations():
    X = [(a, a, b) for a, b in product(range(2), range(3))]
    with pytest.raises(ValueError, match="rank=.*n_terms=5"):
        walsh_hadamard(table_landscape(X, np.arange(6)), max_order=1)


def test_lasso_matches_closed_form_soft_threshold_and_unpenalized_mean():
    X = np.array(list(product([0, 1], repeat=3)))
    z = X - 0.5
    y = 7 + 4 * z[:, 0] - 2 * z[:, 1] + 8 * z[:, 0] * z[:, 1]
    actual = walsh_hadamard(table_landscape(X, y), method="lasso", alpha=0.1, tol=1e-12)
    # Complete binary W-H features are orthogonal. Each slope is soft-thresholded
    # by alpha / mean(feature**2), giving thresholds 0.4 and 1.6 by order.
    expected = {
        "WT": 7,
        "0_1_1": 3.6,
        "0_2_1": -1.6,
        "0_3_1": 0,
        "0_1_1-0_2_1": 6.4,
        "0_1_1-0_3_1": 0,
        "0_2_1-0_3_1": 0,
    }
    assert keyed_coefficients(actual) == pytest.approx(expected, abs=1e-12)


def test_lasso_accepts_underdetermined_input_without_claiming_identifiability():
    land = table_landscape([[0, 0], [0, 1], [1, 0]], [0, 1, 2])
    actual = walsh_hadamard(land, method="lasso", alpha=0.01)
    assert np.isfinite(actual.coefficient).all()
    assert actual.attrs["fit_info"]["method"] == "lasso"
    assert actual.attrs["fit_info"]["rank"] is None


def test_allocation_guard_precedes_term_enumeration():
    land = table_landscape(np.eye(72, dtype=int), np.arange(72))
    with pytest.raises(ValueError, match="max_cells"):
        walsh_hadamard(land, max_order=36, max_cells=1000)


@pytest.mark.parametrize("maximize", [False, True])
def test_boolean_zero_reference_and_fitness_direction(maximize):
    land = BooleanLandscape(maximize=maximize).build_from_data(
        ["11", "10", "01", "00"], [5, 2, 3, 0], verbose=False
    )
    assert keyed_coefficients(walsh_hadamard(land)) == pytest.approx(
        {"WT": 2.5, "0_1_1": 2, "0_2_1": 3, "0_1_1-0_2_1": 0}, abs=1e-13
    )


@pytest.mark.parametrize("scale", [1e-200, 1, 1e200])
@pytest.mark.parametrize("method", ["ols", "lasso"])
def test_fitness_units_and_offset(scale, method):
    X = np.array(list(product([0, 1], repeat=3)))
    y = 8 + 4 * (X[:, 0] - 0.5) + 8 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
    result = walsh_hadamard(
        table_landscape(X, y * scale), method=method, alpha=0.1 * scale, tol=1e-12
    )
    values = {k: v / scale for k, v in keyed_coefficients(result).items()}
    assert values["WT"] == pytest.approx(8)
    assert values["0_1_1"] == pytest.approx(4 if method == "ols" else 3.6)
    assert values["0_1_1-0_2_1"] == pytest.approx(8 if method == "ols" else 6.4)


@pytest.mark.parametrize(
    "method,alpha", [("ols", "cv"), ("lasso", "cv"), ("lasso", 0.1)]
)
def test_constant_fitness_and_zero_order(method, alpha):
    X = list(product(range(2), repeat=3))
    for order in (0, 2, 99):
        result = walsh_hadamard(
            table_landscape(X, [3.5] * 8), order, method=method, alpha=alpha
        )
        assert result.loc[result.order == 0, "coefficient"].item() == pytest.approx(3.5)
        assert np.allclose(result.loc[result.order > 0, "coefficient"], 0, atol=1e-14)


def test_cv_selection_matches_explicit_fold_search_on_independent_design():
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import KFold

    X = np.array(list(product(range(3), range(2), range(2))))
    y = np.random.default_rng(5).normal(size=len(X))
    terms, D = regression_design([3, 2, 2], X, 2)
    mask = np.array([any(t) for t in terms])
    features = D[:, mask]
    alpha_max = np.max(abs(features.T @ (y - y.mean()))) / len(y)
    grid = alpha_max * np.geomspace(1, 1e-3, 100)
    splits = list(KFold(3, shuffle=True, random_state=17).split(X))
    errors = []
    for a in grid:
        errors.append(
            np.mean(
                [
                    np.mean(
                        (
                            Lasso(alpha=a, tol=1e-11, max_iter=10000)
                            .fit(features[train], y[train])
                            .predict(features[test])
                            - y[test]
                        )
                        ** 2
                    )
                    for train, test in splits
                ]
            )
        )
    chosen = grid[np.argmin(errors)]
    model = Lasso(alpha=chosen, tol=1e-11, max_iter=10000).fit(features, y)
    expected = dict(zip(map(term_label, terms), np.r_[model.intercept_, model.coef_]))
    actual = walsh_hadamard(
        table_landscape(X, y),
        method="lasso",
        alpha="cv",
        cv=3,
        random_state=17,
        tol=1e-11,
    )
    assert actual.attrs["fit_info"]["alpha"] == pytest.approx(chosen)
    assert keyed_coefficients(actual) == pytest.approx(expected, abs=1e-8)
    repeated = walsh_hadamard(
        table_landscape(X, y), method="lasso", cv=3, random_state=17, tol=1e-11
    )
    pd.testing.assert_frame_equal(actual, repeated)


def test_chunking_mixed_variables_and_unused_categories():
    X = pd.DataFrame(
        product(["a", "b", "c"], [1, 5, 50], [False, True]),
        columns=["mode", "level", "on"],
    )
    X["mode"] = pd.Categorical(X["mode"], categories=["a", "b", "c", "unused"])
    y = np.random.default_rng(0).normal(size=len(X))
    kinds = {"mode": "categorical", "level": "ordinal", "on": "boolean"}
    land = Landscape().build_from_data(X, y, data_types=kinds, verbose=False)
    small = walsh_hadamard(land, chunk_size=1)
    large = walsh_hadamard(land, chunk_size=1000)
    pd.testing.assert_frame_equal(small, large)
    assert len(small) == 14  # 1 + (2+2+1) + (4+2+2).
    assert small.attrs["reference"] == {1: "a", 2: 1, 3: False}
    assert small.attrs["position_labels"] == {1: "mode", 2: "level", 3: "on"}
    # Numeric spacing of ordinal labels does not alter this discrete basis.
    remapped = X.copy()
    remapped["level"] = remapped.level.map({1: "a", 5: "b", 50: "c"})
    numeric = walsh_hadamard(table_landscape(X, y, kinds=kinds))
    text = walsh_hadamard(table_landscape(remapped, y, kinds=kinds))
    expected = {
        key.replace("1_2_50", "a_2_c").replace("1_2_5", "a_2_b"): value
        for key, value in keyed_coefficients(numeric).items()
    }
    assert keyed_coefficients(text) == pytest.approx(expected, abs=1e-13)


def test_mixed_type_alleles_have_distinct_labels():
    result = walsh_hadamard(table_landscape({"x": [0, 1, "1"]}, [0, 2, 3]), max_order=1)
    assert result.term.is_unique
    assert keyed_coefficients(result) == pytest.approx(
        {"WT": 5 / 3, "0_1_int:1": 2, "0_1_str:1": 3}
    )


def test_invariant_graph_column_and_graph_round_trip(tmp_path):
    land = DNALandscape().build_from_data(["GAC", "GCC"], [0, 3], verbose=False)
    path = tmp_path / "landscape.graphml"
    land.to_graph(str(path))
    restored = DNALandscape.build_from_graph(str(path), verbose=False)
    pd.testing.assert_frame_equal(walsh_hadamard(land), walsh_hadamard(restored))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_order": -1}, "max_order"),
        ({"max_order": True}, "max_order"),
        ({"max_order": 1.5}, "max_order"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"max_cells": np.inf}, "max_cells"),
        ({"max_cells": 0}, "max_cells"),
        ({"method": "auto"}, "method"),
        ({"method": "lasso", "alpha": 0}, "alpha"),
        ({"method": "lasso", "alpha": np.nan}, "alpha"),
        ({"method": "lasso", "alpha": "auto"}, "alpha"),
        ({"method": "lasso", "cv": 1}, "cv"),
        ({"method": "lasso", "cv": 5}, "n_samples=4"),
        ({"method": "lasso", "max_iter": 0}, "max_iter"),
        ({"method": "lasso", "tol": 0}, "tol"),
    ],
)
def test_invalid_parameters(kwargs, match):
    land = table_landscape(list(product([0, 1], repeat=2)), [0, 1, 2, 4])
    with pytest.raises(ValueError, match=match):
        walsh_hadamard(land, **kwargs)


@pytest.mark.parametrize(
    "X,y,match",
    [
        ([[0], [1]], [0, np.inf], "finite"),
        ([[0], [None]], [0, 1], "missing"),
        ([[0], [0]], [0, 1], "unique"),
        (pd.DataFrame(columns=["x"]), [], "n_samples=0"),
    ],
)
def test_invalid_data(X, y, match):
    with pytest.raises(ValueError, match=match):
        walsh_hadamard(table_landscape(X, y))


def test_unbuilt_and_missing_metadata():
    with pytest.raises(RuntimeError):
        walsh_hadamard(BooleanLandscape())
    land = table_landscape([[0], [1]], [0, 1])
    land.data_types = None
    with pytest.raises(ValueError, match="columns"):
        walsh_hadamard(land)


def test_lasso_convergence_warning_is_not_hidden():
    from sklearn.exceptions import ConvergenceWarning

    X = list(product(range(3), repeat=3))[:-4]
    y = np.random.default_rng(77).normal(size=len(X))
    with pytest.warns(ConvergenceWarning):
        walsh_hadamard(
            table_landscape(X, y), method="lasso", alpha=1e-6, max_iter=1, tol=1e-15
        )


def test_long_low_order_space_does_not_multiply_all_state_counts():
    # Product-space size 2**72, but this observed design has only 73 columns.
    X = np.vstack([np.zeros(72), np.eye(72)])
    result = walsh_hadamard(
        table_landscape(
            X, X.sum(axis=1), kind="boolean", kinds=dict.fromkeys(range(72), "boolean")
        ),
        max_order=1,
    )
    assert result.loc[result.order == 0, "coefficient"].item() == pytest.approx(36)
    assert np.allclose(result.loc[result.order == 1, "coefficient"], 1)


def test_coefficients_outside_float64_raise_instead_of_returning_infinity():
    with pytest.raises(ValueError, match="float64"):
        walsh_hadamard(table_landscape([[0], [1]], [-1e308, 1e308]))


def test_extreme_lasso_penalties_have_explicit_behavior():
    land = table_landscape([[0], [1]], [0, 1e-300])
    result = walsh_hadamard(land, method="lasso", alpha=1e300)
    assert result.loc[result.order == 1, "coefficient"].item() == 0
    land = table_landscape([[0], [1]], [0, 1e300])
    with pytest.raises(ValueError, match="alpha is too small"):
        walsh_hadamard(land, method="lasso", alpha=1e-300)


def test_additive_model_has_exact_main_effects_and_zero_interactions():
    X = list(product(range(3), range(2), range(2)))
    y = [2 * a + 3 * b - 4 * c for a, b, c in X]
    table = walsh_hadamard(table_landscape(X, y))
    expected = {term: 0.0 for term in table.term}
    expected.update({"WT": 1.5, "0_1_1": 2, "0_1_2": 4, "0_2_1": 3, "0_3_1": -4})
    assert keyed_coefficients(table) == pytest.approx(expected, abs=1e-13)
