"""Independent additive-model checks for general combinatorial landscapes."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from graphfla import analysis
from graphfla.analysis import _roughness as impl
from graphfla.landscape import BooleanLandscape, Landscape, OrdinalLandscape


def table_landscape(X, y, kinds=None):
    """Minimal data protocol for solver boundaries, without graph filtering."""
    frame = pd.DataFrame(X).copy()
    kinds = kinds or dict.fromkeys(frame.columns, "boolean")
    frame["fitness"] = y
    return SimpleNamespace(get_data=lambda: frame.copy(), data_types=kinds)


@pytest.mark.parametrize(
    "scale,offset",
    [
        (1, 0),
        (-3, 17),
        (1e-200, 0),
        (1e200, 0),
        (1e-308, 0),
        (1e306, 0),
        (1, 1e13),
        (1, -1e13),
    ],
)
def test_known_nonzero_residual_is_affine_invariant(scale, offset):
    X = np.array(list(product([0, 1], repeat=3)))
    z = 2 * X - 1
    y = 2 * z[:, 0] + 4 * z[:, 1] + 6 * z[:, 2] + 3 * z[:, 0] * z[:, 1]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = analysis.r_s_ratio(table_landscape(X, scale * y + offset))
    assert type(actual) is float
    assert actual == pytest.approx(3 / 8, rel=1e-12)


def test_opposite_finite_extremes_do_not_overflow_range():
    X = np.array(list(product([0, 1], repeat=2)))
    assert analysis.r_s_ratio(table_landscape(X, [-1e308, 0, 0, 1e308])) < 1e-14


@pytest.mark.parametrize("constant", [0, 0.1, 1e300])
def test_constant_objective_is_undefined(constant):
    with pytest.warns(UserWarning, match="constant"):
        assert np.isnan(analysis.r_s_ratio(table_landscape([[0], [1]], [constant] * 2)))


@pytest.mark.parametrize("scale", [1, 1e-300, 1e300])
def test_pure_interaction_has_infinite_ratio(scale):
    with pytest.warns(UserWarning, match="zero or near zero"):
        assert (
            analysis.r_s_ratio(
                table_landscape(
                    list(product([0, 1], repeat=2)), np.array([0, 1, 1, 0]) * scale
                )
            )
            == np.inf
        )


@pytest.mark.parametrize("maximize", [True, False])
def test_public_build_profile_and_binary_label_reversal(maximize):
    X = np.array(list(product([0, 1], repeat=2)))
    y = [0, 1, 2, 4]
    land = BooleanLandscape(maximize=maximize).build_from_data(X, y, verbose=False)
    actual = analysis.r_s_ratio(land)
    assert actual == pytest.approx(1 / 8)
    reverse = BooleanLandscape().build_from_data(1 - X, y, verbose=False)
    assert analysis.r_s_ratio(reverse) == pytest.approx(actual)
    assert analysis.profile(land, metrics=["r_s_ratio"])["r_s_ratio"] == pytest.approx(
        actual
    )
    assert "r_s_ratio" in analysis.list_metrics().index


def test_multistate_reference_changes_slope_but_not_roughness():
    # Two backgrounds for each of three states. State means 0, 1, 4;
    # background coefficient 2; residuals (+1,-1),(-1,+1),(0,0).
    X = pd.DataFrame(product(["a", "b", "c"], [False, True]), columns=["mode", "on"])
    y = [1, 1, 0, 4, 4, 6]
    kinds = {"mode": "categorical", "on": "boolean"}
    reference_a = table_landscape(X, y, kinds)
    actual = analysis.r_s_ratio(reference_a)
    assert actual == pytest.approx(np.sqrt(2 / 3) / (7 / 3))
    # Relabel c to be the first category. New contrasts -4, -3, +2.
    X["mode"] = X["mode"].map({"a": "b", "b": "c", "c": "a"})
    assert analysis.r_s_ratio(table_landscape(X, y, kinds)) == pytest.approx(
        np.sqrt(2 / 3) / 3
    )


def test_ordinal_ranks_are_not_physical_spacing_or_categorical_effects():
    X = pd.DataFrame(
        {
            "level": pd.Categorical(
                ["low", "medium", "high"],
                categories=["low", "medium", "high"],
                ordered=True,
            )
        }
    )
    land = OrdinalLandscape().build_from_data(X, [0, 1, 4], verbose=False)
    # Single-variable curvature contributes residuals (1,-2,1)/3.
    assert analysis.r_s_ratio(land) == pytest.approx(np.sqrt(2) / 6)
    land._configs_array = None
    # Fallback follows category order of get_data, which contains plain labels.
    data = land.get_data()
    codes = pd.Categorical(data["level"], ordered=True).codes
    slope = np.cov(codes, data.fitness, ddof=0)[0, 1] / np.var(codes)
    residual = data.fitness - data.fitness.mean() - slope * (codes - codes.mean())
    assert analysis.r_s_ratio(land) == pytest.approx(
        np.sqrt(np.mean(residual**2)) / abs(slope)
    )


@pytest.mark.parametrize("force_blocks", [False, True])
def test_incomplete_mixed_model_matches_independent_svd(force_blocks, monkeypatch):
    rng = np.random.default_rng(8)
    X = pd.DataFrame(
        product(["a", "b", "c"], range(4), [False, True]),
        columns=["mode", "level", "on"],
    )
    X = X.drop(index=[1, 7, 12, 19]).reset_index(drop=True)
    y = rng.normal(size=len(X)) + X.level
    kinds = {"mode": "categorical", "level": "ordinal", "on": "boolean"}
    land = Landscape().build_from_data(X, y, data_types=kinds, verbose=False)
    frame = land.get_data()
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["mode"] == "b",
            frame["mode"] == "c",
            frame.level,
            frame.on,
        ]
    ).astype(float)
    coef = np.linalg.pinv(design) @ frame.fitness.to_numpy()
    expected = np.sqrt(np.mean((frame.fitness - design @ coef) ** 2)) / np.mean(
        abs(coef[1:])
    )
    if force_blocks:
        monkeypatch.setattr(impl, "_DESIGN_BYTES", 256)
    assert analysis.r_s_ratio(land) == pytest.approx(expected, abs=1e-12)


def test_constants_and_unused_categories_do_not_dilute_slope():
    X = pd.DataFrame(list(product([0, 1], repeat=2)))
    X[2] = pd.Categorical(["a"] * 4, categories=["a", "unobserved"])
    X[3] = 1
    X[4] = 0
    land = table_landscape(
        X,
        [0, 1, 2, 4],
        {0: "boolean", 1: "boolean", 2: "categorical", 3: "boolean", 4: "ordinal"},
    )
    assert analysis.r_s_ratio(land) == pytest.approx(1 / 8)


@pytest.mark.parametrize("force_blocks", [False, True])
def test_rank_deficiency_even_with_more_rows_than_coefficients(
    force_blocks, monkeypatch
):
    X = pd.DataFrame(list(product([0, 1], repeat=3)))
    X[3] = X[0]
    if force_blocks:
        monkeypatch.setattr(impl, "_DESIGN_BYTES", 100)
    with pytest.warns(UserWarning, match="rank deficient"):
        assert np.isnan(analysis.r_s_ratio(table_landscape(X, range(8))))


def test_underdetermined_model_is_rejected_before_design_allocation(monkeypatch):
    monkeypatch.setattr(impl, "_design", lambda *a: pytest.fail("allocated design"))
    with pytest.warns(UserWarning, match="not identifiable"):
        assert np.isnan(
            analysis.r_s_ratio(table_landscape([[0] * 100, [1] * 100], [0, 1]))
        )


def test_saturated_fit_is_explicit():
    with pytest.warns(UserWarning, match="saturated"):
        assert (
            analysis.r_s_ratio(table_landscape([[0, 0], [0, 1], [1, 0]], [0, 1, 2]))
            < 1e-14
        )


def test_workspace_limit_before_allocation(monkeypatch):
    monkeypatch.setattr(impl, "_QR_BYTES", 8)
    with pytest.warns(UserWarning, match="workspace limit"):
        assert np.isnan(analysis.r_s_ratio(table_landscape([[0], [1]], [0, 1])))


def test_numerical_solver_failure_is_explicit(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD failed")

    monkeypatch.setattr(impl, "lstsq", fail)
    with pytest.warns(UserWarning, match="did not converge"):
        assert np.isnan(analysis.r_s_ratio(table_landscape([[0], [1]], [0, 1])))


@pytest.mark.parametrize("y", [[0, np.nan], [0, np.inf]])
def test_nonfinite_objective_rejected(y):
    with pytest.raises(ValueError, match="finite"):
        analysis.r_s_ratio(table_landscape([[0], [1]], y))


def test_empty_population_is_undefined():
    with pytest.warns(UserWarning, match="no configurations"):
        assert np.isnan(
            analysis.r_s_ratio(table_landscape(pd.DataFrame(columns=[0]), []))
        )


def test_unsupported_type_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        analysis.r_s_ratio(table_landscape([[0], [1]], [0, 1], {0: "continuous"}))


def test_misaligned_ordinal_cache_uses_observed_category_order():
    land = table_landscape([[10], [20], [100]], [0, 1, 4], {0: "ordinal"})
    land._configs_array = np.array([[1000], [2000]])
    assert analysis.r_s_ratio(land) == pytest.approx(np.sqrt(2) / 6)


def test_missing_categorical_state_is_not_an_implicit_reference():
    land = table_landscape([["a"], [None], ["b"]], [0, 1, 4], {0: "categorical"})
    with pytest.raises(ValueError, match="must not be missing"):
        analysis.r_s_ratio(land)


def test_nonfinite_ordinal_codes_rejected_before_solver():
    land = table_landscape([[10], [20], [100]], [0, 1, 4], {0: "ordinal"})
    land._configs_array = np.array([[0], [np.nan], [2]])
    with pytest.raises(ValueError, match="codes must be finite"):
        analysis.r_s_ratio(land)


def test_unbuilt_landscape_raises():
    with pytest.raises(RuntimeError):
        analysis.r_s_ratio(Landscape())
