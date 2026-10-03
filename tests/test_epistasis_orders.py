"""Order attribution, integrated results and reuse without a second fit."""

from itertools import product
import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

from graphfla.analysis import higher_order_epistasis, profile, walsh_hadamard
from graphfla.landscape import BooleanLandscape, Landscape
from validation.oracles.walsh import regression_design, transform


def build(X, y):
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(np.asarray(X).shape[1])])
    return Landscape().build_from_data(
        frame,
        y,
        data_types=dict.fromkeys(frame, "categorical"),
        epsilon=0,
        verbose=False,
    )


def label(term):
    return "-".join(f"0_{i + 1}_{a}" for i, a in enumerate(term) if a) or "WT"


def test_known_binary_variance_spectrum_and_cumulative_gains():
    X = np.array(list(product(range(2), repeat=3)))
    z = 2 * X - 1
    y = 5 + z[:, 0] + 2 * z[:, 0] * z[:, 1] + 3 * z.prod(axis=1)
    result = walsh_hadamard(build(X, y), max_order=3)
    assert isinstance(result, Bunch)
    summary = result.order_summary
    assert summary.order.tolist() == [0, 1, 2, 3]
    assert summary.r2.tolist() == pytest.approx([0, 1 / 14, 5 / 14, 1])
    assert summary.delta_r2.tolist() == pytest.approx([0, 1 / 14, 4 / 14, 9 / 14])
    assert summary.model_variance_fraction.tolist() == pytest.approx(
        [0, 1 / 14, 4 / 14, 9 / 14]
    )
    assert summary.n_terms.tolist() == [1, 4, 7, 8]
    assert summary.n_nonzero.isna().all()  # Sparsity is reported only for Lasso.
    assert summary.rmse.iloc[-1] < 1e-13


@pytest.mark.parametrize("arities", [(3, 3), (2, 3, 4)])
def test_multistate_spectrum_matches_independent_full_space_predictions(arities):
    X, T = transform(arities)
    y = np.random.default_rng(23).normal(size=len(X))
    D = np.linalg.inv(T)
    expected_coefficients = T @ y
    degrees = np.count_nonzero(X, axis=1)
    expected = [0] + [
        np.var(D[:, degrees == k] @ expected_coefficients[degrees == k]) / np.var(y)
        for k in range(1, len(arities) + 1)
    ]
    result = walsh_hadamard(build(X, y), max_order=len(arities))
    np.testing.assert_allclose(
        result.order_summary.model_variance_fraction, expected, atol=1e-13
    )
    np.testing.assert_allclose(result.order_summary.delta_r2, expected, atol=1e-13)


def test_incomplete_profile_refits_instead_of_dropping_full_model_terms():
    X, _ = transform([3, 2, 2])
    X = np.array(X[:-1])
    y = np.random.default_rng(23).normal(size=len(X))
    result = walsh_hadamard(build(X, y), max_order=2)
    expected = []
    for k in range(3):
        _, D = regression_design([3, 2, 2], X, k)
        beta = np.linalg.lstsq(D, y, rcond=None)[0]
        expected.append(1 - np.sum((y - D @ beta) ** 2) / np.sum((y - y.mean()) ** 2))
    np.testing.assert_allclose(result.order_summary.r2, expected, atol=1e-13)
    terms, D = regression_design([3, 2, 2], X, 1)
    coefs = result.coefficients.set_index("term").coefficient
    dropped_prediction = D @ np.array([coefs[label(t)] for t in terms])
    dropped_r2 = 1 - np.sum((y - dropped_prediction) ** 2) / np.sum((y - y.mean()) ** 2)
    assert abs(dropped_r2 - expected[1]) > 1e-5


def test_lasso_model_variance_is_not_labeled_observed_explained_variance():
    result = walsh_hadamard(build([[0], [1]], [-1, 1]), method="lasso", alpha=0.25)
    row = result.order_summary.iloc[-1]
    assert row.r2 == pytest.approx(0.75)
    assert row.delta_r2 == pytest.approx(0.75)
    assert row.model_variance_fraction == pytest.approx(1)
    assert row.n_nonzero == 1


def test_existing_result_is_reused_without_encoding_or_fitting(monkeypatch):
    X = np.array(list(product(range(2), repeat=3)))
    result = walsh_hadamard(build(X, X.sum(axis=1)), max_order=3)

    def unexpected(*args, **kwargs):
        raise AssertionError("A cached result must not trigger a fit")

    monkeypatch.setattr(np.linalg, "lstsq", unexpected)
    core = importlib.import_module("graphfla.analysis.epistasis._walsh")
    monkeypatch.setattr(core, "_encode_input", unexpected)
    pd.testing.assert_frame_equal(higher_order_epistasis(result), result.order_summary)
    short = higher_order_epistasis(result, max_order=1)
    assert short.order.tolist() == [0, 1]
    short.loc[0, "r2"] = 123
    assert result.order_summary.r2.iloc[0] == 0
    with pytest.raises(ValueError, match="computed"):
        higher_order_epistasis(result, max_order=4)
    with pytest.raises(ValueError, match="Fitting parameters"):
        higher_order_epistasis(result, method="lasso")


def test_design_and_highest_order_are_not_computed_twice(monkeypatch):
    core = importlib.import_module("graphfla.analysis.epistasis._walsh")
    original = core._design_matrix
    calls = []

    def count(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(core, "_design_matrix", count)
    fits = []
    lstsq = np.linalg.lstsq

    def count_fit(A, *args, **kwargs):
        fits.append(A.shape[1])
        return lstsq(A, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", count_fit)
    X = np.array(list(product(range(2), repeat=4)))
    walsh_hadamard(build(X, np.random.default_rng(3).normal(size=16)), max_order=3)
    assert len(calls) == 1
    assert sorted(fits) == [5, 11, 15]


def test_score_only_compatibility_allows_rank_deficient_predictions():
    land = build([[0, 0], [0, 1], [1, 0]], [0, 1, 2])
    with pytest.raises(ValueError, match="underdetermined"):
        walsh_hadamard(land)
    result = higher_order_epistasis(land)
    assert result.r2.iloc[-1] == pytest.approx(1)
    assert result.model_variance_fraction.isna().all()
    assert result.iloc[-1]["rank"] < result.iloc[-1].n_terms


def test_constant_landscape_has_no_defined_variance_fractions():
    X = list(product(range(2), repeat=2))
    with pytest.warns(UserWarning, match="constant"):
        result = walsh_hadamard(build(X, [3] * 4))
    assert result.order_summary.r2.isna().all()
    assert result.order_summary.delta_r2.isna().all()
    assert result.order_summary.model_variance_fraction.isna().all()
    assert np.all(result.order_summary.rmse == 0)


def test_legacy_order_keyword_and_profile_scalar_contract():
    X = np.array(list(product(range(2), repeat=3)))
    land = build(X, X.sum(axis=1) + 3 * X[:, 0] * X[:, 1])
    expected = higher_order_epistasis(land, max_order=1)
    with pytest.warns(FutureWarning, match="order"):
        actual = higher_order_epistasis(land, order=1)
    pd.testing.assert_frame_equal(actual, expected)
    p = profile(
        land,
        include=["higher_order_epistasis"],
        params={"higher_order_epistasis": {"max_order": 1}},
        on_error="raise",
    )
    assert p.iloc[0] == pytest.approx(expected.r2.iloc[-1])


def test_tree_depth_gains_do_not_measure_interaction_order():
    from sklearn.tree import DecisionTreeRegressor

    X = np.array(list(product(range(2), repeat=4)))
    y = X.sum(axis=1)  # Purely additive: exactly zero order >=2.
    land = BooleanLandscape().build_from_data(X, y, verbose=False)
    summary = higher_order_epistasis(land, max_order=4)
    np.testing.assert_allclose(summary.delta_r2, [0, 1, 0, 0, 0], atol=1e-13)
    scores = [
        DecisionTreeRegressor(max_depth=k, random_state=0).fit(X, y).score(X, y)
        for k in range(1, 5)
    ]
    assert scores == pytest.approx([0.25, 0.5, 0.75, 1])


def test_truncated_model_spectrum_has_its_own_denominator_and_reuses_qr(monkeypatch):
    calls = []
    qr = np.linalg.qr

    def counted(matrix, **kwargs):
        calls.append(matrix.shape)
        return qr(matrix, **kwargs)

    monkeypatch.setattr(np.linalg, "qr", counted)
    X = np.array(list(product(range(2), repeat=6)))
    z = 2 * X - 1
    y = 2 * z[:, 0] + 3 * z[:, 0] * z[:, 1] + 4 * z[:, 0] * z[:, 1] * z[:, 2]
    summary = walsh_hadamard(build(X, y), max_order=2).order_summary
    np.testing.assert_allclose(summary.delta_r2, [0, 4 / 29, 9 / 29], atol=1e-13)
    np.testing.assert_allclose(
        summary.model_variance_fraction, [0, 4 / 13, 9 / 13], atol=1e-13
    )
    assert calls == [(64, 23)]


@pytest.mark.parametrize("scale", [1e-200, 1, 1e200])
def test_order_fractions_are_stable_in_extreme_fitness_units(scale):
    X = np.array(list(product(range(2), repeat=3)))
    z = 2 * X - 1
    y = scale * (2 * z[:, 0] + 3 * z[:, 0] * z[:, 1])
    result = walsh_hadamard(build(X, y))
    np.testing.assert_allclose(
        result.order_summary.delta_r2, [0, 4 / 13, 9 / 13], atol=1e-13
    )
    np.testing.assert_allclose(
        result.order_summary.model_variance_fraction, [0, 4 / 13, 9 / 13], atol=1e-13
    )
    assert np.isfinite(result.order_summary.rmse).all()


def test_lasso_cv_uses_the_same_folds_at_every_order(monkeypatch):
    core = importlib.import_module("graphfla.analysis.epistasis._walsh")
    solve = core._lasso_fit
    splits = []

    def capture(*args, **kwargs):
        splits.append(kwargs["splits"])
        return solve(*args, **kwargs)

    monkeypatch.setattr(core, "_lasso_fit", capture)
    X = np.array(list(product(range(2), repeat=4)))
    result = walsh_hadamard(
        build(X, np.random.default_rng(3).normal(size=16)),
        max_order=3,
        method="lasso",
        cv=3,
        random_state=None,
    )
    assert len(splits) == 3 and all(s is splits[0] for s in splits)
    assert result.order_summary.alpha.iloc[1:].gt(0).all()


def test_compatibility_rejects_ambiguous_order_names_and_bad_jobs():
    land = build([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="only max_order"):
        higher_order_epistasis(land, max_order=1, order=1)
    with pytest.raises(ValueError, match="n_jobs"):
        walsh_hadamard(land, n_jobs=0)


def test_regularized_negative_gain_matches_independent_refits():
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold

    X = np.array(list(product(range(3), range(2), range(2))))[:-2]
    y = np.random.default_rng(2).normal(size=len(X))
    result = walsh_hadamard(build(X, y), max_order=2, method="lasso", cv=3, tol=1e-12)
    reference = [0.0]
    for k in (1, 2):
        _, D = regression_design([3, 2, 2], X, k)
        features = D[:, 1:]
        largest = np.max(abs(features.T @ (y - y.mean()))) / len(y)
        model = LassoCV(
            alphas=largest * np.geomspace(1, 1e-3, 100),
            cv=KFold(3, shuffle=True, random_state=0),
            tol=1e-12,
            max_iter=10000,
        ).fit(features, y)
        reference.append(model.score(features, y))
    np.testing.assert_allclose(result.order_summary.r2, reference, atol=1e-9)
    assert result.order_summary.delta_r2.iloc[-1] < -0.1
    assert result.order_summary.delta_r2.iloc[-1] == pytest.approx(
        reference[-1] - reference[-2], abs=1e-9
    )


def test_invariant_imported_columns_do_not_trigger_combinatorial_work(monkeypatch):
    core = importlib.import_module("graphfla.analysis.epistasis._walsh")
    choose = core.combinations

    def bounded(sites, order):
        assert len(sites) <= 2, "Invariant sites must be excluded before enumeration"
        return choose(sites, order)

    monkeypatch.setattr(core, "combinations", bounded)
    X = np.column_stack(
        [list(product(range(2), repeat=2)), np.zeros((4, 32), dtype=int)]
    )
    land = build(X, [0, 1, 2, 4])
    # Imported/filtered graphs may retain metadata for now-invariant variables.
    land.data_types = dict.fromkeys([f"x{i}" for i in range(34)], "categorical")
    result = walsh_hadamard(land, max_order=17)
    assert result.fit_info["max_order"] == 2
    assert len(result.coefficients) == 4


def test_ols_order_spectrum_is_invariant_to_multistate_reference_change():
    X = np.array(list(product(range(3), range(4))))
    y = np.random.default_rng(8).normal(size=len(X))
    original = walsh_hadamard(build(X, y))
    perm = np.r_[len(X) - 1, np.arange(len(X) - 1)]
    reordered = walsh_hadamard(build(X[perm], y[perm]))
    assert original.fit_info["reference"] != reordered.fit_info["reference"]
    np.testing.assert_allclose(
        original.order_summary.model_variance_fraction,
        reordered.order_summary.model_variance_fraction,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        original.order_summary.delta_r2, reordered.order_summary.delta_r2, atol=1e-13
    )
