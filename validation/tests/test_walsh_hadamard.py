"""Faure et al. (2024): printed targets, independent equations and author replay."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

from graphfla.analysis import walsh_hadamard
from graphfla.landscape import DNALandscape, Landscape
from validation.oracles.walsh import regression_design, transform
from validation.testing import assert_case_matches


SYNTHETIC_CASES = [
    "faure.multistate.walsh.v1",
    "faure.author.matrix.v1",
    "faure.author.lasso.v1",
]


def label(term):
    return "-".join(f"0_{j + 1}_{a}" for j, a in enumerate(term) if a) or "WT"


def coefficients(table):
    return dict(zip(table.term, table.coefficient))


def landscape(X, y):
    frame = pd.DataFrame(X, columns=["component", "level", "switch"])
    return Landscape().build_from_data(
        frame,
        y,
        data_types={
            "component": "categorical",
            "level": "ordinal",
            "switch": "categorical",
        },
        epsilon=0,
        verbose=False,
    )


@pytest.fixture(scope="module")
def synthetic(verified_literature_inputs):
    case = next(c for c in SYNTHETIC_CASES if c in verified_literature_inputs)
    data = json.loads(verified_literature_inputs[case][0].read_text())
    X, y = np.array(data["configurations"]), np.array(data["fitness"])
    return data["arities"], X, y


@pytest.fixture(scope="module")
def author(verified_literature_inputs):
    case = next(
        c
        for c in ("faure.author.matrix.v1", "faure.author.lasso.v1")
        if c in verified_literature_inputs
    )
    path = verified_literature_inputs[case][1]
    # Audited, MIT-licensed pure functions; this fixed path has already passed
    # the literature contract's mandatory SHA-256 check. No downloads occur.
    spec = importlib.util.spec_from_file_location("faure_author_matrices", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.literature_case("faure.table1.walsh.v1", role="paper_result")
def test_table1_published_coefficients(literature_inputs):
    frame = pd.read_csv(literature_inputs["faure.table1.walsh.v1"][0])
    sequences = ["A" * 5 + a + "A" * 59 + b for a, b in frame.sequence]
    land = DNALandscape().build_from_data(
        sequences, frame.fitness, epsilon=0, verbose=False
    )
    result = walsh_hadamard(land, max_order=2)
    assert_case_matches("faure.table1.walsh.v1", coefficients(result))
    assert {p for positions in result.positions for p in positions} == {6, 66}
    assert len(result) == land.n_configs == 9
    _, T = transform([3, 3])
    expected = T @ frame.fitness.to_numpy()
    names = [
        "WT",
        "C_66_A",
        "C_66_T",
        "G_6_A",
        "G_6_A-C_66_A",
        "G_6_A-C_66_T",
        "G_6_T",
        "G_6_T-C_66_A",
        "G_6_T-C_66_T",
    ]
    np.testing.assert_allclose(
        result.set_index("term").loc[names, "coefficient"], expected, atol=2e-14, rtol=0
    )


@pytest.mark.literature_case("faure.multistate.walsh.v1", role="independent_check")
def test_mixed_state_background_differences(synthetic):
    arities, X, y = synthetic
    states, T = transform(arities)
    expected = dict(zip(map(label, states), map(float, T @ y)))
    assert_case_matches("faure.multistate.walsh.v1", expected)
    actual = walsh_hadamard(landscape(X, y), max_order=3)
    assert_case_matches("faure.multistate.walsh.v1", coefficients(actual))
    assert coefficients(actual) == pytest.approx(expected, abs=2e-12)


@pytest.mark.literature_case("faure.author.matrix.v1", role="author_result")
def test_author_matrices_and_incomplete_fit(synthetic, author):
    arities, X, y = synthetic
    states, T = transform(arities)
    strings = ["".join(map(str, row)) for row in states]
    # Author H takes genotype rows / coefficient columns; the forward
    # transform is transposed before left multiplication by V.
    H = author.H_matrix(strings, strings, num_states=arities)
    V = author.V_matrix(strings, num_states=arities)
    np.testing.assert_allclose(V @ H.T, T, rtol=0, atol=2e-15)
    author_design = author.H_matrix(
        strings, strings, num_states=arities, invert=True
    ) @ author.V_matrix(strings, num_states=arities, invert=True)
    np.testing.assert_allclose(author_design @ T, np.eye(len(X)), atol=1e-14)
    for arity in (2, 3):
        from itertools import product

        s = ["".join(map(str, row)) for row in product(range(arity), repeat=2)]
        np.testing.assert_allclose(
            author.H_matrix_recursive(2, arity), author.H_matrix(s, s, arity).T
        )
    terms, D = regression_design(arities, X[:-1], 2)
    columns = [states.index(t) for t in terms]
    selected = author_design[:-1, columns]
    reference = np.linalg.lstsq(selected, y[:-1], rcond=None)[0]
    actual = coefficients(walsh_hadamard(landscape(X[:-1], y[:-1]), max_order=2))
    errors = np.array([actual[label(t)] for t in terms]) - reference
    assert_case_matches(
        "faure.author.matrix.v1",
        {
            "max_design_error": float(np.max(abs(selected - D))),
            "max_coefficient_error": float(np.max(abs(errors))),
        },
    )


@pytest.mark.literature_case("faure.author.lasso.v1", role="author_result")
def test_author_lasso_on_centered_complete_design(synthetic, author):
    arities, X, y = synthetic
    strings = ["".join(map(str, row)) for row in X]
    D = author.H_matrix(
        strings, strings, num_states=arities, invert=True
    ) @ author.V_matrix(strings, num_states=arities, invert=True)
    model = Lasso(alpha=0.05, fit_intercept=False, max_iter=100000, tol=1e-12).fit(D, y)
    assert_case_matches(
        "faure.author.lasso.v1", dict(zip(map(label, X), map(float, model.coef_)))
    )
    actual = coefficients(
        walsh_hadamard(
            landscape(X, y),
            max_order=3,
            method="lasso",
            alpha=0.05,
            max_iter=100000,
            tol=1e-12,
        )
    )
    assert_case_matches("faure.author.lasso.v1", actual)
    coefs = np.array([actual[label(t)] for t in X])
    gradient = D.T @ (D @ coefs - y) / len(y)
    assert abs(gradient[0]) < 1e-10
    active = abs(coefs[1:]) > 1e-10
    np.testing.assert_allclose(
        gradient[1:][active], -0.05 * np.sign(coefs[1:][active]), atol=1e-8
    )
    assert np.all(abs(gradient[1:][~active]) <= 0.05 + 1e-8)


@pytest.mark.literature_case("faure.multistate.walsh.v1", role="input_check")
def test_synthetic_population_is_retained(synthetic):
    arities, X, y = synthetic
    land = landscape(X, y)
    assert land.n_configs == len(X) == np.prod(arities) == 24
    np.testing.assert_array_equal(land.get_data()[list(land.data_types)], X)
    np.testing.assert_array_equal(land.graph.vs["fitness"], y)
