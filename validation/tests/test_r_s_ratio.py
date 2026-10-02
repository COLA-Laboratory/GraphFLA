"""Published r/s targets and independently encoded additive models."""

import numpy as np
import pytest

from graphfla import analysis
from graphfla.analysis._roughness import _additive_fit, _predictors
from validation.r_s_ratio import boolean_input, kuo_input
from validation.testing import assert_case_matches


BOOLEAN = {"chou": "szendro.chou.rs.v1", "csi": "ferretti.csi.rs.v1"}


@pytest.fixture(scope="module")
def boolean_datasets(verified_literature_inputs):
    return {
        name: boolean_input(verified_literature_inputs[case][0], name)
        for name, case in BOOLEAN.items()
        if case in verified_literature_inputs
    }


@pytest.fixture(scope="module")
def kuo_dataset(verified_literature_inputs):
    case = next(c for c in verified_literature_inputs if c.startswith("song.kuo.rs."))
    return kuo_input(verified_literature_inputs[case][0])


def boolean_cases(role):
    return [
        pytest.param(name, marks=pytest.mark.literature_case(case, role=role))
        for name, case in BOOLEAN.items()
    ]


@pytest.mark.parametrize("name", boolean_cases("paper_result"))
def test_boolean_paper_result(name, boolean_datasets):
    _, _, landscape, ref = boolean_datasets[name]
    for value in (analysis.r_s_ratio(landscape), ref["ratio"]):
        assert_case_matches(BOOLEAN[name], {"ratio": value})


@pytest.mark.parametrize("name", boolean_cases("independent_check"))
def test_boolean_coefficients_and_residuals(name, boolean_datasets):
    X, y, landscape, ref = boolean_datasets[name]
    # Complete Boolean designs also admit a closed-form, orthogonal-basis
    # calculation independent of both least-squares implementations.
    slopes = 2 * np.mean(y[:, None] * (2 * X - 1), axis=0)
    residuals = y - y.mean() - np.sum((X - 0.5) * slopes, axis=1)
    assert ref["coefficients"][1:] == pytest.approx(slopes, abs=1e-13)
    assert ref["residuals"] == pytest.approx(residuals, abs=1e-13)
    coef, rank, r = _additive_fit(_predictors(landscape, landscape.get_data()), y)
    assert rank == X.shape[1] + 1
    assert coef == pytest.approx(ref["coefficients"], abs=1e-13)
    assert r == pytest.approx(ref["roughness"], abs=1e-13)
    assert analysis.r_s_ratio(landscape) == pytest.approx(ref["ratio"], abs=1e-12)


@pytest.mark.parametrize("name", boolean_cases("input_check"))
def test_boolean_population(name, boolean_datasets):
    X, y, landscape, _ = boolean_datasets[name]
    assert len(X) == len(np.unique(X, axis=0)) == 2 ** X.shape[1]
    assert landscape.n_configs == len(X) and np.isfinite(y).all()
    assert np.isin(X, [0, 1]).all()


@pytest.mark.literature_case("song.kuo.rs.ols.v1", role="independent_check")
def test_kuo_reference_conventions(kuo_dataset):
    _, results = kuo_dataset
    for key in ("public", "independent"):
        values = {
            base: row[key]["ratio"] if key == "independent" else row[key]
            for base, row in results.items()
        }
        assert_case_matches("song.kuo.rs.ols.v1", values)
    a, u = (results[base]["independent"] for base in ("A", "U"))
    assert a["rank"] == u["rank"] == 28
    # Same fitted function, different reference contrasts and denominator.
    # Use the case's 1e-10 numerical budget: 197,890-row reductions and
    # two differently conditioned dense designs accumulate rounding error.
    np.testing.assert_allclose(a["residuals"], u["residuals"], rtol=0, atol=1e-10)
    assert a["roughness"] == pytest.approx(u["roughness"], abs=1e-13)
    assert a["slope"] != pytest.approx(u["slope"])


@pytest.mark.literature_case("song.kuo.rs.author.v1", role="author_result")
def test_kuo_author_ridge_result(kuo_dataset):
    _, results = kuo_dataset
    assert_case_matches(
        "song.kuo.rs.author.v1", {"ratio": results["U"]["author_ridge"]}
    )
    # An OLS value rounding to the same number does not validate Ridge.
    assert abs(results["U"]["author_ridge"] - results["U"]["public"]) > 1e-5


@pytest.mark.literature_case("song.kuo.rs.ols.v1", role="input_check")
def test_kuo_population(kuo_dataset):
    frame, _ = kuo_dataset
    assert len(frame) == frame.sequences.nunique() == 197890
    assert frame.sequences.str.len().eq(9).all()
    assert np.isfinite(frame.fitness).all()
    states = np.asarray([list(s) for s in frame.sequences])
    assert np.array_equal(states, frame[[f"pos{i}" for i in range(1, 10)]].to_numpy())
    assert all(set(states[:, j]) == set("ACGU") for j in range(9))
