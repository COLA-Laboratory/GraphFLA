"""Ferretti et al. (2016), DOI 10.1016/j.jtbi.2016.01.037.

Eqs. (1), (2), (11); Figure 4c. Three full 32-row inputs, under 3 KiB combined.
The input and claim distinctions are described in validation/GAMMA_REVIEW.md.
"""

import math
from itertools import combinations

import numpy as np
import pytest

from graphfla import analysis
from graphfla.analysis.epistasis.gamma import _gamma_position_pair_worker
from validation.gamma import CASES, reproduce
from validation.testing import assert_case_matches


@pytest.fixture(scope="module")
def datasets(verified_literature_inputs):
    selected = {
        case_id: paths[0]
        for case_id, paths in verified_literature_inputs.items()
        if case_id.startswith("ferretti.")
    }
    # The paper and equation cases share csI input. Verify every case through
    # the suite contract, but compute each distinct input only once.
    cache = {path: reproduce(path) for path in dict.fromkeys(selected.values())}
    return {case_id: cache[path] for case_id, path in selected.items()}


@pytest.mark.literature_case("ferretti.csi.gamma.figure4.v1", role="paper_result")
def test_csi_published_gamma(datasets):
    *_, reference, actual = datasets["ferretti.csi.gamma.figure4.v1"]
    for value in (reference["gamma"]["value"], actual["gamma"]):
        assert_case_matches("ferretti.csi.gamma.figure4.v1", {"gamma": value})


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(
            case_id,
            marks=pytest.mark.literature_case(case_id, role="independent_check"),
        )
        for case_id in CASES.values()
    ],
)
def test_complete_landscape_equations(case_id, datasets):
    variants, fitness, landscape, reference, actual = datasets[case_id]
    assert len(variants) == landscape.n_configs == 32
    assert len(set(variants)) == 32
    assert_case_matches(case_id, actual)
    for metric in ("gamma", "gamma_star"):
        assert actual[metric] == pytest.approx(reference[metric]["value"], abs=1e-13)
        assert reference[metric]["directed_quadruples"] == 640
    # A second, level-correlation calculation: Eq. (2) agrees here because the
    # population is a complete regular hypercube. Do not extend to sparse data.
    X = np.array(variants)
    centered = fitness - np.mean(fitness)
    correlations = {}
    for distance in (1, 2):
        pairs = [
            (i, j)
            for i, j in combinations(range(32), 2)
            if np.count_nonzero(X[i] != X[j]) == distance
        ]
        correlations[distance] = np.mean(
            [centered[i] * centered[j] for i, j in pairs]
        ) / np.var(fitness)
    assert actual["gamma"] == pytest.approx(
        (correlations[1] - correlations[2]) / (1 - correlations[1]), abs=1e-13
    )
    # Compare all 20 ordered position-pair contributions, so aggregate
    # cancellation cannot hide a mapping, orientation or denominator error.
    alleles = [np.unique(X[:, j]) for j in range(5)]
    for p1 in range(5):
        for p2 in range(5):
            if p1 == p2:
                continue
            n, d, sn, sd, e = _gamma_position_pair_worker(
                X,
                fitness,
                p1,
                p2,
                alleles[p1],
                alleles[p2],
                np.delete(np.arange(5), [p1, p2]),
            )
            assert (
                4 * math.ldexp(n, 2 * e),
                4 * math.ldexp(d, 2 * e),
            ) == pytest.approx(
                reference["gamma"]["by_position_pair"][p1, p2], abs=1e-12
            )
            assert (4 * sn, 4 * sd) == reference["gamma_star"]["by_position_pair"][
                p1, p2
            ]


@pytest.mark.literature_case("ferretti.csi.equations.v1", role="independent_check")
def test_csi_sign_denominator_includes_neutral_effects(datasets):
    _, _, _, reference, actual = datasets["ferretti.csi.equations.v1"]
    assert reference["gamma_star"]["numerator"] == 168
    assert reference["gamma_star"]["denominator"] == 624
    assert actual["gamma_star"] == pytest.approx(7 / 26)
    # This is intentionally not certified as the printed 0.25 reproduction.


@pytest.mark.literature_case("ferretti.tem.equations.v1", role="independent_check")
def test_tem_sign_denominator_includes_neutral_effects(datasets):
    _, _, landscape, reference, actual = datasets["ferretti.tem.equations.v1"]
    assert reference["gamma_star"]["numerator"] == 324
    assert reference["gamma_star"]["denominator"] == 528
    assert actual["gamma_star"] == pytest.approx(27 / 44)
    assert analysis.gamma_star(landscape, n_jobs=2) == actual["gamma_star"]


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(
            case_id, marks=pytest.mark.literature_case(case_id, role="input_check")
        )
        for case_id in CASES.values()
    ],
)
def test_complete_pinned_population(case_id, datasets):
    variants, fitness, landscape, _, _ = datasets[case_id]
    X = np.asarray(variants)
    assert X.shape == (32, 5) and len(set(variants)) == 32
    assert np.isin(X, [0, 1]).all() and np.isfinite(fitness).all()
    assert landscape.n_configs == 32 and landscape.n_vars == 5
    assert (
        np.count_nonzero(np.triu((X[:, None, :] != X[None, :, :]).sum(axis=2) == 1))
        == 80
    )
