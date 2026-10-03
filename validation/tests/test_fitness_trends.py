"""Johnson (2019) counts and Papkou (2023) independent edge-fit validation.

See FITNESS_TRENDS_REVIEW.md: a per-mutation numerical-kernel check does not
validate the public pooled estimand, and Papkou does not print fit coefficients.
"""

import numpy as np
import pytest
from scipy.stats import linregress, t

from graphfla import analysis as A
from graphfla.analysis.epistasis._fitness_trends import _BivariateMoments
from validation.fitness_trends import independent_trends, johnson_input, papkou_input
from validation.testing import assert_case_matches

JOHNSON = "johnson2019.fitness_trends.counts.v1"
PAPKOU = {
    "raw": "papkou.fitness_trends.raw.v1",
    "clipped": "papkou.fitness_trends.clipped.v1",
}


@pytest.fixture(scope="module")
def johnson(verified_literature_inputs):
    return johnson_input(verified_literature_inputs[JOHNSON][0])


@pytest.mark.literature_case(JOHNSON, role="paper_result")
def test_johnson_published_per_mutation_counts(johnson):
    counts = dict(
        mutations=len(johnson), significant=0, negative=0, positive=0, increasing_cost=0
    )
    for _, group in johnson:
        moments = _BivariateMoments()
        moments.update(group.Fitness.to_numpy(), group.s.to_numpy())
        slope = moments.statistic("regression")
        r = moments.statistic("pearson")
        p = 2 * t.sf(abs(r) * np.sqrt((len(group) - 2) / (1 - r * r)), len(group) - 2)
        if p < 0.05:
            counts["significant"] += 1
            counts["negative"] += int(slope < 0)
            counts["positive"] += int(slope > 0)
            counts["increasing_cost"] += int(slope < 0 and group.s.mean() < 0)
    assert_case_matches(JOHNSON, counts)


@pytest.mark.literature_case(JOHNSON, role="independent_check")
def test_johnson_every_fit_and_population(johnson):
    assert len({name for name, _ in johnson}) == 80
    for _, group in johnson:
        assert not group.Sample.duplicated().any()
        assert len(group) >= 50 and np.isfinite(group[["Fitness", "s"]]).all().all()
        m = _BivariateMoments()
        # Deliberately divide the observations differently from production
        # graph blocks to exercise stable moment merging on measured data.
        for chunk in np.array_split(group[["Fitness", "s"]].to_numpy(), 3):
            m.update(chunk[:, 0], chunk[:, 1])
        independent = linregress(group.Fitness, group.s)
        assert m.statistic("regression") == pytest.approx(independent.slope, abs=2e-13)
        assert m.statistic("pearson") == pytest.approx(independent.rvalue, abs=2e-13)


@pytest.fixture(scope="module")
def papkou(verified_literature_inputs):
    return {
        label: papkou_input(*verified_literature_inputs[case], clip=label == "clipped")
        for label, case in PAPKOU.items()
        if case in verified_literature_inputs
    }


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(
            label, marks=pytest.mark.literature_case(case, role="independent_check")
        )
        for label, case in PAPKOU.items()
    ],
)
def test_papkou_edge_statistics(label, papkou):
    ls, source, target = papkou[label]
    observed = {
        f"{kind}_{method}": fn(ls, method)
        for kind, fn in [
            ("returns", A.diminishing_returns_index),
            ("costs", A.increasing_costs_index),
        ]
        for method in ["pearson", "spearman", "regression"]
    }
    assert_case_matches(PAPKOU[label], observed)
    assert_case_matches(PAPKOU[label], independent_trends(source, target))
    # Same raw edge population: reverse-cost x equals gain-background x + gain.
    assert np.all(target > source)
    x, y = source - source.mean(), target - source
    y = y - y.mean()
    assert np.dot(target - target.mean(), y) == pytest.approx(
        np.dot(x, y) + np.dot(y, y), rel=2e-13
    )


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(label, marks=pytest.mark.literature_case(case, role="input_check"))
        for label, case in PAPKOU.items()
    ],
)
def test_papkou_author_edge_population(label, papkou):
    ls, source, target = papkou[label]
    assert ls.graph.vcount() == len(set(ls.graph.vs["name"])) == 135178
    assert ls.graph.ecount() == len(source) == 324044
    assert ls.graph.is_simple() and ls.graph.is_connected(mode="weak")
    assert np.all(target > source)
    if label == "clipped":
        assert source.min() == -0.507774
    else:
        assert source.min() < -0.507774
