"""Lyons et al. (2020), Nature Ecology & Evolution 4:1685–1693.

DOI: 10.1038/s41559-020-01286-y. Results p. 1686; Methods p. 1692.
Full-population resource review: validation/IDIOSYNCRASY_TEST_AUDIT.md.
"""

import numpy as np
import pytest

from validation.idiosyncrasy import reproduce
from validation.testing import assert_case_matches

PAPER = "lyons.trna.iid.v1"
POPULATION = "lyons.trna.population.v1"
SEEDED = "lyons.trna.iid.seed0.v1"
FIGURE = "lyons.trna.fig1a.control.v1"


@pytest.fixture(scope="module")
def trna(verified_literature_inputs):
    # All four claims use this same, already hash-verified complete population.
    case_id = next(c for c in verified_literature_inputs if c.startswith("lyons.trna."))
    (path,) = verified_literature_inputs[case_id]
    return reproduce(path, details=True)


@pytest.mark.literature_case(PAPER, role="paper_result")
def test_lyons_published_landscape_mean(trna):
    report = trna["report"]
    for key in ("independent_author_procedure", "production_kernel_with_author_seeds"):
        observed = {name: report[key][name] for name in ("mean", "sem")}
        assert_case_matches(PAPER, observed)


@pytest.mark.literature_case(PAPER, role="independent_check")
def test_all_mutations_match_independent_enumeration(trna):
    assert trna["production_keys"] == trna["effect_keys"]
    np.testing.assert_array_equal(trna["production_counts"], trna["reference_counts"])
    np.testing.assert_allclose(
        trna["production_sds"], trna["reference_sds"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        trna["production_ratios"], trna["reference_ratios"], rtol=1e-12, atol=1e-12
    )


@pytest.mark.literature_case(POPULATION, role="input_check")
def test_complete_reference_population(trna):
    frame = trna["frame"]
    assert frame.sequence.is_unique and frame.sequence.str.len().eq(72).all()
    assert np.isfinite(frame.fitness).all() and (frame.fitness > 0).all()
    assert_case_matches(POPULATION, trna["report"]["population"])
    assert len(trna["loaded_fitness"]) == len(frame)
    np.testing.assert_allclose(
        trna["loaded_fitness"], frame.fitness, rtol=1e-14, atol=1e-15
    )
    assert trna["reference_counts"].min() == 3


@pytest.mark.literature_case(SEEDED, role="independent_check")
def test_public_global_matches_independent_stream(trna):
    actual = trna["report"]["public_global_seed0"]
    for key in ("independent_oracle", "serial", "parallel"):
        assert_case_matches(SEEDED, {"mean": actual[key]})
    assert actual["parallel"] == actual["serial"]


@pytest.mark.literature_case(FIGURE, role="independent_check")
def test_fig1a_released_procedure(trna):
    # This reproduces the released procedure, not the conflicting printed value.
    actual = trna["report"]["independent_author_procedure"]
    assert_case_matches(
        FIGURE,
        {
            "backgrounds": actual["example_n"],
            "effect_std": actual["example_effect_sd"],
            "control_std": actual["example_control_sd"],
            "index": actual["example_index"],
        },
    )
