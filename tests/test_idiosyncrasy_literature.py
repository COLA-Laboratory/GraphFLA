"""Published numeric anchors and independent same-stream empirical comparison."""

import pytest

from validation.contract import load_definitions, matches
from validation.idiosyncrasy import FIXTURE, reproduce


@pytest.fixture(scope="module")
def trna_result():
    return reproduce()


def test_lyons_published_landscape_mean(trna_result):
    actual = trna_result["independent_author_procedure"]
    assert actual["example_n"] == 88
    assert actual["example_effect_sd"] == pytest.approx(0.13, abs=0.005)
    assert actual["example_index_under_global_notebook_policy"] == pytest.approx(
        0.49, abs=0.005
    )
    case = load_definitions(FIXTURE.parents[3] / "validation")[2]["lyons.trna.iid.v1"]
    observed = {key: actual[key] for key in ("mean", "sem")}
    assert matches(case["expected"], observed, case["comparison"])
    assert (
        trna_result["production_kernel_with_author_seeds"]["max_abs_per_mutation_error"]
        < 1e-12
    )


def test_lyons_example_author_code_disagrees_with_printed_control(trna_result):
    # Do not adjust the seed or widen the paper tolerance to hide this mismatch.
    # The released Fig. 1a cell specifies 4033, yielding 0.508..., while the text
    # reports 0.49. These are author-code reproduction targets, not paper values.
    actual = trna_result["independent_author_procedure"]
    assert actual["example_control_sd"] == pytest.approx(0.24903011647303494, abs=1e-12)
    assert actual["example_index"] == pytest.approx(0.5080668755924872, abs=1e-12)
    assert abs(actual["example_index"] - 0.49) > 0.005


def test_lyons_complete_population_public_global(trna_result):
    assert trna_result["population"] == {
        "viable_genotypes": 28530,
        "isolates": 3903,
        "directed_mutations": 828,
    }
    actual = trna_result["public_global_seed0"]
    assert actual["serial"] == pytest.approx(actual["independent_oracle"], abs=1e-12)
    assert actual["parallel"] == actual["serial"]
    # This analytic limit is explicitly a different estimator, not an alternate
    # acceptance target for the paper's finite-control statistic.
    assert trna_result["old_analytic_full_population"] == pytest.approx(
        0.5942481263, abs=1e-10
    )
