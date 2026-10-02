"""Wagner 2023, Nat. Commun. 14:3624, doi:10.1038/s41467-023-39321-8.

Author replay and the corrected general estimator are separate evidence claims.
"""

from functools import lru_cache
import pytest

from validation.ee_mutations import reproduce_dataset
from validation.testing import assert_case_matches, input_paths


@lru_cache(maxsize=2)
def result(kind):
    input_paths(f"wagner.{kind}.ee.author.v1")
    return reproduce_dataset(kind)


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            k,
            marks=pytest.mark.literature_case(
                f"wagner.{k}.ee.author.v1", role="author_result"
            ),
        )
        for k in ["protein", "rna"]
    ],
)
def test_author_replay(kind):
    replay = result(kind)
    counts = replay["paper_replay_rounded_effects"]
    assert_case_matches(
        f"wagner.{kind}.ee.author.v1",
        {
            "ordered_pairs": replay["ordered_pairs"],
            "beneficial": counts["beneficial"],
            "deleterious": counts["deleterious"],
        },
    )
    assert replay["author_flag_mismatches"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            k,
            marks=pytest.mark.literature_case(
                f"wagner.{k}.ee.general.v1", role="independent_check"
            ),
        )
        for k in ["protein", "rna"]
    ],
)
def test_general_estimator(kind):
    replay = result(kind)
    assert_case_matches(
        f"wagner.{kind}.ee.general.v1",
        {
            "ordered_pairs": replay["ordered_pairs"],
            **replay["public_neighborhood_variation_counts"],
        },
    )
    assert replay["corrected_kernel_oracle_mismatches"] == 0
    assert replay["public_table_oracle_mismatches"] == 0
