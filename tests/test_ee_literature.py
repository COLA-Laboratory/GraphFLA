"""Published EE outputs and the corrected estimator are separate evidence."""

import pytest

from validation.ee_mutations import reproduce_dataset


@pytest.mark.parametrize("kind", ["protein", "rna"])
def test_wagner_author_replay_and_corrected_kernel(kind):
    result = reproduce_dataset(kind)
    assert result["author_flag_mismatches"] == 0
    assert result["corrected_kernel_oracle_mismatches"] == 0
