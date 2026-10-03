"""Independent order profiles derived from the Faure multistate formalism."""

import json

import numpy as np
import pandas as pd
import pytest

from graphfla.analysis import walsh_hadamard
from graphfla.landscape import DNALandscape, Landscape
from validation.testing import assert_case_matches


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(
            case, marks=pytest.mark.literature_case(case, role="independent_check")
        )
        for case in ("faure.table1.order_summary.v1", "faure.lasso.order_summary.v1")
    ],
)
def test_independent_order_summary(case_id, literature_inputs):
    path = literature_inputs[case_id][0]
    if path.suffix == ".csv":
        data = pd.read_csv(path)
        landscape = DNALandscape().build_from_data(
            data.sequence, data.fitness, epsilon=0, verbose=False
        )
        options = dict(max_order=2)
    else:
        data = json.loads(path.read_text())
        X = pd.DataFrame(data["configurations"], columns=["a", "b", "c"])
        landscape = Landscape().build_from_data(
            X,
            data["fitness"],
            data_types=dict.fromkeys(X, "categorical"),
            epsilon=0,
            verbose=False,
        )
        options = dict(
            max_order=3, method="lasso", alpha=0.05, max_iter=100000, tol=1e-12
        )
    result = walsh_hadamard(landscape, **options)
    table = result.order_summary
    fields = ["r2", "delta_r2", "model_variance_fraction"]
    assert_case_matches(case_id, {key: table[key].tolist() for key in fields})
    if options.get("method") == "lasso":
        # The spectrum's denominator is fitted-model variance, whereas R2 gains
        # are measured against observed fitness variance and include shrinkage.
        assert table.r2.iloc[-1] < 1
        assert table.model_variance_fraction.sum() == pytest.approx(1)
        assert not np.allclose(table.delta_r2, table.model_variance_fraction)
    else:
        np.testing.assert_allclose(
            table.delta_r2, table.model_variance_fraction, atol=1e-12
        )
