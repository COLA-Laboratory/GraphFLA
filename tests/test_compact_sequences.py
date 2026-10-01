"""Sequence storage must preserve the general input pipeline's public results."""

import itertools

import numpy as np
import pandas as pd
import pytest

from graphfla._data import SequenceHandler
from graphfla.landscape import DNALandscape, ProteinLandscape, RNALandscape
from graphfla.landscape import _build


@pytest.mark.parametrize(
    "cls,alphabet",
    [(DNALandscape, "AT"), (RNALandscape, "AU"), (ProteinLandscape, "AW")],
)
@pytest.mark.parametrize("container", [list, tuple, np.array, pd.Series])
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("filter_mode", ["any", "both"])
def test_compact_and_frame_paths_agree(
    monkeypatch, cls, alphabet, container, maximize, filter_mode
):
    a, b = alphabet
    sequences = [
        a * 70 + x + a * 3 + y + a * 25 + z
        for x, y, z in itertools.product(alphabet, repeat=3)
    ]
    sequences = [sequences[3].lower(), *sequences, b * len(sequences[0])]
    fitness = np.array([0.25, 0, 0.5, 0.5, 99, 1.5, 2, 3, 4, 10])
    X = container(sequences)
    if isinstance(X, pd.Series):
        X.index = np.arange(len(X)) + 100
    options = dict(tau=1, filter_mode=filter_mode, epsilon=0.1, verbose=False)
    compact = cls(maximize=maximize).build_from_data(X, fitness, **options)
    monkeypatch.setattr(_build, "prepare_sequences", lambda *args: None)
    frame = cls(maximize=maximize).build_from_data(X, fitness, **options)

    assert compact.graph.get_edgelist() == frame.graph.get_edgelist()
    assert compact.graph.es.attributes() == frame.graph.es.attributes()
    for attribute in frame.graph.es.attributes():
        assert compact.graph.es[attribute] == frame.graph.es[attribute]
    assert compact.graph.vs.attributes() == frame.graph.vs.attributes()
    pd.testing.assert_frame_equal(compact.get_data(), frame.get_data())
    pd.testing.assert_series_equal(compact.configs, frame.configs)
    assert compact.data_types == frame.data_types
    assert compact.config_dict == frame.config_dict
    assert compact.plateaus == frame.plateaus
    assert compact.lo_index == frame.lo_index


def test_custom_sequence_handler_is_not_bypassed():
    class ShiftFitness(SequenceHandler):
        def prepare(self, X, f, verbose=True):
            data, fitness, types, n_vars = super().prepare(X, f, verbose=verbose)
            return data, fitness + 10, types, n_vars

    landscape = DNALandscape()
    landscape.register_input_handler("sequence", ShiftFitness(list("ACGT")))
    landscape.build_from_data(["AAA", "AAT"], [0, 1], verbose=False)
    assert landscape.graph.vs["fitness"] == [10.0, 11.0]
