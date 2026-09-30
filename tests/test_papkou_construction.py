"""Compare the complete DHFR construction with the authors' published graph."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphfla.landscape import DNALandscape


FIXTURES = Path(__file__).parent / "fixtures" / "papkou2023"


@pytest.mark.integration
def test_papkou_author_graph():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, digest in manifest["fixtures"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest

    data = pd.read_csv(FIXTURES / "fitness.csv.gz", float_precision="round_trip")
    reference = pd.read_csv(
        FIXTURES / "edges.ncol.gz", sep=" ", header=None, names=["source", "target"]
    )
    assert len(data) == 261333
    assert data.sequence.is_unique
    landscape = DNALandscape().build_from_data(
        data.sequence,
        data.fitness,
        tau=-0.507774,
        filter_mode="both",
        epsilon=0,
        verbose=False,
    )
    assert landscape.shape == (135178, 324044)
    assert landscape.graph.is_simple()
    assert landscape.graph.is_connected(mode="weak")

    sequences = [
        "".join(row) for row in zip(*(landscape.graph.vs[f"pos_{i}"] for i in range(9)))
    ]
    expected_edges = set(reference.itertuples(index=False, name=None))
    actual_edges = {
        (sequences[u], sequences[v]) for u, v in landscape.graph.get_edgelist()
    }
    assert len(expected_edges) == len(reference) == 324044
    assert set(sequences) == set(reference.to_numpy().ravel())
    assert actual_edges == expected_edges

    fitness = data.set_index("sequence").fitness.loc[sequences].to_numpy()
    np.testing.assert_array_equal(landscape.graph.vs["fitness"], fitness)
    edges = np.asarray(landscape.graph.get_edgelist())
    np.testing.assert_array_equal(
        landscape.graph.es["delta_fit"], fitness[edges[:, 1]] - fitness[edges[:, 0]]
    )
    assert np.all(fitness[edges[:, 1]] > fitness[edges[:, 0]])
    assert np.all(fitness[edges[:, 1]] >= -0.507774)
