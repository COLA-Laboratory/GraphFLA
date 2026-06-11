"""Build / input-format, deprecation-alias, and plateau-regression tests.

The original per-feature smoke tests (assert isinstance(result, float) /
assert "key" in result) were removed: they checked only return type, never
correctness, and are superseded by the value-pinned suites in test_metrics.py,
test_construction.py, and test_golden_landscapes.py. What remains exercises
input-format handling (list / DataFrame int+str cols / generic type=), the
deprecation aliases (FutureWarning), and the plateau-aware local-optima logic.
"""

import pytest


import pandas as pd


import random


from itertools import product


import numpy as np


from graphfla.analysis import *


from graphfla.landscape import (
    BooleanLandscape,
    DNALandscape,
    RNALandscape,
    ProteinLandscape,
    Landscape,
)


from graphfla.problems import NK


def generate_sequences(n, alphabets):
    sequences = ["".join(p) for p in product(alphabets, repeat=n)]
    return sequences


def generate_random_fitness(num_sequences):
    return [random.uniform(0, 100) for _ in range(num_sequences)]


@pytest.fixture(scope="module")
def boolean_landscape_data():
    n = 4
    k = 2
    problem = NK(n, k)
    X, fitness = problem.get_data()
    return X, fitness


@pytest.fixture(scope="module")
def boolean_landscape(boolean_landscape_data):
    X, fitness = boolean_landscape_data
    landscape = BooleanLandscape()
    landscape.build_from_data(
        X,
        fitness,
        verbose=False,
    )
    return landscape


@pytest.fixture(scope="module")
def dna_sequence_data():
    n_seq = 2  # Sequence length
    sequences = generate_sequences(n_seq, alphabets=["A", "C", "G", "T"])
    fitness = generate_random_fitness(len(sequences))
    return sequences, fitness


@pytest.fixture(scope="module")
def rna_sequence_data():
    n_seq = 2
    sequences = generate_sequences(n_seq, alphabets=["A", "C", "G", "U"])
    fitness = generate_random_fitness(len(sequences))
    return sequences, fitness


PROTEIN_ALPHABETS_FULL = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


PROTEIN_ALPHABETS_SUBSET = PROTEIN_ALPHABETS_FULL[:4]


@pytest.fixture(scope="module")
def protein_sequence_data():
    n_seq = 3  # Keep it small for testing
    sequences = generate_sequences(n_seq, alphabets=PROTEIN_ALPHABETS_SUBSET)
    fitness = generate_random_fitness(len(sequences))
    return sequences, fitness


def test_build_boolean_landscape(boolean_landscape_data):
    X, fitness = boolean_landscape_data
    landscape = BooleanLandscape()
    landscape.build_from_data(
        X,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


def test_build_dna_landscape_from_list(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    landscape = DNALandscape()
    landscape.build_from_data(
        sequences,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


@pytest.mark.parametrize(
    "as_series",
    [False, True],
    ids=["list", "series"],
)
def test_build_dna_landscape_filter_any_accepts_sequence_like_inputs(as_series):
    sequences = ["AA", "AC", "AG", "AT", "CA"]
    if as_series:
        sequences = pd.Series(sequences, name="seq")
    fitness = np.array([0.8, 0.85, 0.9, 0.95, 0.1])
    landscape = DNALandscape()
    landscape.build_from_data(
        sequences,
        fitness,
        tau=0.5,
        filter_mode="any",
        verbose=False,
    )

    assert landscape.graph.vcount() == 4
    assert all(value >= 0.5 for value in landscape.graph.vs["fitness"])


def test_build_dna_landscape_from_df_int_cols(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    X_dna = pd.DataFrame([list(s) for s in sequences])
    landscape = DNALandscape()
    landscape.build_from_data(
        X_dna,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


def test_build_dna_landscape_from_df_str_cols(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    X_dna = pd.DataFrame([list(s) for s in sequences])
    X_dna.columns = [f"pos_{i}" for i in range(X_dna.shape[1])]
    landscape = DNALandscape()
    landscape.build_from_data(
        X_dna,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


def test_build_generic_landscape_dna(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    landscape = Landscape(kind="dna")
    landscape.build_from_data(
        sequences,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


def test_build_rna_landscape_from_list(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    landscape = RNALandscape()
    landscape.build_from_data(
        sequences,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


def test_build_rna_landscape_from_df_int_cols(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    X_rna = pd.DataFrame([list(s) for s in sequences])
    landscape = RNALandscape()
    landscape.build_from_data(
        X_rna,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


def test_build_rna_landscape_from_df_str_cols(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    X_rna = pd.DataFrame([list(s) for s in sequences])
    X_rna.columns = [f"pos_{i}" for i in range(X_rna.shape[1])]
    landscape = RNALandscape()
    landscape.build_from_data(
        X_rna,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


def test_build_generic_landscape_rna(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    landscape = Landscape(kind="rna")
    landscape.build_from_data(sequences, fitness, verbose=False)
    assert landscape.n_configs == 16
    assert landscape.graph.ecount() > 0


def test_build_protein_landscape_from_list(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    landscape = ProteinLandscape()
    landscape.build_from_data(
        sequences,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 64
    assert landscape.graph.ecount() > 0


def test_build_protein_landscape_from_df_int_cols(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    X_protein = pd.DataFrame([list(s) for s in sequences])
    landscape = ProteinLandscape()
    landscape.build_from_data(
        X_protein,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 64
    assert landscape.graph.ecount() > 0


def test_build_protein_landscape_from_df_str_cols(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    X_protein = pd.DataFrame([list(s) for s in sequences])
    X_protein.columns = [f"pos_{i}" for i in range(X_protein.shape[1])]
    landscape = ProteinLandscape()
    landscape.build_from_data(
        X_protein,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 64
    assert landscape.graph.ecount() > 0


def test_build_generic_landscape_protein(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    landscape = Landscape(kind="protein")
    landscape.build_from_data(
        sequences,
        fitness,
        verbose=False,
    )
    assert landscape.n_configs == 64
    assert landscape.graph.ecount() > 0


def test_plateau_containing_global_max_is_still_local_optimum():
    X = pd.DataFrame(list(product([0, 1], repeat=3)))
    fitness = [10.0, 9.6, 0.0, 9.1, -1.0, -1.0, 0.0, 9.7]

    landscape = BooleanLandscape()
    landscape.build_from_data(
        X,
        fitness,
        epsilon=0.5,
        verbose=False,
    )

    assert landscape.n_lo == 2  # distinct optima: plateau {0,1,3} + singleton 7
    assert landscape.n_lo_members == 4  # member nodes: 0, 1, 3, 7
    assert landscape.lo_index == [0, 1, 3, 7]
    assert landscape._peak_index == [0, 7]
    assert landscape.plateau_lo_index == [0]
    assert landscape.plateaus[0] == [0, 1, 3]
    assert landscape.graph.vs["is_lo"] == [
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    landscape.basins  # lazily compute basins (populates basin_index)
    assert [landscape.graph.vs["basin_index"][i] for i in [0, 1, 3]] == [0, 0, 0]


def test_plateau_regression_no_zero_local_optima():
    X = pd.DataFrame(list(product([0, 1], repeat=5)))
    fitness = [
        -1.1599119397622075,
        0.6558030084807731,
        0.8412870743140487,
        0.9648669963989759,
        -1.7702534901744165,
        -1.888160042310207,
        0.985754214100929,
        -0.05748776728510379,
        1.4481631496200673,
        -0.8291688360102019,
        0.9208080312017977,
        -0.6643568253334865,
        -0.65715768237795,
        1.0941330885608245,
        0.6444863655928873,
        1.3135399812532982,
        -0.9631077235435761,
        0.3272278805444076,
        -0.08292316594241168,
        0.6175586221504188,
        -0.624660158560445,
        0.9854652119195818,
        0.7162382007229012,
        0.570823371894565,
        0.45577948038404925,
        0.23033813421912877,
        0.5264799381021805,
        -1.6830230096239733,
        -0.4691233473478074,
        -1.798799136014413,
        -0.63883339770879,
        -0.6969173560844198,
    ]

    landscape = BooleanLandscape()
    landscape.build_from_data(
        X,
        fitness,
        epsilon=1.0,
        verbose=False,
    )

    assert landscape.go_index == 8
    assert landscape.n_lo > 0
    assert landscape.graph.vs["is_lo"][8]
    assert 8 in landscape.lo_index


