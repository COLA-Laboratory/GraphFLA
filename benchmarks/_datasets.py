"""Pinned empirical inputs and deterministic construction workloads."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graphfla.landscape import (
    BooleanLandscape,
    DNALandscape,
    Landscape,
    OrdinalLandscape,
    ProteinLandscape,
    RNALandscape,
    SequenceLandscape,
)

ROOT = Path(__file__).resolve().parents[1]
DMS = json.loads((ROOT / "benchmarks/data/manifest.json").read_text())["datasets"]
CLASSES = dict(
    boolean=BooleanLandscape,
    dna=DNALandscape,
    rna=RNALandscape,
    protein=ProteinLandscape,
    ordinal=OrdinalLandscape,
)
REAL = {
    "WReOs": ("ordinal", "Materials/WReOs/simplex.csv", ("W", "Re"), "TOPSIS_Ci"),
    "CR6261": (
        "boolean",
        "BioSequence/Phillips2021_CR6261_h1.csv",
        "sequences",
        "fitness",
    ),
    "TrpB3I": (
        "protein",
        "BioSequence/Johnston2024_TrpB3I.csv",
        "sequences",
        "fitness",
    ),
    "Westmann": ("dna", "BioSequence/Westmann2024.csv", "sequences", "fitness"),
    "CR9114": (
        "boolean",
        "BioSequence/Phillips2021_CR9114_h1.csv",
        "sequences",
        "fitness",
    ),
    "GB1": ("protein", "BioSequence/Wu2016_GB1.csv", "sequences", "fitness"),
}
SYNTHETIC = [
    "synthetic-boolean",
    "synthetic-ordinal",
    "synthetic-dna",
    "synthetic-rna",
    "synthetic-sequence",
    "synthetic-hpo",
]
DATASETS = list(REAL) + ["Papkou", "Papkou-filtered"] + list(DMS) + SYNTHETIC


def load_real(name):
    kind, relative, xcol, fcol = REAL[name]
    columns = list(xcol) if isinstance(xcol, tuple) else [xcol]
    dtypes = None if isinstance(xcol, tuple) else {xcol: str}
    data = pd.read_csv(ROOT / "data" / relative, dtype=dtypes)
    data = data.dropna(subset=[*columns, fcol]).reset_index(drop=True)
    X = data[columns] if isinstance(xcol, tuple) else data[xcol]
    return CLASSES[kind], X, data[fcol]


def nk_boolean(n=12, k=2, seed=0):
    rng = np.random.default_rng(seed)
    tables = rng.random((n, 1 << (k + 1)))
    numbers = np.arange(1 << n)
    bits = (numbers[:, None] >> np.arange(n - 1, -1, -1)) & 1
    fitness = np.zeros(len(bits))
    for i in range(n):
        key = np.zeros(len(bits), dtype=int)
        for b in range(k + 1):
            key = (key << 1) | bits[:, (i + b) % n]
        fitness += tables[i, key]
    return pd.Series([format(int(v), f"0{n}b") for v in numbers]), pd.Series(
        fitness / n
    )


def random_ordinal(levels=6, n_vars=3, seed=1):
    rows = list(itertools.product(range(levels), repeat=n_vars))
    X = pd.DataFrame(rows, columns=[f"x{i}" for i in range(n_vars)])
    return X, pd.Series(np.random.default_rng(seed).standard_normal(len(X)))


def sequence_cube(alphabet, length, seed=2):
    sequences = ["".join(row) for row in itertools.product(alphabet, repeat=length)]
    return pd.Series(sequences), pd.Series(
        np.random.default_rng(seed).normal(size=len(sequences))
    )


def load_dataset(name):
    """Return a class, full input, fitness and construction keyword arguments."""
    if name in REAL:
        return (*load_real(name), {})
    if name in DMS:
        spec = DMS[name]
        data = pd.read_csv(ROOT / "benchmarks/data" / spec["file"])
        sequences = data[spec["sequence_column"]]
        if spec["kind"] == "rna":
            sequences = sequences.str.upper().str.replace("T", "U", regex=False)
        return CLASSES[spec["kind"]], sequences, data[spec["fitness_column"]], {}
    if name in {"Papkou", "Papkou-filtered"}:
        data = pd.read_csv(
            ROOT / "tests/fixtures/papkou2023/fitness.csv.gz",
            float_precision="round_trip",
        )
        options = (
            dict(tau=-0.507774, filter_mode="both") if name.endswith("filtered") else {}
        )
        return DNALandscape, data.sequence, data.fitness, options
    if name == "synthetic-boolean":
        return BooleanLandscape, *nk_boolean(), {}
    if name == "synthetic-ordinal":
        return OrdinalLandscape, *random_ordinal(), {}
    if name == "synthetic-hpo":
        X = pd.DataFrame(
            itertools.product(
                [False, True], ["relu", "tanh", "gelu"], range(8), range(8)
            ),
            columns=["bias", "activation", "depth", "width"],
        )
        fitness = pd.Series(np.random.default_rng(3).normal(size=len(X)))
        types = dict(
            bias="boolean", activation="categorical", depth="ordinal", width="ordinal"
        )
        return Landscape, X, fitness, {"data_types": types}
    sequences = {
        "synthetic-dna": (DNALandscape, "ACGT", 6),
        "synthetic-rna": (RNALandscape, "ACGU", 4),
        "synthetic-sequence": (SequenceLandscape, "ABC", 5),
    }
    if name in sequences:
        cls, alphabet, length = sequences[name]
        return cls, *sequence_cube(alphabet, length), {}
    raise ValueError(f"Unknown benchmark dataset: {name}")


def build_dataset(name):
    cls, X, f, kwargs = load_dataset(name)
    return cls().build_from_data(X, f, verbose=False, **kwargs)
