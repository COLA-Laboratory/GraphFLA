"""Landscape datasets for the asv benchmarks.

Two sources:

* **Real** empirical landscapes committed under ``data/`` (GB1, Papkou DHFR,
  CR9114/CR6261 antibodies, TrpB3I, Westmann, WReOs) — the curated set the
  benchmarks are primarily built around. They ship in the repo, so asv can use
  them reproducibly; a benchmark raises :class:`NotImplementedError` (which asv
  treats as *skip*) if the data file is not found on a given checkout.
* **Synthetic** Kauffman-NK / ordinal-grid generators — deterministic and fully
  self-contained, used for quick portable runs and as a fallback.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from graphfla.landscape import (
    BooleanLandscape,
    OrdinalLandscape,
    DNALandscape,
    ProteinLandscape,
)

__all__ = ["REAL", "load_real", "nk_boolean", "random_ordinal", "nk_dna"]

_CLS = {
    "protein": ProteinLandscape,
    "dna": DNALandscape,
    "boolean": BooleanLandscape,
    "ordinal": OrdinalLandscape,
}

# name -> (kind, path-relative-to-data/, X column(s), fitness column).
# Ordered small -> large so a partial run hits the cheap ones first.
REAL = {
    "WReOs": ("ordinal", "Materials/WReOs/simplex.csv", ("W", "Re"), "TOPSIS_Ci"),
    "CR6261": ("boolean", "BioSequence/Phillips2021_CR6261_h1.csv", "sequences", "fitness"),
    "TrpB3I": ("protein", "BioSequence/Johnston2024_TrpB3I.csv", "sequences", "fitness"),
    "Westmann": ("dna", "BioSequence/Westmann2024.csv", "sequences", "fitness"),
    "CR9114": ("boolean", "BioSequence/Phillips2021_CR9114_h1.csv", "sequences", "fitness"),
    "GB1": ("protein", "BioSequence/Wu2016_GB1.csv", "sequences", "fitness"),
    "Papkou": ("dna", "BioSequence/Papkou2023_DHFR_RAW.csv", "seq", "fitness"),
}


def _data_root():
    """Locate the repo's ``data/`` directory (``benchmarks/`` sits at repo root).

    Walks up from this file, then tries the cwd; returns ``None`` if not found.
    """
    here = Path(__file__).resolve()
    for base in (here.parent.parent, *here.parents, Path.cwd()):
        if (base / "data" / "BioSequence").is_dir():
            return base / "data"
    return None


def load_real(name):
    """Return ``(landscape_cls, X, f)`` for a curated real landscape.

    Raises ``NotImplementedError`` (asv -> skip) when the data file is absent, so
    the suite still runs on a checkout without ``data/``.
    """
    kind, rel, xcol, fcol = REAL[name]
    root = _data_root()
    path = root / rel if root is not None else None
    if path is None or not path.exists():
        raise NotImplementedError(f"real dataset {name!r}: {rel} not found")
    if isinstance(xcol, tuple):  # ordinal: several numeric columns -> DataFrame
        cols = [*xcol, fcol]
        df = pd.read_csv(path, usecols=cols).dropna(subset=cols).reset_index(drop=True)
        return _CLS[kind], df[list(xcol)], df[fcol]
    # sequence / boolean: a single string column (keep as str, never parse to int)
    df = pd.read_csv(path, dtype={xcol: str}).dropna(subset=[xcol, fcol]).reset_index(drop=True)
    return _CLS[kind], df[xcol], df[fcol]


# --------------------------------------------------------------------------
# Synthetic generators (deterministic, self-contained — no data files)
# --------------------------------------------------------------------------

def nk_boolean(n=12, k=2, seed=0):
    """Kauffman NK model on the full boolean cube (``2**n`` 0/1-string genotypes)."""
    rng = np.random.default_rng(seed)
    tables = rng.random((n, 1 << (k + 1)))
    g = np.arange(1 << n, dtype=np.int64)
    bits = ((g[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(np.int64)
    f = np.zeros(len(g))
    for i in range(n):
        key = np.zeros(len(g), dtype=np.int64)
        for b in range(k + 1):
            key = (key << 1) | bits[:, (i + b) % n]
        f += tables[i, key]
    f /= n
    X = pd.Series([format(int(v), f"0{n}b") for v in g])
    return X, pd.Series(f)


def random_ordinal(levels=6, n_vars=3, seed=1):
    """Full ordinal grid (``levels**n_vars`` genotypes) with random fitness."""
    rng = np.random.default_rng(seed)
    combos = list(itertools.product(range(levels), repeat=n_vars))
    X = pd.DataFrame(combos, columns=[f"x{i}" for i in range(n_vars)])
    return X, pd.Series(rng.standard_normal(len(combos)))


def nk_dna(length=6, k=1, seed=2):
    """NK model on the full DNA cube (``4**length`` ACGT sequences)."""
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    n_states = 4
    tables = rng.random((length, n_states ** (k + 1)))
    codes = np.array(
        list(itertools.product(range(n_states), repeat=length)), dtype=np.int64
    )
    f = np.zeros(len(codes))
    for i in range(length):
        key = np.zeros(len(codes), dtype=np.int64)
        for b in range(k + 1):
            key = key * n_states + codes[:, (i + b) % length]
        f += tables[i, key]
    f /= length
    seqs = ["".join(alphabet[c]) for c in codes]
    return pd.Series(seqs), pd.Series(f)
