"""Synthetic, self-contained landscape generators for the asv benchmarks.

No external data files: every benchmark builds a deterministic synthetic
landscape, so runs are portable and reproducible across machines and commits.
Sizes are kept modest (a few thousand to ~16k configurations) so the suite
stays fast enough for ``asv`` to repeat each benchmark for stable timings.
"""

import itertools

import numpy as np
import pandas as pd

__all__ = ["nk_boolean", "random_ordinal", "nk_dna"]


def nk_boolean(n=12, k=2, seed=0):
    """Kauffman NK model on the full boolean cube (``2**n`` genotypes).

    Locus ``i`` interacts with the ``k`` adjacent loci (cyclic); each locus draws
    a U(0, 1) contribution per local neighbourhood and fitness is the mean over
    loci -- tunable, non-degenerate ruggedness so the epistasis/motif metrics see
    realistic input. Returns ``(X, f)`` with ``X`` a Series of 0/1 bitstrings.
    """
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
    """Full ordinal grid (``levels**n_vars`` genotypes) with random fitness.

    Returns ``(X, f)`` with ``X`` a DataFrame of ordinal codes (columns x0..xk).
    """
    rng = np.random.default_rng(seed)
    combos = list(itertools.product(range(levels), repeat=n_vars))
    X = pd.DataFrame(combos, columns=[f"x{i}" for i in range(n_vars)])
    return X, pd.Series(rng.standard_normal(len(combos)))


def nk_dna(length=6, k=1, seed=2):
    """NK model on the full DNA cube (``4**length`` sequences over ACGT).

    Returns ``(X, f)`` with ``X`` a Series of DNA sequence strings.
    """
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
