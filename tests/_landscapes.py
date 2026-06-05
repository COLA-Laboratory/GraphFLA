"""Shared builders for controlled, analytically-known landscapes.

These helpers construct small landscapes whose exact structure and metric
values can be derived by hand, so tests can pin *correctness* (the computed
number is right) rather than merely *smoke* (the call returns a float). The
synthetic families mirror the gold-standard anchors used during verification:

- ``onemax``        : f = number of 1s. Purely additive, single peak at the
  all-ones corner, no ties between neighbours. Anchors: gamma=1, r/s=0,
  FDC=-1, neutrality=0, GO-accessibility=1.
- ``additive_landscape`` : f = w . x with random per-locus weights. Additive
  but with an arbitrary (not all-ones) optimum.
- ``hoc_landscape`` : House-of-Cards, i.i.d. random fitness (maximally rugged).
- ``from_map``      : build a boolean landscape from an explicit
  {genotype-tuple: fitness} dict, for hand-designed epistasis squares.
"""

import warnings
from itertools import product

import numpy as np

from graphfla.landscape import BooleanLandscape
from graphfla.problems import NK


def build_bool(X, f, maximize=True, **kw):
    """Build a BooleanLandscape from iterables of configs and fitness values."""
    ls = BooleanLandscape(maximize=maximize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls.build_from_data(list(X), list(f), verbose=False, **kw)
    return ls


def from_map(fitmap, n, maximize=True, **kw):
    """Build a boolean landscape from a {genotype-tuple: fitness} mapping."""
    X = [np.array(g) for g in fitmap]
    f = [fitmap[g] for g in fitmap]
    return build_bool(X, f, maximize=maximize, **kw)


def onemax(n, maximize=True, **kw):
    """Complete boolean n-cube with f = number of 1s (additive, single-peaked)."""
    X = [np.array(c) for c in product([0, 1], repeat=n)]
    f = [float(sum(c)) for c in X]
    return build_bool(X, f, maximize=maximize, **kw)


def additive_landscape(n, seed=0, **kw):
    """Pure additive landscape f = sum_j w_j * x_j (no epistasis). Returns (ls, w)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    X = [np.array(c) for c in product([0, 1], repeat=n)]
    f = [float(np.dot(w, c)) for c in X]
    return build_bool(X, f, **kw), w


def hoc_landscape(n, seed=0, **kw):
    """House-of-Cards: i.i.d. random fitness per genotype (maximally rugged)."""
    rng = np.random.default_rng(seed)
    X = [np.array(c) for c in product([0, 1], repeat=n)]
    f = list(rng.normal(size=len(X)))
    return build_bool(X, f, **kw)


def nk_landscape(n, k, seed=0, **kw):
    """Boolean landscape from the NK generator (K=0 is purely additive)."""
    X, f = NK(n, k, seed=seed).get_data()
    return build_bool(X, f, **kw)


# --- Hand-designed 2-locus epistasis squares (validated against Ferretti 2016
#     and Poelwijk/Weinreich motif classes). Each maps a genotype tuple to a
#     fitness; the named epistasis type is the *only* type present. ---
MAGNITUDE_SQUARE = {(0, 0): 0.0, (1, 0): 1.0, (0, 1): 1.0, (1, 1): 3.0}
SIGN_SQUARE = {(0, 0): 0.0, (1, 0): -1.0, (0, 1): 1.0, (1, 1): 2.0}
RECIPROCAL_SQUARE = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0, (1, 1): 1.0}


def square3(kind, scale=(1.0, 1.3)):
    """Embed a pure epistasis square across a background locus (3 loci total).

    gamma* anchors (Ferretti 2016): magnitude -> +1, sign -> 0,
    reciprocal -> -1. The two background copies use slightly different
    magnitudes so the mutation effects vary (gamma is then a genuine
    correlation, not 0/0).
    """
    fmap = {}
    for c2, sc in zip([0, 1], scale):
        if kind == "magnitude":
            base = {(0, 0): 0.0, (1, 0): 1.0 * sc, (0, 1): 1.0 * sc, (1, 1): 3.0 * sc}
        elif kind == "sign":
            base = {(0, 0): 0.0, (1, 0): -1.0 * sc, (0, 1): 1.0 * sc, (1, 1): 2.0 * sc}
        elif kind == "reciprocal":
            base = {(0, 0): 1.0 * sc, (1, 0): 0.0, (0, 1): 0.0, (1, 1): 1.0 * sc}
        else:
            raise ValueError(kind)
        for (a, b), v in base.items():
            fmap[(a, b, c2)] = v
    return from_map(fmap, 3)


# A 3-cube with exactly two peaks (000 and 111) for basin/optima tests.
# Index i (0..7) is bitstring format(i, '03b'); fitness chosen so 000 and 111
# are the only out-degree-0 nodes, 000 is the global optimum.
TWO_PEAK_3CUBE = {
    (0, 0, 0): 10.0,  # peak A (global optimum)
    (0, 0, 1): 1.0,
    (0, 1, 0): 2.0,
    (0, 1, 1): 3.0,
    (1, 0, 0): 4.0,
    (1, 0, 1): 5.0,
    (1, 1, 0): 6.0,
    (1, 1, 1): 9.0,  # peak B
}
