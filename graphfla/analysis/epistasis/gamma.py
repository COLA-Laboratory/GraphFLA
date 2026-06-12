"""Gamma epistasis statistics (decay of fitness-effect correlation by distance)."""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .._utils import _pythonize, _pack_rows


def _gamma_pair_via_dict(Xcodes, f, p1, p2, alleles1, alleles2, other):
    """Original dict-grouping gamma pair contribution; the high-dimensional
    fallback used when the background does not pack into an int64 key."""
    col1 = Xcodes[:, p1]
    col2 = Xcodes[:, p2]
    bg = Xcodes[:, other]
    groups = {}
    for i in range(Xcodes.shape[0]):
        key = (col1[i], col2[i])
        d = groups.get(key)
        if d is None:
            d = groups[key] = {}
        bk = bg[i].tobytes()
        if bk not in d:
            d[bk] = f[i]
    num = den = snum = sden = 0.0
    for ai in range(len(alleles1)):
        for aj in range(ai + 1, len(alleles1)):
            a, A_ = alleles1[ai], alleles1[aj]
            for bi in range(len(alleles2)):
                for bj in range(bi + 1, len(alleles2)):
                    b, B_ = alleles2[bi], alleles2[bj]
                    g_ab = groups.get((a, b))
                    g_Ab = groups.get((A_, b))
                    g_aB = groups.get((a, B_))
                    g_AB = groups.get((A_, B_))
                    if not (g_ab and g_Ab and g_aB and g_AB):
                        continue
                    common = g_ab.keys() & g_Ab.keys() & g_aB.keys() & g_AB.keys()
                    if not common:
                        continue
                    common = list(common)
                    n = len(common)
                    bvec = np.fromiter((g_ab[k] - g_Ab[k] for k in common), float, n)
                    Bvec = np.fromiter((g_aB[k] - g_AB[k] for k in common), float, n)
                    num += float(np.dot(bvec, Bvec))
                    den += 0.5 * float(np.dot(bvec, bvec) + np.dot(Bvec, Bvec))
                    sb = np.sign(bvec)
                    sB = np.sign(Bvec)
                    snum += float(np.dot(sb, sB))
                    sden += 0.5 * float(np.count_nonzero(sb) + np.count_nonzero(sB))
    return num, den, snum, sden


def _gamma_position_pair_worker(Xcodes, f, p1, p2, alleles1, alleles2, other):
    """Pooled gamma / gamma* contributions for one ordered position pair (p1, p2).

    Implements the Ferretti et al. (2016) correlation of fitness effects: for the
    p1-mutation, correlate its effect on backgrounds with allele ``b`` at p2 with
    its effect on backgrounds with allele ``B`` at p2, across all shared genetic
    backgrounds. The correlation is *non-centered* (a raw second-moment ratio, as
    in eq. 3 of the paper), so it equals +1 for additive landscapes rather than
    being undefined. Returns the partial sums ``(num, den, snum, sden)`` to be
    pooled across all ordered pairs by :func:`_gamma_statistics`, giving
    ``gamma = num / den`` and ``gamma_star = snum / sden``.
    """
    # Group nodes by background. When the background columns pack into an int64
    # key (boolean / DNA / ordinal / low-dimensional protein) this is a fast 1D
    # unique and the allele-quadruple correlation vectorises over a fitness grid.
    # For high-cardinality, many-column backgrounds (high-dim protein) packing
    # overflows int64 -- there the original dict grouping is faster, so fall back.
    bg_ids, n_bg = _pack_rows(Xcodes[:, other])
    if bg_ids is None:
        return _gamma_pair_via_dict(Xcodes, f, p1, p2, alleles1, alleles2, other)

    col1 = Xcodes[:, p1]
    col2 = Xcodes[:, p2]
    A1 = len(alleles1)
    A2 = len(alleles2)
    # alleles1/alleles2 are sorted-unique, so searchsorted gives the local index.
    a1_local = np.searchsorted(alleles1, col1)
    a2_local = np.searchsorted(alleles2, col2)
    # G[bg, i, j] = fitness of the genotype (alleles1[i] @ p1, alleles2[j] @ p2, bg);
    # NaN where that genotype is absent. Each genotype is unique -> no cell collides.
    G = np.full((n_bg, A1, A2), np.nan)
    G[bg_ids, a1_local, a2_local] = f

    num = den = snum = sden = 0.0
    # Same allele-quadruple loops as the dict path, but each (num,den,snum,sden)
    # update is vectorised over the background axis instead of set intersections.
    for ai in range(A1):
        for aj in range(ai + 1, A1):
            for bi in range(A2):
                g_ai_bi = G[:, ai, bi]
                g_aj_bi = G[:, aj, bi]
                for bj in range(bi + 1, A2):
                    bvec = g_ai_bi - g_aj_bi          # effect at p2=bi: f(a,b)-f(A,b)
                    Bvec = G[:, ai, bj] - G[:, aj, bj]  # effect at p2=bj: f(a,B)-f(A,B)
                    mask = ~(np.isnan(bvec) | np.isnan(Bvec))  # shared backgrounds
                    if not mask.any():
                        continue
                    bv = bvec[mask]
                    Bv = Bvec[mask]
                    num += float(np.dot(bv, Bv))
                    den += 0.5 * float(np.dot(bv, bv) + np.dot(Bv, Bv))
                    sb = np.sign(bv)
                    sB = np.sign(Bv)
                    snum += float(np.dot(sb, sB))
                    sden += 0.5 * float(np.count_nonzero(sb) + np.count_nonzero(sB))
    return num, den, snum, sden


def _gamma_statistics(landscape, n_jobs=-1):
    """Calculate both gamma statistics for internal reuse."""
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    df = landscape.get_data()
    X = df[list(landscape.data_types.keys())]

    if landscape.n_vars < 2:
        warnings.warn(
            "Gamma statistics require at least 2 variables so that fitness "
            f"effects of one mutation can be compared; this landscape has "
            f"{landscape.n_vars}. Returning NaN.",
            UserWarning,
        )
        return {"gamma": np.nan, "gamma_star": np.nan}

    # Appearance-order codes (match original df[pos].unique() iteration); memmapped to workers.
    f = df["fitness"].to_numpy(dtype=float)
    Xcodes = np.column_stack([pd.factorize(X[c])[0] for c in X.columns]).astype(np.int32)
    P = Xcodes.shape[1]
    alleles = [np.unique(Xcodes[:, j]) for j in range(P)]
    position_pairs = [(p1, p2) for p1 in range(P) for p2 in range(P) if p1 != p2]

    # Process-based (loky) parallelism over ordered position pairs; thread backend was GIL-bound.
    results = Parallel(n_jobs=n_jobs)(
        delayed(_gamma_position_pair_worker)(
            Xcodes, f, p1, p2, alleles[p1], alleles[p2], np.delete(np.arange(P), [p1, p2])
        )
        for p1, p2 in position_pairs
    )

    # Pool over all ordered pairs (both orderings cover both square sides) into
    # the single global non-centered correlation of Ferretti et al. (2016).
    num = sum(r[0] for r in results)
    den = sum(r[1] for r in results)
    snum = sum(r[2] for r in results)
    sden = sum(r[3] for r in results)

    return {
        "gamma": num / den if den else np.nan,
        "gamma_star": snum / sden if sden else np.nan,
    }


def gamma(landscape, n_jobs=-1):
    """
    Calculates the gamma and gamma_star statistics for a fitness landscape.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object containing fitness data.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (all available cores).

    Returns
    -------
    float
        The traditional gamma statistic value. Values close to -1 or 1 indicate
        strong epistatic interactions in magnitude, while values close to 0 indicate
        weak or no epistasis.

    Notes
    -----
    - The gamma statistic measures the correlation between fitness effects of mutations
      across different genetic backgrounds, providing a measure of epistatic interactions
      in the landscape.
    - It is computed as the non-centered (raw second-moment) correlation of fitness
      effects pooled over all square motifs, following eq. (3) of Ferretti et al.
      (2016). Hence a purely additive landscape gives gamma = 1, a House-of-Cards
      landscape gives gamma ~ 0, and a reciprocal-sign-dominated landscape gives
      gamma < 0.
    - The gamma_star statistic focuses only on sign consistency, ignoring the magnitude
      of fitness effects. It indicates whether mutations tend to have consistent
      directional effects across different genetic backgrounds.

    References
    ----------
    .. [1] L. Ferretti et al., "Measuring epistasis in fitness landscapes: The
       correlation of fitness effects of mutations", J. Theor. Biol. 396, 132-143 (2016).
    """

    stats = _gamma_statistics(landscape, n_jobs=n_jobs)
    return _pythonize(stats["gamma"])


def gamma_star(landscape, n_jobs=-1):
    """
    Calculate the gamma-star statistic for a fitness landscape.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object containing fitness data.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (all available cores).

    Returns
    -------
    float
        The gamma-star statistic that only considers sign consistency.
        Values close to 1 indicate consistent sign epistasis across
        backgrounds, values close to -1 indicate opposing sign patterns,
        and values close to 0 indicate random sign patterns.
    """
    stats = _gamma_statistics(landscape, n_jobs=n_jobs)
    return _pythonize(stats["gamma_star"])
