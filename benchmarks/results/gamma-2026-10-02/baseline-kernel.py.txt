"""Gamma epistasis statistics (decay of fitness-effect correlation by distance)."""

import math
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .._utils import _pythonize, _pack_rows


def _gamma_effect_contribution(a, b, c, d):
    """Scaled second moments for parallel effects a-b and c-d.

    Track powers of two rather than squaring effects in their original units.
    This also permits pooling tiny observed squares when unrelated genotypes
    have enormous fitness. Signs are evaluated on the original values.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        bv, Bv = a - b, c - d
        num = float(np.dot(bv, Bv))
        den = 0.5 * float(np.dot(bv, bv) + np.dot(Bv, Bv))
    # Infinite differences still have the correct signs. Compute them before
    # rescaling, which could erase tiny nonzero effects in a mixed-scale batch.
    sb, sB = np.sign(bv), np.sign(Bv)
    snum = float(np.dot(sb, sB))
    sden = 0.5 * float(np.count_nonzero(sb) + np.count_nonzero(sB))
    # Ordinary units keep the original arithmetic and avoid extra array passes.
    # The conservative range leaves ample headroom when pooling finite inputs.
    if 2.0**-500 <= den <= 2.0**500:
        return num, den, snum, sden, 0
    shift = 0
    if np.isinf(bv).any() or np.isinf(Bv).any():
        # Finite endpoints can differ by up to twice float64's maximum.
        bv = np.ldexp(a, -1) - np.ldexp(b, -1)
        Bv = np.ldexp(c, -1) - np.ldexp(d, -1)
        shift = 1
    largest = max(np.max(np.abs(bv)), np.max(np.abs(Bv)))
    if largest == 0:
        return 0.0, 0.0, snum, sden, 0
    exponent = math.frexp(largest)[1]
    bv, Bv = np.ldexp(bv, -exponent), np.ldexp(Bv, -exponent)
    num = float(np.dot(bv, Bv))
    den = 0.5 * float(np.dot(bv, bv) + np.dot(Bv, Bv))
    return num, den, snum, sden, exponent + shift


def _merge_gamma_contributions(total, addition):
    """Pool raw moments at a shared exponent, retaining effect-size weights."""
    num, den, snum, sden, exponent = total
    n, d, sn, sd, e = addition
    if d:
        if not den:
            num, den, exponent = n, d, e
        else:
            common = max(exponent, e)
            num = math.ldexp(num, 2 * (exponent - common)) + math.ldexp(n, 2 * (e - common))
            den = math.ldexp(den, 2 * (exponent - common)) + math.ldexp(d, 2 * (e - common))
            exponent = common
    return num, den, snum + sn, sden + sd, exponent


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
    total = (0.0, 0.0, 0.0, 0.0, 0)
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
                    corners = [np.fromiter((g[k] for k in common), float, n)
                               for g in (g_ab, g_Ab, g_aB, g_AB)]
                    total = _merge_gamma_contributions(
                        total, _gamma_effect_contribution(*corners)
                    )
    return total


def _gamma_position_pair_worker(Xcodes, f, p1, p2, alleles1, alleles2, other):
    """Pooled gamma / gamma* contributions for one ordered position pair (p1, p2).

    Implements the Ferretti et al. (2016) correlation of fitness effects: for the
    p1-mutation, correlate its effect on backgrounds with allele ``b`` at p2 with
    its effect on backgrounds with allele ``B`` at p2, across all shared genetic
    backgrounds. The correlation is *non-centered* (a raw second-moment ratio, as
    in Eq. (1) of the paper), so it equals +1 for additive landscapes rather than
    being undefined. Returns ``(num, den, snum, sden, exponent)`` to be
    pooled across all ordered pairs by :func:`_gamma_statistics`, giving
    ``gamma = num / den`` and ``gamma_star = snum / sden`` after rescaling
    the numeric moments to a common power-of-two exponent.
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

    total = (0.0, 0.0, 0.0, 0.0, 0)
    # Same allele-quadruple loops as the dict path, but each (num,den,snum,sden)
    # update is vectorised over the background axis instead of set intersections.
    for ai in range(A1):
        for aj in range(ai + 1, A1):
            for bi in range(A2):
                g_ai_bi = G[:, ai, bi]
                g_aj_bi = G[:, aj, bi]
                for bj in range(bi + 1, A2):
                    corners = (g_ai_bi, g_aj_bi, G[:, ai, bj], G[:, aj, bj])
                    mask = ~np.logical_or.reduce([np.isnan(v) for v in corners])
                    if not mask.any():
                        continue
                    total = _merge_gamma_contributions(
                        total, _gamma_effect_contribution(*(v[mask] for v in corners))
                    )
    return total


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
    total = (0.0, 0.0, 0.0, 0.0, 0)
    for result in results:
        total = _merge_gamma_contributions(total, result)
    num, den, snum, sden, _ = total

    return {
        "gamma": num / den if den else np.nan,
        "gamma_star": snum / sden if sden else np.nan,
    }


def gamma(landscape, n_jobs=-1):
    """Measure the correlation of mutation effects across neighboring backgrounds.

    Parameters
    ----------
    landscape : Landscape
        Built landscape. Fitness is used on its supplied scale; apply any
        scientifically appropriate log transformation before construction.
        All observed allele pairs at two distinct variables are considered.
    n_jobs : int, default=-1
        Number of joblib workers. ``-1`` uses all available CPUs; ``1`` runs
        serially.

    Returns
    -------
    float
        Non-centered correlation in [-1, 1]. One means equal mutation effects
        across every observed square; negative values indicate opposing
        effects. Returns NaN if there are no complete squares or every effect
        on those squares is zero.

    Warns
    -----
    UserWarning
        If fewer than two variables remain in the built landscape.

    See Also
    --------
    gamma_star : Correlation of the signs of mutation effects.

    Notes
    -----
    For parallel effects a and b, pool their products and squared effects:
    ``sum(a*b) / sum((a*a + b*b)/2)`` over both directions of every complete
    two-variable square [1]_. Pool sums, rather than averaging square ratios.
    Every allele-pair combination has equal weight, including in multiallelic
    landscapes; fitness offsets and nonzero linear rescaling leave gamma
    unchanged.

    Missing corners exclude a square from both sums. This describes the
    observed squares, not the paper's separate distance-correlation estimator
    for missing data (Eq. (2)). Enumeration uses retained configurations,
    independently of graph edges, ordinal step restrictions and construction
    epsilon. A value of one on incomplete data does not establish global
    additivity.

    References
    ----------
    .. [1] Ferretti, L. et al. (2016). Measuring epistasis in fitness landscapes:
       The correlation of fitness effects of mutations. Journal of Theoretical
       Biology, 396, 132-143. Eqs. (1), (3), Appendix C.1.
       https://doi.org/10.1016/j.jtbi.2016.01.037

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import gamma
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> round(gamma(landscape, n_jobs=1), 6)
    0.888889
    """
    stats = _gamma_statistics(landscape, n_jobs=n_jobs)
    return _pythonize(stats["gamma"])


def gamma_star(landscape, n_jobs=-1):
    """Measure the correlation of mutation-effect signs across backgrounds.

    Parameters
    ----------
    landscape : Landscape
        Built landscape. Positive, zero and negative fitness differences are
        assigned +1, 0 and -1, respectively. Only exact ties are neutral;
        construction epsilon does not set a sign tolerance for this metric.
    n_jobs : int, default=-1
        Number of joblib workers. ``-1`` uses all available CPUs; ``1`` runs
        serially.

    Returns
    -------
    float
        Sign correlation in [-1, 1]. One indicates consistent nonzero signs,
        minus one indicates reversed signs, and zero indicates cancellation
        or absence of nonzero parallel products. Returns NaN if there are no
        complete squares or all effects on those squares are neutral.

    Warns
    -----
    UserWarning
        If fewer than two variables remain in the built landscape.

    See Also
    --------
    gamma : Square enumeration and pooling conventions shared by both metrics.
    classify_epistasis : Proportions of directed graph motifs.

    Notes
    -----
    Apply the pooling formula of :func:`gamma` to effect signs [1]_. A zero
    effect contributes zero to the numerator and its squared-effect term;
    it does not cause the entire square to be discarded. The function uses
    Eq. (11) with zero tolerance, not the optional positive-tolerance variant.

    The identity ``gamma_star = 1 - phi_sign - 2*phi_reciprocal`` (Eq. (12))
    requires no neutral effects and the same square population. It need not
    hold for :func:`classify_epistasis` when graph filtering removes edges,
    graph motifs differ from variable squares, or motif counts are sampled.

    References
    ----------
    .. [1] Ferretti, L. et al. (2016). Measuring epistasis in fitness landscapes:
       The correlation of fitness effects of mutations. Journal of Theoretical
       Biology, 396, 132-143. Eqs. (10)-(12), Appendix C.3.
       https://doi.org/10.1016/j.jtbi.2016.01.037

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import gamma_star
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 0, 1, 2], verbose=False)
    >>> round(gamma_star(landscape, n_jobs=1), 6)
    0.666667
    """
    stats = _gamma_statistics(landscape, n_jobs=n_jobs)
    return _pythonize(stats["gamma_star"])
