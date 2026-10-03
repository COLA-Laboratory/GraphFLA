"""Correlation of mutation effects on observed two-variable squares."""

import math
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .._utils import _pythonize, _pack_rows


# At most 8 MiB for a dense float64 grid per worker. Sparse groups avoid a
# Cartesian allocation when most allele combinations have not been observed.
_MAX_GRID_CELLS = 1 << 20
_EMPTY = (0.0, 0.0, 0.0, 0.0, 0)


def _numeric_moments(effects, counts):
    """Sum products of distinct background effects and their paired squares."""
    if effects.shape[1] == 2:
        return (
            float(np.dot(effects[:, 0], effects[:, 1])),
            0.5 * float(np.dot(effects.ravel(), effects.ravel())),
        )
    # Prefix products avoid subtracting two nearly equal squared sums. Each
    # effect occurs in exactly (count - 1) pairs, including zero effects.
    prefix = np.cumsum(effects, axis=1)
    num = float(np.einsum("ij,ij->", effects[:, 1:], prefix[:, :-1]))
    squares = np.einsum("ij,ij->i", effects, effects)
    den = 0.5 * float(np.dot(counts - 1, squares))
    return num, den


def _background_moments(a, b, valid, statistic):
    """Pool one focal allele pair over all pairs of observed backgrounds."""
    counts = np.count_nonzero(valid, axis=1)
    usable = counts >= 2
    if not usable.any():
        return _EMPTY
    a, b, valid, counts = a[usable], b[usable], valid[usable], counts[usable]
    with np.errstate(over="ignore", invalid="ignore"):
        effects = np.where(valid, a - b, 0.0)
    snum = sden = 0.0
    if statistic != "gamma":
        # Take signs before any scaling: even a tiny nonzero effect counts.
        signs = np.sign(effects)
        signed_sum = signs.sum(axis=1)
        nonzero = np.count_nonzero(signs, axis=1)
        snum = 0.5 * float(np.sum(signed_sum * signed_sum - nonzero))
        sden = 0.5 * float(np.dot(counts - 1, nonzero))
    if statistic == "gamma_star":
        return 0.0, 0.0, snum, sden, 0
    with np.errstate(over="ignore", invalid="ignore"):
        num, den = _numeric_moments(effects, counts)
    if 2.0**-500 <= den <= 2.0**500:
        return num, den, snum, sden, 0
    shift = 0
    if np.isinf(effects).any():
        # Finite endpoints can differ by up to twice float64's maximum.
        effects = np.where(valid, np.ldexp(a, -1) - np.ldexp(b, -1), 0.0)
        shift = 1
    largest = np.max(np.abs(effects))
    if largest == 0:
        return 0.0, 0.0, snum, sden, 0
    exponent = math.frexp(largest)[1]
    num, den = _numeric_moments(np.ldexp(effects, -exponent), counts)
    return num, den, snum, sden, exponent + shift


def _merge_gamma_contributions(total, addition):
    """Pool raw moments at a shared exponent, retaining effect-size weights."""
    num, den, snum, sden, exponent = total
    n, d, sn, sd, e = addition
    if d:
        if not den:
            num, den, exponent = n, d, e
        elif exponent == e:
            num, den = num + n, den + d
        else:
            common = max(exponent, e)
            num = math.ldexp(num, 2 * (exponent - common)) + math.ldexp(
                n, 2 * (e - common)
            )
            den = math.ldexp(den, 2 * (exponent - common)) + math.ldexp(
                d, 2 * (e - common)
            )
            exponent = common
    return num, den, snum + sn, sden + sd, exponent


def _gamma_grid_moments(grid, statistic):
    """One focal position; vectorize over its background alleles and groups."""
    total = _EMPTY
    for ai, aj in combinations(range(grid.shape[1]), 2):
        a, b = grid[:, ai, :], grid[:, aj, :]
        valid = ~(np.isnan(a) | np.isnan(b))
        total = _merge_gamma_contributions(
            total, _background_moments(a, b, valid, statistic)
        )
    return total


def _gamma_pair_via_dict(
    Xcodes, f, p1, p2, alleles1, alleles2, other, *, both=False, statistic=None
):
    """Enumerate observed groups without allocating an allele Cartesian grid."""
    groups = {}
    for i, background in enumerate(Xcodes[:, other]):
        groups.setdefault(background.tobytes(), []).append(i)
    total = _EMPTY
    directions = [(p1, p2), (p2, p1)] if both else [(p1, p2)]
    for indices in groups.values():
        if len(indices) < 4:
            continue
        for focal, background in directions:
            rows = {}
            for i in indices:
                rows.setdefault(Xcodes[i, focal], {})[Xcodes[i, background]] = f[i]
            # An allele with only one background cannot belong to a square.
            rows = [row for row in rows.values() if len(row) >= 2]
            for left, right in combinations(rows, 2):
                common = sorted(left.keys() & right.keys())
                if len(common) < 2:
                    continue
                a = np.array([[left[key] for key in common]])
                b = np.array([[right[key] for key in common]])
                total = _merge_gamma_contributions(
                    total,
                    _background_moments(a, b, np.ones(a.shape, dtype=bool), statistic),
                )
    return total


def _gamma_position_pair_worker(
    Xcodes, f, p1, p2, alleles1, alleles2, other, *, both=False, statistic=None
):
    """Return (num, den, sign_num, sign_den, exponent) for a position pair.

    Each focal allele pair compares its effects across every pair of background
    alleles. Missing corners are excluded from both moments. ``both`` reuses
    background grouping for the reverse focal position; the default preserves
    the ordered-pair convention used in the independent literature checks.
    """
    bg_ids, n_bg = _pack_rows(Xcodes[:, other])
    if bg_ids is None:
        return _gamma_pair_via_dict(
            Xcodes, f, p1, p2, alleles1, alleles2, other, both=both, statistic=statistic
        )
    keep = np.bincount(bg_ids)[bg_ids] >= 4
    if not keep.any():
        return _EMPTY
    # Fewer than four configurations in a background cannot complete a square.
    # Compact retained groups/alleles before estimating the dense allocation.
    if keep.all():
        inverse = bg_ids
        a1_local = np.searchsorted(alleles1, Xcodes[:, p1])
        a2_local = np.searchsorted(alleles2, Xcodes[:, p2])
        A1, A2 = len(alleles1), len(alleles2)
    else:
        retained, inverse = np.unique(bg_ids[keep], return_inverse=True)
        n_bg = len(retained)
        _, a1_local = np.unique(Xcodes[keep, p1], return_inverse=True)
        _, a2_local = np.unique(Xcodes[keep, p2], return_inverse=True)
        A1, A2 = int(a1_local.max()) + 1, int(a2_local.max()) + 1
    cells = n_bg * A1 * A2
    if cells > _MAX_GRID_CELLS or cells > 8 * np.count_nonzero(keep):
        return _gamma_pair_via_dict(
            Xcodes, f, p1, p2, alleles1, alleles2, other, both=both, statistic=statistic
        )
    grid = np.full((n_bg, A1, A2), np.nan)
    grid[inverse, a1_local, a2_local] = f[keep]
    total = _gamma_grid_moments(grid, statistic)
    if both:
        total = _merge_gamma_contributions(
            total, _gamma_grid_moments(grid.swapaxes(1, 2), statistic)
        )
    return total


def _gamma_statistics(landscape, n_jobs=-1, *, statistic=None):
    """Calculate requested moments; None also supports joint internal checks."""
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )
    if landscape.n_vars < 2:
        warnings.warn(
            "Gamma statistics require at least 2 variables so that fitness "
            "effects of one mutation can be compared; this landscape has "
            f"{landscape.n_vars}. Returning NaN.",
            UserWarning,
            stacklevel=3,
        )
        return {"gamma": np.nan, "gamma_star": np.nan}
    df = landscape.get_data()
    f = df["fitness"].to_numpy(dtype=float)
    Xcodes = np.column_stack(
        [pd.factorize(df[c])[0] for c in landscape.data_types]
    ).astype(np.int32)
    P = Xcodes.shape[1]
    alleles = [np.unique(Xcodes[:, j]) for j in range(P)]

    def contributions():
        for p1, p2 in combinations(range(P), 2):
            yield (
                Xcodes,
                f,
                p1,
                p2,
                alleles[p1],
                alleles[p2],
                np.delete(np.arange(P), [p1, p2]),
            )

    if n_jobs == 1:
        results = (
            _gamma_position_pair_worker(*args, both=True, statistic=statistic)
            for args in contributions()
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_gamma_position_pair_worker)(*args, both=True, statistic=statistic)
            for args in contributions()
        )
    total = _EMPTY
    for result in results:
        total = _merge_gamma_contributions(total, result)
    num, den, snum, sden, _ = total
    return {
        "gamma": num / den if den else np.nan,
        "gamma_star": snum / sden if sden else np.nan,
    }


def gamma(landscape, n_jobs=-1) -> float:
    r"""Return the correlation of mutation effects across neighboring backgrounds.

    Parameters
    ----------
    landscape : Landscape
        Built landscape. Fitness is used on its supplied scale; apply any
        scientifically appropriate log transformation before construction.
        All observed allele pairs at two distinct variables are considered.
        Retained configurations define the population, independently of graph
        edges, ordinal step restrictions and construction epsilon.
    n_jobs : int or None, default=-1
        Number of joblib workers. ``-1`` uses all available CPUs and ``1`` runs
        serially. ``None`` follows the active joblib configuration, or uses
        one worker without a configuration. Zero is invalid.

    Returns
    -------
    gamma_value : float
        Non-centered correlation in [-1, 1]. One means equal mutation effects
        across every observed square; negative values indicate opposing
        effects. Returns NaN if there are no complete squares or every effect
        on those squares is zero.

    Raises
    ------
    graphfla.exceptions.NotBuiltError
        If the landscape has not been built.
    ValueError
        If the graph lacks fitness values, or the worker count is invalid.

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
    two-variable square [1]_. Each allele-pair combination has equal weight;
    averaging square ratios gives a different statistic. Fitness offsets and
    nonzero linear rescaling leave gamma unchanged.

    Missing corners exclude a square from both sums. This describes the
    observed squares, not the paper's separate distance-correlation estimator
    for missing data (Eq. (2)). A value of one on incomplete data does not
    establish global additivity.

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
    stats = _gamma_statistics(landscape, n_jobs=n_jobs, statistic="gamma")
    return _pythonize(stats["gamma"])


def gamma_star(landscape, n_jobs=-1) -> float:
    r"""Return the correlation of mutation-effect signs across backgrounds.

    Parameters
    ----------
    landscape : Landscape
        Built landscape. Positive, zero and negative fitness differences are
        assigned +1, 0 and -1, respectively. Only exact ties are neutral;
        construction epsilon does not set a sign tolerance for this metric.
    n_jobs : int or None, default=-1
        Number of joblib workers. ``-1`` uses all available CPUs and ``1`` runs
        serially. ``None`` follows the active joblib configuration, or uses
        one worker without a configuration. Zero is invalid.

    Returns
    -------
    gamma_star_value : float
        Sign correlation in [-1, 1]. One indicates consistent nonzero signs,
        minus one indicates reversed signs, and zero indicates cancellation
        or absence of nonzero parallel products. Returns NaN if there are no
        complete squares or all effects on those squares are neutral.

    Raises
    ------
    graphfla.exceptions.NotBuiltError
        If the landscape has not been built.
    ValueError
        If the graph lacks fitness values, or the worker count is invalid.

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
    stats = _gamma_statistics(landscape, n_jobs=n_jobs, statistic="gamma_star")
    return _pythonize(stats["gamma_star"])
