"""Mutation-effect variation (Lyons et al. 2020) and fitness-trend indices."""

import warnings
from itertools import permutations
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .._utils import _pack_rows
from ._fitness_trends import _edge_fitness_trend


def _validate_min_pairs(min_pairs):
    if (
        isinstance(min_pairs, bool)
        or not isinstance(min_pairs, Integral)
        or min_pairs < 2
    ):
        raise ValueError("min_pairs must be an integer of at least 2.")


def _idiosyncratic_data(landscape):
    """Read the retained landscape population without altering its fitness scale."""
    data = landscape.get_data()
    if landscape.data_types is None:
        raise ValueError("Configuration columns are required to match backgrounds.")
    X = data[list(landscape.data_types)]
    f = data["fitness"].to_numpy(dtype=float)
    if not np.all(np.isfinite(f)):
        raise ValueError("Fitness values must be finite.")
    if X.isna().any().any():
        raise ValueError("Configuration values must not be missing.")
    if X.duplicated().any():
        raise ValueError("Configurations must be unique; aggregate replicates first.")
    codes, labels = [], []
    for col in X:
        c, alleles = pd.factorize(X[col], sort=True)
        codes.append(c)
        labels.append(alleles)
    Xcodes = (
        np.column_stack(codes).astype(np.int32)
        if codes
        else np.empty((len(f), 0), dtype=np.int32)
    )
    return X, Xcodes, f, labels


def _idiosyncratic_position_worker(Xcodes, f, j):
    """Return (allele A, allele B, effect SD, count) for each directed mutation.

    Background matching is shared by every allele pair at a position. No random
    numbers are drawn in workers, so job scheduling cannot change the estimate.
    """
    other = np.delete(np.arange(Xcodes.shape[1]), j)
    col = Xcodes[:, j]
    alleles = np.unique(col)
    if len(alleles) < 2 or not len(f):
        return []
    bg_ids, n_bg = _pack_rows(Xcodes[:, other])
    summaries = {}
    if bg_ids is not None:
        fit = []
        for a in alleles:
            arr = np.full(n_bg, np.nan)
            rows = np.flatnonzero(col == a)
            arr[bg_ids[rows]] = f[rows]
            fit.append(arr)
        for ai, a in enumerate(alleles):
            for bi in range(ai + 1, len(alleles)):
                b = alleles[bi]
                mask = np.isfinite(fit[ai]) & np.isfinite(fit[bi])
                effects = fit[bi][mask] - fit[ai][mask]
                summaries[a, b] = (
                    float(np.std(effects)) if len(effects) else np.nan,
                    len(effects),
                )
    else:
        # Mixed-radix packing can overflow on long sequences. Byte keys preserve
        # exact background identity without constructing a dense genotype cube.
        bgcols = Xcodes[:, other]
        fit = [
            {bgcols[i].tobytes(): f[i] for i in np.flatnonzero(col == a)}
            for a in alleles
        ]
        for ai, a in enumerate(alleles):
            for bi in range(ai + 1, len(alleles)):
                b = alleles[bi]
                common = sorted(fit[ai].keys() & fit[bi].keys())
                effects = np.asarray([fit[bi][k] - fit[ai][k] for k in common])
                summaries[a, b] = (
                    float(np.std(effects)) if len(effects) else np.nan,
                    len(effects),
                )
    return [
        (a, b, *summaries[min(a, b), max(a, b)]) for a, b in permutations(alleles, 2)
    ]


def _idiosyncratic_ratio(effect_sd, n_pairs, fitness_pool, rng):
    """Lyons' matched-size control: one with-replacement draw of n pairs.

    Keeping this numeric kernel separate allows exact reproduction of a study's
    recorded RNG stream without embedding its incidental seeds in the metric.
    """
    pairs = rng.choice(fitness_pool, size=(n_pairs, 2), replace=True)
    control_sd = float(np.std(pairs[:, 1] - pairs[:, 0]))
    if control_sd == 0:
        warnings.warn(
            "The sampled random-pair control has zero standard deviation; "
            "the idiosyncratic index is undefined (NaN).",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(effect_sd / control_sd)


def idiosyncratic_index(landscape, mutation, min_pairs: int = 3, *, seed=None) -> float:
    r"""Return a mutation's idiosyncratic index from matched backgrounds.

    The index compares the standard deviation of one mutation's effects with
    that of an equally sized sample of random genotype-pair differences [1]_.
    Effects and controls use the fitness scale supplied in the landscape.

    Parameters
    ----------
    landscape : Landscape
        Built landscape with unique configurations and finite fitness values.
        Both matched backgrounds and random controls use the genotypes retained
        in ``landscape.get_data()``.
    mutation : tuple of (source, position, target)
        Allele substitution to evaluate. ``position`` is a configuration-column
        label from ``landscape.data_types``, not a positional column index.
        ``source`` and ``target`` must be distinct observed alleles at that
        position. For example, ``(0, "bit_0", 1)`` changes the first Boolean
        feature from 0 to 1.
    min_pairs : int, default=3
        Minimum number of observed matching backgrounds, at least 2. All
        matching backgrounds are used when this threshold is met. The control
        then contains the same number of random pairs. This threshold is a
        GraphFLA estimation guard, not a cutoff specified in [1]_.
    seed : int or None, default=None
        Seed for a local NumPy RandomState. An integer reproduces the control
        sample for the same ordered input; None starts a fresh random stream.
        NumPy's global RNG state is not modified.

    Returns
    -------
    index : float
        Ratio of observed-effect SD to sampled-control SD. Values can exceed 1.
        Returns NaN for a constant-fitness landscape, fewer than ``min_pairs``
        backgrounds, or a sampled control with zero SD. A well-defined value of
        zero indicates constant mutation effects across the observed backgrounds.

    Raises
    ------
    graphfla.exceptions.NotBuiltError
        If the landscape has not been built.
    ValueError
        If ``min_pairs`` or seed is invalid, the mutation uses an unknown
        position or allele, or source and target are equal. Also raised for missing
        configuration values, duplicate configurations, or nonfinite fitness.

    Warns
    -----
    RuntimeWarning
        If the sampled control has zero SD. No replacement sample is drawn.

    See Also
    --------
    global_idiosyncratic_index : Average the index over directed mutations.
    gamma : Measure correlations of mutation effects between nearby backgrounds.

    Notes
    -----
    Match genotypes at all positions except the focal one. Divide the SD of
    their fitness differences by that of equally many random-pair differences,
    sampling both endpoints independently with replacement. Both SDs use
    ``ddof=0``; matching does not depend on graph edges.

    With ``seed=None``, results can differ across calls. Fitness is used as
    supplied. The index measures background dependence, including
    variation that can arise from a nonlinear global fitness map.

    References
    ----------
    .. [1] Lyons, Daniel M., Zhengting Zou, Haiqing Xu, and Jianzhi Zhang.
       "Idiosyncratic epistasis creates universals in mutational effects and
       evolutionary trajectories." Nature Ecology & Evolution 4 (2020):
       1685-1693. https://doi.org/10.1038/s41559-020-01286-y.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import idiosyncratic_index
    >>> sequences = ["000", "001", "010", "011", "100", "101", "110", "111"]
    >>> fitness = [0., 1., 2., 3., 1., 2., 4., 6.]
    >>> landscape = BooleanLandscape().build_from_data(
    ...     sequences, fitness, epsilon=0, verbose=False
    ... )
    >>> value = idiosyncratic_index(landscape, (0, "bit_0", 1), seed=0)
    >>> round(value, 3)
    0.308
    """
    _validate_min_pairs(min_pairs)
    rng = np.random.RandomState(seed)
    A, pos, B = mutation
    X, Xcodes, f, labels = _idiosyncratic_data(landscape)
    if pos not in X.columns:
        raise ValueError(f"Position {pos!r} is not a configuration column.")
    j = X.columns.get_loc(pos)
    for allele in (A, B):
        if allele not in labels[j]:
            raise ValueError(f"Allele {allele!r} not found at position {pos!r}.")
    if A == B:
        raise ValueError("A mutation must change the allele (A != B).")
    if len(f) < 2 or np.all(f == f[0]):
        return float("nan")
    a, b = labels[j].get_loc(A), labels[j].get_loc(B)
    for aa, bb, effect_sd, n in _idiosyncratic_position_worker(Xcodes, f, j):
        if (aa, bb) == (a, b):
            if n < min_pairs:
                return float("nan")
            return _idiosyncratic_ratio(effect_sd, n, f, rng)
    return float("nan")


def global_idiosyncratic_index(
    landscape, n_jobs=-1, seed=None, min_pairs: int = 3
) -> float:
    r"""Return the mean idiosyncratic index across directed mutations.

    Apply the SD ratio defined by Lyons et al. [1]_ to each eligible directed
    mutation and return their arithmetic mean. Every mutation receives equal
    weight, irrespective of its number of observed backgrounds.

    Parameters
    ----------
    landscape : Landscape
        Built landscape with unique configurations and finite fitness values.
        Mutation effects and random controls both use the genotypes retained
        in ``landscape.get_data()``. Apply the intended fitness transformation
        and population selection before constructing the landscape.
    n_jobs : int or None, default=-1
        Number of parallel jobs for background matching. -1 uses all available
        cores; 1 runs serially. This does not change a fixed-seed result.
    seed : int or None, default=None
        Seed for a local NumPy RandomState. An integer reproduces the result for
        the same ordered input and parameters. None starts a fresh random stream.
        NumPy's global RNG state is not modified.
    min_pairs : int, default=3
        Minimum number of matching backgrounds for a mutation to contribute,
        at least 2. Mutations below this threshold are omitted. All matching
        backgrounds of eligible mutations are used, with equally sized random
        controls. The paper specifies no minimum for this index.

    Returns
    -------
    mean_index : float
        Arithmetic mean of the eligible mutation indices. Values can exceed 1.
        Returns NaN for constant fitness, no eligible mutations, or a zero-SD
        control for any eligible mutation. A failed control is not omitted
        from the mean or resampled.

    Raises
    ------
    graphfla.exceptions.NotBuiltError
        If the landscape has not been built.
    ValueError
        If ``min_pairs`` or the random seed is invalid, or configurations are
        missing or duplicated, or fitness contains nonfinite values.

    Warns
    -----
    RuntimeWarning
        If any eligible mutation's sampled control has zero SD.

    See Also
    --------
    idiosyncratic_index : Estimate the index for one specified mutation.
    gamma : Measure correlations of mutation effects between nearby backgrounds.

    Notes
    -----
    Apply :func:`idiosyncratic_index`'s calculation to each ordered allele pair
    at each position. Forward and reverse mutations receive independent
    controls. Average over eligible mutations, not positions or backgrounds.

    Matching uses configuration values rather than graph edges. Controls use
    only retained genotypes, so graph pruning changes the reference population.

    References
    ----------
    .. [1] Lyons, Daniel M., Zhengting Zou, Haiqing Xu, and Jianzhi Zhang.
       "Idiosyncratic epistasis creates universals in mutational effects and
       evolutionary trajectories." Nature Ecology & Evolution 4 (2020):
       1685-1693. https://doi.org/10.1038/s41559-020-01286-y.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import global_idiosyncratic_index
    >>> sequences = ["000", "001", "010", "011", "100", "101", "110", "111"]
    >>> fitness = [0., 1., 2., 3., 1., 2., 4., 6.]
    >>> landscape = BooleanLandscape().build_from_data(
    ...     sequences, fitness, epsilon=0, verbose=False
    ... )
    >>> round(global_idiosyncratic_index(landscape, n_jobs=1, seed=0), 3)
    0.416
    """
    _validate_min_pairs(min_pairs)
    rng = np.random.RandomState(seed)
    _, Xcodes, f, _ = _idiosyncratic_data(landscape)
    if len(f) < 2 or np.all(f == f[0]):
        return float("nan")
    per_pos = Parallel(n_jobs=n_jobs)(
        delayed(_idiosyncratic_position_worker)(Xcodes, f, j)
        for j in range(Xcodes.shape[1])
    )
    values = [
        _idiosyncratic_ratio(effect_sd, n, f, rng)
        for rows in per_pos
        for _, _, effect_sd, n in rows
        if n >= min_pairs
    ]
    return float(np.mean(values)) if values else float("nan")


def diminishing_returns_index(
    landscape,
    method: Literal["pearson", "spearman", "regression"] = "pearson",
) -> float:
    r"""Return the pooled trend of beneficial effects with background fitness.

    Parameters
    ----------
    landscape : Landscape
        Built landscape with a directed graph of improving transitions and
        finite node fitness. Use a one-step neighborhood for mutation-level
        interpretation. The graph's retained edges determine the population;
        construction filters and missing configurations are not undone.
    method : {"pearson", "spearman", "regression"}, default="pearson"
        Pearson correlation, Spearman correlation with average ranks for ties,
        or the ordinary least-squares slope with an intercept. Each edge has
        equal weight. Effects are differences on the supplied fitness scale.

    Returns
    -------
    index : float
        Trend between starting fitness and the positive improvement for each
        retained edge. Negative values describe smaller gains on better
        backgrounds. Fitness is negated for minimization, so the interpretation
        is unchanged. Returns NaN for fewer than two eligible edges or constant
        starting fitness. A constant effect gives NaN for correlation and zero
        for regression. No significance test or p-value is returned.

    Raises
    ------
    RuntimeError
        If the landscape has not been built.
    ValueError
        If method is invalid, the graph or fitness is missing, fitness is
        nonfinite, or the graph is undirected or contains worsening edges.

    Warns
    -----
    UserWarning
        If the requested statistic is undefined.

    See Also
    --------
    increasing_costs_index : Trend of reverse-mutation cost with fitness.

    Notes
    -----
    For each stored improving edge u -> v, correlate q(u) with q(v) - q(u),
    where q = fitness for maximization and q = -fitness for minimization.
    Zero-effect edges are excluded. This pools individual transitions [1]_,
    rather than averaging effects per node or tracking a fixed mutation across
    backgrounds. Edge attributes such as ``delta_fit`` are not used.

    The result is descriptive: effect-sign selection, differing mutation
    composition (even in an additive landscape) and shared measurement error
    can produce a trend without demonstrating mutation-specific global
    epistasis. Fitness transformations can change it. Pearson and regression
    take O(V + E) time and O(V + B)
    auxiliary memory with fixed block size B; exact Spearman requires O(E)
    additional memory and O(E log E) time.

    References
    ----------
    .. [1] Huang, M., Zhou, S. and Li, K. (2025). Augmenting Biological Fitness
       Prediction Benchmarks with Landscapes Features from GraphFLA. NeurIPS 38,
       Appendix C.3.2. https://doi.org/10.52202/085713-1180.
    .. [2] Papkou, A. et al. (2023). A rugged yet easily navigable fitness
       landscape. Science 382, eadh3860, Fig. S22.
       https://doi.org/10.1126/science.adh3860.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import diminishing_returns_index
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0., 2., 3., 4.],
    ...     epsilon=0, verbose=False
    ... )
    >>> round(diminishing_returns_index(landscape, method="regression"), 3)
    -0.444
    """
    return _edge_fitness_trend(landscape, method, costs=False)


def increasing_costs_index(
    landscape,
    method: Literal["pearson", "spearman", "regression"] = "pearson",
) -> float:
    r"""Return the pooled trend of deleterious cost with background fitness.

    Parameters
    ----------
    landscape : Landscape
        Built landscape with a directed graph of improving transitions and
        finite node fitness. Reverse each retained edge to represent a worsening
        move. A mutation-level interpretation requires a reversible, one-step
        neighborhood. Construction filters and missing configurations remain
        part of the input population.
    method : {"pearson", "spearman", "regression"}, default="pearson"
        Pearson correlation, Spearman correlation with average ranks for ties,
        or the ordinary least-squares slope with an intercept. Each reverse
        transition has equal weight; its cost is a positive fitness difference.

    Returns
    -------
    index : float
        Trend between the better endpoint's fitness and the cost of moving to
        its worse neighbor. Positive values describe larger costs on better
        backgrounds. Fitness is negated for minimization. Returns NaN for fewer
        than two eligible edges or constant background fitness. A constant cost
        gives NaN for correlation and zero for regression. No significance test
        or p-value is returned.

    Raises
    ------
    RuntimeError
        If the landscape has not been built.
    ValueError
        If method is invalid, the graph or fitness is missing, fitness is
        nonfinite, or the graph is undirected or contains worsening edges.

    Warns
    -----
    UserWarning
        If the requested statistic is undefined.

    See Also
    --------
    diminishing_returns_index : Corresponding beneficial-effect trend and
        shared interpretation and resource limits.

    Notes
    -----
    For each stored improving edge u -> v, correlate q(v) with q(v) - q(u),
    using the same oriented fitness and edge population as
    ``diminishing_returns_index``. Zero-effect edges are excluded. This is the
    pooled cost convention of [1]_; it is not the distribution of
    mutation-specific regressions studied by Johnson et al. [2]_. Its sign
    alone does not establish global epistasis or statistical significance.

    References
    ----------
    .. [1] Huang, M., Zhou, S. and Li, K. (2025). Augmenting Biological Fitness
       Prediction Benchmarks with Landscapes Features from GraphFLA. NeurIPS 38,
       Appendix C.3.2. https://doi.org/10.52202/085713-1180.
    .. [2] Johnson, M. S. et al. (2019). Higher-fitness yeast genotypes are less
       robust to deleterious mutations. Science 366, 490-493, Figs. 3-4.
       https://doi.org/10.1126/science.aay4199.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import increasing_costs_index
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0., 2., 3., 4.],
    ...     epsilon=0, verbose=False
    ... )
    >>> round(increasing_costs_index(landscape, method="regression"), 3)
    -0.364
    """
    return _edge_fitness_trend(landscape, method, costs=True)
