"""Mutation-effect variation (Lyons et al. 2020) and fitness-trend indices."""

import warnings
from itertools import permutations
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from joblib import Parallel, delayed

from .._utils import _pythonize, _pack_rows


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


def idiosyncratic_index(landscape, mutation, min_pairs: int = 3):
    r"""Estimate a mutation's idiosyncratic index from matched backgrounds.

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

    Returns
    -------
    index : float
        Ratio of observed-effect SD to sampled-control SD. Values can exceed 1.
        Returns NaN for a constant-fitness landscape, fewer than ``min_pairs``
        backgrounds, or a sampled control with zero SD. A well-defined value of
        zero indicates constant mutation effects across the observed backgrounds.

    Raises
    ------
    RuntimeError
        If the landscape has not been built.
    ValueError
        If ``min_pairs`` is invalid, the mutation uses an unknown position or
        allele, or source and target are equal. Also raised for missing
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
    For a substitution A to B, match configurations that agree at every other
    feature and calculate ``f(B, background) - f(A, background)``. With n such
    backgrounds, independently draw n pairs of genotypes with replacement from
    the retained population. Repeated genotypes and self-pairs are allowed.
    The index is

    .. math::

        I_{\mathrm{id}}(m) =
        \frac{\operatorname{SD}[\Delta f_m(b)]}
             {\operatorname{SD}[f(V_i)-f(U_i)]}.

    Both standard deviations use ``ddof=0``. The finite control is part of the
    estimator; replacing it with ``sqrt(2) * std(fitness)`` changes the statistic.
    Matching depends on configuration values, not on graph edges or their
    orientation. Any pair of observed alleles at the focal feature can be used.

    This function draws a fresh local random stream and does not change NumPy's
    global RNG state. Its current API has no seed parameter, so repeated calls
    can differ. Use the seeded global function for a reproducible landscape mean.

    No log transformation or measurement-error correction is applied. Removed
    genotypes cannot contribute to the control, including isolates pruned during
    construction. This index measures background dependence; it does not isolate
    residual interactions after fitting a global epistasis model.

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
    >>> value = idiosyncratic_index(landscape, (0, "bit_0", 1))
    >>> isinstance(value, float)
    True
    """
    _validate_min_pairs(min_pairs)
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
            return _idiosyncratic_ratio(effect_sd, n, f, np.random.RandomState())
    return float("nan")


def global_idiosyncratic_index(landscape, n_jobs=-1, seed=None, min_pairs: int = 3):
    r"""Estimate the mean idiosyncratic index across directed mutations.

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
    n_jobs : int, default=-1
        Number of parallel jobs for background matching. -1 uses all available
        cores; 1 runs serially. This does not change a fixed-seed result.
    seed : int, default=None
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
    RuntimeError
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
    Enumerate each position and ordered pair of observed alleles. For each
    mutation, match configurations at all other positions, calculate its effects,
    and divide their population SD by that of an equally sized random control.
    Both control endpoints are drawn independently with replacement from the
    retained fitness population. Both SDs use ``ddof=0``. The landscape mean is

    .. math::

        \overline{I}_{\mathrm{id}} =
        \frac{1}{|\mathcal{M}|}\sum_{m\in\mathcal{M}} I_{\mathrm{id}}(m),

    where M contains the mutations meeting ``min_pairs``. A mean of position
    means or a mean weighted by background count would be a different summary.
    Forward and reverse mutations have the same observed SD but receive
    independent controls here.

    Random draws follow configuration-column order, then sorted source and
    target allele order, after parallel matching. The published tRNA notebook
    reinitializes its seed for each background count; that study-specific seed
    policy is not used by this function. Equal seeds only reproduce equal input
    ordering and the same sampling procedure.

    Graph adjacency does not define the matched pairs. Any two observed alleles
    at one feature are considered, including nonadjacent ordinal values. However,
    genotypes pruned during graph construction remain unavailable to both effects
    and controls. Reproducing a study requires its full reference population,
    including isolated genotypes when the study includes them.

    Fitness is used as supplied. The result summarizes background dependence,
    which can also arise from a nonlinear global fitness map; it is not a test
    separating global epistasis from specific interactions. The function returns
    a scalar, without per-position summaries or an uncertainty estimate.

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
    """Measures diminishing returns epistasis in a fitness landscape.

    Diminishing returns epistasis occurs when the fitness benefit of new
    beneficial mutations decreases as the background fitness increases. This
    function quantifies this trend by calculating the correlation between the
    fitness of each genotype (node) and the average fitness improvement
    provided by its direct successors (fitter one-mutant neighbors). A
    significant negative correlation indicates diminishing returns.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object. The landscape graph
        must have a 'fitness' attribute for each node.
    method : {'pearson', 'spearman', 'regression'}, default='pearson'
        The method used to calculate the diminishing returns index.
        'pearson' for Pearson correlation coefficient,
        'spearman' for Spearman rank correlation coefficient,
        'regression' for the slope of a linear regression.

    Returns
    -------
    correlation_or_slope : float
        For 'pearson' or 'spearman': The correlation coefficient between node fitness
        and average successor fitness improvement.
        For 'regression': The slope of the linear regression.
        Returns NaN if calculation is not possible.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If the graph is missing or the 'fitness' attribute is not found.
        If the correlation method is invalid.
    """
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    # Mean improvement toward the optimum across each node's improving out-edges.
    # `delta_fit` is |Δfitness| -- the positive improvement magnitude on every
    # improving edge (both maximize and minimize) -- so the per-node mean is simply
    # the delta_fit-weighted out-strength / out-degree: one C-level pass, no
    # edge-list materialisation (fast and memory-light). Fallback recomputes from
    # fitness via the sparse adjacency when delta_fit is absent. NaN = local optima.
    fitness = np.asarray(landscape.graph.vs["fitness"], dtype=float)
    node_fitnesses = fitness
    outdeg = np.asarray(landscape.graph.outdegree(), dtype=float)
    nodes_with_successors = int(np.count_nonzero(outdeg > 0))

    # Checked before the improvements are computed: with fewer than two such
    # nodes the correlation is undefined anyway, and the sparse-adjacency
    # fallback below cannot be built on an edgeless graph (fully neutral input).
    if nodes_with_successors < 2:
        warnings.warn(
            "Not enough nodes with successors to calculate correlation for diminishing returns.",
            UserWarning,
        )
        return np.nan

    if "delta_fit" in landscape.graph.es.attributes():
        per_node = np.asarray(
            landscape.graph.strength(mode="out", weights="delta_fit"), dtype=float
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_successor_improvement = np.where(outdeg > 0, per_node / outdeg, np.nan)
    else:
        mean_succ_fit = landscape.graph.get_adjacency_sparse().dot(fitness)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_succ_fit = np.where(outdeg > 0, mean_succ_fit / outdeg, np.nan)
        avg_successor_improvement = (
            mean_succ_fit - fitness if landscape.maximize
            else fitness - mean_succ_fit
        )

    node_fitnesses_series = pd.Series(node_fitnesses)
    avg_improvement_series = pd.Series(avg_successor_improvement)

    mask = ~avg_improvement_series.isna()
    if mask.sum() < 2:
        warnings.warn(
            "Not enough valid data points after NaN omission to calculate correlation.",
            UserWarning,
        )
        return np.nan
    node_fitnesses = node_fitnesses_series[mask]
    avg_improvement = avg_improvement_series[mask]

    if method == "pearson":
        corr_func = pearsonr
    elif method == "spearman":
        corr_func = spearmanr
    elif method == "regression":
        try:
            X = np.array(node_fitnesses).reshape(-1, 1)
            y = np.array(avg_improvement)

            X_with_const = np.column_stack((np.ones(X.shape[0]), X))  # add intercept

            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            slope = beta[1]

            n = len(X)
            if n <= 2:
                return slope

            y_pred = X_with_const.dot(beta)
            residual_SS = np.sum((y - y_pred) ** 2)
            X_mean = np.mean(X)
            X_var = np.sum((X.reshape(-1) - X_mean) ** 2)

            if X_var == 0:
                return slope

            return slope
        except Exception as e:
            warnings.warn(f"Could not calculate regression: {e}", UserWarning)
            return np.nan
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'regression'")

    try:
        correlation, _ = corr_func(node_fitnesses, avg_improvement)
        return _pythonize(correlation)
    except Exception as e:
        warnings.warn(f"Could not calculate correlation: {e}", UserWarning)
        return np.nan


def increasing_costs_index(
    landscape,
    method: Literal["pearson", "spearman", "regression"] = "pearson",
) -> float:
    """Measures increasing cost epistasis in a fitness landscape.

    Increasing cost epistasis occurs when the fitness cost (reduction) of
    deleterious mutations increases as the background fitness increases. This
    function quantifies this trend by calculating the correlation between the
    fitness of each genotype (node) and the average fitness cost incurred
    by mutations leading *to* that node from its direct predecessors (less fit
    one-mutant neighbors). A significant positive correlation indicates
    increasing cost.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object. The landscape graph
        must have a 'fitness' attribute for each node.
    method : {'pearson', 'spearman', 'regression'}, default='pearson'
        The method used to calculate the increasing costs index.
        'pearson' for Pearson correlation coefficient,
        'spearman' for Spearman rank correlation coefficient,
        'regression' for the slope of a linear regression.

    Returns
    -------
    correlation_or_slope : float
        For 'pearson' or 'spearman': The correlation coefficient between node fitness
        and average predecessor fitness cost.
        For 'regression': The slope of the linear regression.
        Returns NaN if calculation is not possible.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If the graph is missing or the 'fitness' attribute is not found.
        If the correlation method is invalid.
    """
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found."
            " Landscape must be built first."
        )

    # Mirror of diminishing_returns_index over IN-edges: mean cost across each
    # node's improving predecessors. delta_fit is the positive cost magnitude on
    # every improving edge, so the per-node mean is the delta_fit-weighted
    # in-strength / in-degree (fast, memory-light). Fallback via the transposed
    # sparse adjacency when delta_fit is absent. NaN for source nodes.
    fitness = np.asarray(landscape.graph.vs["fitness"], dtype=float)
    node_fitnesses = fitness
    indeg = np.asarray(landscape.graph.indegree(), dtype=float)
    nodes_with_predecessors = int(np.count_nonzero(indeg > 0))

    # Checked before the costs are computed, for the same reason as in
    # ``diminishing_returns_index``.
    if nodes_with_predecessors < 2:
        warnings.warn(
            "Not enough nodes with predecessors to calculate correlation for increasing cost.",
            UserWarning,
        )
        return np.nan

    if "delta_fit" in landscape.graph.es.attributes():
        per_node = np.asarray(
            landscape.graph.strength(mode="in", weights="delta_fit"), dtype=float
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_predecessor_cost = np.where(indeg > 0, per_node / indeg, np.nan)
    else:
        mean_pred_fit = landscape.graph.get_adjacency_sparse().T.dot(fitness)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_pred_fit = np.where(indeg > 0, mean_pred_fit / indeg, np.nan)
        avg_predecessor_cost = (
            fitness - mean_pred_fit if landscape.maximize
            else mean_pred_fit - fitness
        )

    node_fitnesses_series = pd.Series(node_fitnesses)
    avg_cost_series = pd.Series(avg_predecessor_cost)

    mask = ~avg_cost_series.isna()
    if mask.sum() < 2:
        warnings.warn(
            "Not enough valid data points after NaN omission to calculate correlation.",
            UserWarning,
        )
        return np.nan
    node_fitnesses = node_fitnesses_series[mask]
    avg_cost = avg_cost_series[mask]

    if method == "pearson":
        corr_func = pearsonr
    elif method == "spearman":
        corr_func = spearmanr
    elif method == "regression":
        try:
            X = np.array(node_fitnesses).reshape(-1, 1)
            y = np.array(avg_cost)

            X_with_const = np.column_stack((np.ones(X.shape[0]), X))  # add intercept

            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            slope = beta[1]

            return slope
        except Exception as e:
            warnings.warn(f"Could not calculate regression: {e}", UserWarning)
            return np.nan
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'regression'")

    try:
        correlation, _ = corr_func(node_fitnesses, avg_cost)
        return _pythonize(correlation)
    except Exception as e:
        warnings.warn(f"Could not calculate correlation: {e}", UserWarning)
        return np.nan
