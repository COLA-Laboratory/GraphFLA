"""Statistical classification of one-site evolvability-enhancing mutations."""

import numpy as np
import pandas as pd
from numbers import Real
from scipy.stats import t

_MOMENT_CELLS = 131072


def _nonfocal_moments(configs, f, pairs, fitness_variance):
    """Compute node/site moments with bounded, degree-bucketed neighbor blocks.

    Use centered squared deviations, not total-minus-focal second moments:
    the latter loses small non-focal variance next to large focal effects.
    Graph indices use O(N + E) storage, without a dense node/site cube.
    Blocks target _MOMENT_CELLS entries; one wide row may exceed that target.
    """
    n_edges = len(pairs)
    positions = np.empty(n_edges, dtype=np.intp)
    step = max(1, _MOMENT_CELLS // max(1, configs.shape[1]))
    for start in range(0, n_edges, step):
        block = pairs[start:start + step]
        changed = configs[block[:, 0]] != configs[block[:, 1]]
        if np.any(changed.sum(axis=1) != 1):
            raise ValueError("EE mutations require one-site neighbor pairs.")
        positions[start:start + len(block)] = changed.argmax(axis=1)
    source = np.concatenate((pairs[:, 0], pairs[:, 1]))
    target = np.concatenate((pairs[:, 1], pairs[:, 0]))
    position = np.tile(positions, 2)
    if not n_edges:
        empty = np.empty((0, 3))
        return source, target, position, empty, empty.copy()

    # CSR adjacency in target-ID order, independent of input edge orientation.
    order = np.lexsort((target, source))
    neighbors, sites = target[order], position[order]
    degree = np.bincount(source, minlength=len(f))
    offsets = np.concatenate(([0], np.cumsum(degree)))
    keys, inverse = np.unique(source * configs.shape[1] + position, return_inverse=True)
    nodes, focal = np.divmod(keys, configs.shape[1])
    sizes = degree[nodes]
    buckets = np.floor(np.log2(sizes)).astype(int)
    moments = np.empty((len(keys), 3))
    for bucket in np.unique(buckets):
        group = np.flatnonzero(buckets == bucket)
        width = int(sizes[group].max())
        rows_per_block = max(1, _MOMENT_CELLS // width)
        columns = np.arange(width)
        for start in range(0, len(group), rows_per_block):
            rows = group[start:start + rows_per_block]
            indices = offsets[nodes[rows], None] + columns
            valid = columns < sizes[rows, None]
            indices = np.minimum(indices, len(neighbors) - 1)
            keep = valid & (sites[indices] != focal[rows, None])
            count = keep.sum(axis=1)
            values = f[neighbors[indices]]
            mean = np.divide(np.where(keep, values, 0).sum(axis=1), count,
                             out=np.full(len(rows), np.nan), where=count > 0)
            if fitness_variance is None:
                squared = (values - mean[:, None])**2
                numerator = np.where(keep, squared, 0).sum(axis=1)
                denominator = count
            else:
                numerator = np.where(keep, fitness_variance[neighbors[indices]], 0).sum(axis=1)
                denominator = count**2
            variance = np.divide(numerator, denominator, out=np.full(len(rows), np.nan),
                                 where=count > 0)
            moments[rows, 0], moments[rows, 1], moments[rows, 2] = mean, variance, count
    left = moments[inverse]
    right = np.concatenate((left[n_edges:], left[:n_edges]))
    return source, target, position, left, right


def _validate_fdr(fdr):
    if (not isinstance(fdr, Real) or isinstance(fdr, (bool, np.bool_))
            or not np.isfinite(fdr) or not 0 < fdr < 1):
        raise ValueError("fdr must be a finite real number strictly between 0 and 1.")


def _bh_adjusted_pvalues(pvalues):
    """BH adjusted p-values, keeping untestable hypotheses in the family."""
    pvalues = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(pvalues)
    values = np.where(valid, pvalues, 1.0)
    order = np.argsort(values, kind="stable")
    if not len(values):
        return values
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    adjusted[~valid] = np.nan
    return adjusted


def _benjamini_hochberg(pvalues, fdr=0.01):
    """BH step-up decisions; untestable hypotheses remain in the family."""
    _validate_fdr(fdr)
    return _bh_adjusted_pvalues(pvalues) <= fdr


def _ee_pvalues(difference, variance, n):
    """Two-sided one-sample t test, n=min(neighborhood sizes), df=n-1."""
    pvalues = np.full(len(difference), np.nan)
    regular = (n >= 2) & (variance > 0)
    statistic = difference[regular] / np.sqrt(variance[regular] / n[regular])
    pvalues[regular] = 2 * t.sf(np.abs(statistic), n[regular] - 1)
    constant = (n >= 2) & (variance == 0)
    # A point mass equal to the null supplies no evidence against it; a
    # different point mass is the zero-variance limit of the t statistic.
    pvalues[constant] = np.where(difference[constant] == 0, 1.0, 0.0)
    return pvalues


def _ee_statistics(
    configs, fitness, pairs, *, fitness_variance=None, epsilon=0, fdr=0.01
):
    """Classify both orientations of represented, undirected one-site pairs.

    By default, use the population variance of each neighborhood's fitness
    values (ddof=0). Explicit per-genotype measurement variances instead
    propagate as sum(var)/k**2, where k is the neighborhood size.
    """
    _validate_fdr(fdr)
    configs = np.asarray(configs)
    fitness = np.asarray(fitness, dtype=float)
    if configs.ndim != 2 or len(configs) != len(fitness):
        raise ValueError("Configurations must align with the fitness values.")
    if not np.isfinite(fitness).all():
        raise ValueError("Fitness values must be finite.")
    if fitness_variance is not None:
        fitness_variance = np.asarray(fitness_variance, dtype=float)
        if (fitness_variance.shape != fitness.shape
                or not np.isfinite(fitness_variance).all()
                or np.any(fitness_variance < 0)):
            raise ValueError("Fitness variances must be finite, nonnegative and aligned.")
    # Center before taking means so a large arbitrary offset does not enter
    # the neighborhood subtraction or the roundoff guard.
    f = fitness - fitness[0] if len(fitness) else fitness.copy()
    pairs = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    source, target, position, left, right = _nonfocal_moments(
        configs, f, pairs, fitness_variance
    )
    delta_fitness = f[target] - f[source]
    delta_mean = right[:, 0] - left[:, 0]
    variance = left[:, 1] + right[:, 1]
    n = np.minimum(left[:, 2], right[:, 2])
    p_effect = _ee_pvalues(delta_mean - delta_fitness, variance, n)
    p_zero = _ee_pvalues(delta_mean, variance, n)
    q_effect = _bh_adjusted_pvalues(p_effect)
    q_zero = _bh_adjusted_pvalues(p_zero)
    reject_effect = q_effect <= fdr
    reject_zero = q_zero <= fdr
    relevant_p = np.where(delta_fitness > 0, p_effect, p_zero)
    reject = np.where(delta_fitness > 0, reject_effect, reject_zero)
    scale = np.maximum.reduce([
        np.abs(left[:, 0]), np.abs(right[:, 0]),
        np.abs(f[source]), np.abs(f[target]),
    ])
    roundoff = 8 * np.finfo(float).eps * scale
    excess = delta_mean - np.maximum(0, delta_fitness)
    ee = reject & (excess > epsilon + roundoff)
    return pd.DataFrame({
        "source": source, "target": target, "position": position,
        "delta_fitness": delta_fitness, "delta_neighbor_fitness": delta_mean,
        "mean_source": left[:, 0], "mean_target": right[:, 0],
        "variance_source": left[:, 1], "variance_target": right[:, 1],
        "n_source": left[:, 2].astype(int), "n_target": right[:, 2].astype(int),
        "p_effect": p_effect, "p_zero": p_zero,
        "q_effect": q_effect, "q_zero": q_zero, "excess": excess,
        "reject_effect": reject_effect, "reject_zero": reject_zero,
        "testable": np.isfinite(relevant_p), "ee": ee,
    })


def _landscape_ee_statistics(landscape, epsilon=0, *, fdr=0.01):
    """Use graph relations, including retained neutral adjacency, exactly once."""
    data = landscape.get_data()
    if landscape.data_types is None:
        raise ValueError("Configuration columns are required to exclude the focal site.")
    X = data[list(landscape.data_types)]
    if X.isna().any().any() or X.duplicated().any():
        raise ValueError("Configurations must be unique and contain no missing values.")
    configs = X.to_numpy()
    f = data["fitness"].to_numpy(dtype=float)
    if not landscape.maximize:
        f = -f
    pairs = {tuple(sorted(edge)) for edge in landscape.graph.get_edgelist()}
    for u, neighbors in (getattr(landscape, "_neutral_neighbors", None) or {}).items():
        pairs.update(tuple(sorted((u, v))) for v in neighbors)
    statistics = _ee_statistics(configs, f, sorted(pairs), epsilon=epsilon, fdr=fdr)
    source = statistics["source"].to_numpy()
    target = statistics["target"].to_numpy()
    positions = statistics["position"].to_numpy()
    statistics["source_allele"] = configs[source, positions]
    statistics["target_allele"] = configs[target, positions]
    statistics["position"] = pd.Series(X.columns.take(positions), dtype=object)
    return statistics
