"""Statistical classification of one-site evolvability-enhancing mutations."""

import numpy as np
import pandas as pd
from scipy.stats import t


def _benjamini_hochberg(pvalues, fdr=0.01):
    """BH step-up decisions; untestable hypotheses remain in the family."""
    pvalues = np.asarray(pvalues, dtype=float)
    ordered = np.sort(np.where(np.isfinite(pvalues), pvalues, 1.0))
    passing = ordered <= fdr * np.arange(1, len(ordered) + 1) / max(1, len(ordered))
    if not passing.any():
        return np.zeros(len(pvalues), dtype=bool)
    return np.isfinite(pvalues) & (pvalues <= ordered[passing][-1])


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


def _ee_statistics(configs, fitness, pairs, *, fitness_variance=None, epsilon=0):
    """Classify both orientations of represented, undirected one-site pairs.

    By default, use the population variance of each neighborhood's fitness
    values (ddof=0). Explicit per-genotype measurement variances instead
    propagate as sum(var)/k**2, where k is the neighborhood size.
    """
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
    adjacent = [[] for _ in f]
    positions = []
    for u, v in pairs:
        changed = np.flatnonzero(configs[u] != configs[v])
        if len(changed) != 1:
            raise ValueError("EE mutations require one-site neighbor pairs.")
        pos = int(changed[0])
        positions.append(pos)
        adjacent[u].append((v, pos))
        adjacent[v].append((u, pos))

    moments = {}
    for u, neighbors in enumerate(adjacent):
        for pos in {pos for _, pos in neighbors}:
            ids = [v for v, other in neighbors if other != pos]
            k = len(ids)
            if not k:
                moments[u, pos] = (np.nan, np.nan, 0)
                continue
            values = f[ids]
            variance = (float(np.var(values)) if fitness_variance is None
                        else float(np.sum(fitness_variance[ids]) / k**2))
            moments[u, pos] = (float(np.mean(values)), variance, k)

    source = np.concatenate((pairs[:, 0], pairs[:, 1]))
    target = np.concatenate((pairs[:, 1], pairs[:, 0]))
    position = np.tile(positions, 2).astype(np.intp)
    left = np.asarray([moments[u, p] for u, p in zip(source, position)]).reshape(-1, 3)
    right = np.asarray([moments[v, p] for v, p in zip(target, position)]).reshape(-1, 3)
    delta_fitness = f[target] - f[source]
    delta_mean = right[:, 0] - left[:, 0]
    variance = left[:, 1] + right[:, 1]
    n = np.minimum(left[:, 2], right[:, 2])
    p_effect = _ee_pvalues(delta_mean - delta_fitness, variance, n)
    p_zero = _ee_pvalues(delta_mean, variance, n)
    reject_effect = _benjamini_hochberg(p_effect)
    reject_zero = _benjamini_hochberg(p_zero)
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
        "reject_effect": reject_effect, "reject_zero": reject_zero,
        "testable": np.isfinite(relevant_p), "ee": ee,
    })


def _landscape_ee_statistics(landscape, epsilon):
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
    return _ee_statistics(configs, f, sorted(pairs), epsilon=epsilon)
