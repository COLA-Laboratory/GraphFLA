"""Literal per-mutation EE oracle; suitable for bounded basic tests."""

import numpy as np
from scipy.stats import ttest_ind_from_stats


def reference_statistics(
    configs, fitness, pairs, variance=None, author_bug=False, fdr=0.01
):
    """Literal per-mutation oracle, independent of production helpers."""
    adjacency = [set() for _ in fitness]
    for u, v in pairs:
        adjacency[u].add(v)
        adjacency[v].add(u)
    directed = list(pairs) + [(v, u) for u, v in pairs]
    rows = []
    for u, v in directed:
        positions = [p for p, (a, b) in enumerate(zip(configs[u], configs[v])) if a != b]
        assert len(positions) == 1
        p = positions[0]
        left = sorted(w for w in adjacency[u] if configs[w][p] == configs[u][p])
        right = sorted(w for w in adjacency[v] if configs[w][p] == configs[v][p])
        n = min(len(left), len(right))
        if not left or not right:
            rows.append((u, v, fitness[v]-fitness[u], np.nan, np.nan, n))
            continue
        m1, m2 = np.mean(fitness[left]), np.mean(fitness[right])
        if variance is None:
            v1, v2 = np.var(fitness[left]), np.var(fitness[right])
        else:
            v1 = sum(variance[w] for w in left) / len(left)**2
            v2 = sum(variance[w] for w in right) / len(right)**2
        rows.append((u, v, fitness[v]-fitness[u], m2-m1,
                     2*v2 if author_bug else v1+v2, n))
    rows = np.asarray(rows).reshape(-1, 6)
    dw, dm, variances, n = rows[:, 2:].T
    with np.errstate(divide="ignore", invalid="ignore"):
        p_effect = ttest_ind_from_stats(dm, np.sqrt(variances), n,
                                       dw, 0, 2, equal_var=False).pvalue
        p_zero = ttest_ind_from_stats(dm, np.sqrt(variances), n,
                                     0, 0, 2, equal_var=False).pvalue
    # Specify the limiting and untestable cases independently of SciPy's NaNs.
    for pval, diff in [(p_effect, dm-dw), (p_zero, dm)]:
        pval[n < 2] = np.nan
        mask = (n >= 2) & (variances == 0)
        pval[mask] = np.where(diff[mask] == 0, 1.0, 0.0)

    def decisions(pvalues):
        values = sorted(1.0 if not np.isfinite(p) else p for p in pvalues)
        cutoff = -1.0
        for rank, p in enumerate(values, 1):
            threshold = fdr * rank / len(values)
            passes = p < threshold if author_bug else p <= threshold
            if passes:
                cutoff = p
        return pvalues <= cutoff

    flag_effect = np.where(decisions(p_effect), np.sign(dm-dw), 0).astype(int)
    flag_zero = np.where(decisions(p_zero), np.sign(dm), 0).astype(int)
    return dict(source=rows[:, 0].astype(int), target=rows[:, 1].astype(int),
                effect=dw, delta_mean=dm, p_effect=p_effect, p_zero=p_zero,
                flag_effect=flag_effect, flag_zero=flag_zero)
