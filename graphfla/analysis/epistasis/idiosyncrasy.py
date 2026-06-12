"""Idiosyncratic epistasis indices.

Idiosyncratic index plus the diminishing-returns and increasing-costs
indices derived from per-background mutation effects.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from joblib import Parallel, delayed

from .._utils import _pythonize, _pack_rows


def _idiosyncratic_position_worker(Xcodes, f, std_baseline, j, min_pairs):
    """Idiosyncratic indices for ALL allele pairs at position ``j``, sharing ONE
    background grouping (computed once, not per pair).

    Gives values identical to :func:`idiosyncratic_index`'s core: keep-first dedup
    per background (a no-op since each (allele, background) is a unique genotype),
    analytic random-pair baseline, and NaN for too-few-background mutations so they
    are excluded from -- rather than dilute -- the landscape average. Returns the
    list of per-mutation indices for this position.
    """
    P = Xcodes.shape[1]
    other = np.delete(np.arange(P), j)
    col = Xcodes[:, j]
    alleles = np.unique(col)  # sorted codes -> same mutation set/order as before
    bg_ids, n_bg = _pack_rows(Xcodes[:, other])
    out = []
    if bg_ids is not None:
        # Per-allele fitness indexed by background id; one O(V) fill per allele.
        fit = []
        for a in alleles:
            arr = np.full(n_bg, np.nan)
            rows = np.flatnonzero(col == a)
            arr[bg_ids[rows]] = f[rows]
            fit.append(arr)
        for ai in range(len(alleles)):
            fa = fit[ai]
            for bi in range(ai + 1, len(alleles)):
                fb = fit[bi]
                mask = ~(np.isnan(fa) | np.isnan(fb))  # shared backgrounds
                if int(np.count_nonzero(mask)) < min_pairs:
                    out.append(np.nan)
                    continue
                eff = fb[mask] - fa[mask]
                out.append(float(np.std(eff) / std_baseline))
    else:
        # High-dim fallback: per-allele dict grouping, still shared across pairs.
        bgcols = Xcodes[:, other]
        dicts = []
        for a in alleles:
            d = {}
            for i in np.flatnonzero(col == a):
                k = bgcols[i].tobytes()
                if k not in d:
                    d[k] = f[i]
            dicts.append(d)
        for ai in range(len(alleles)):
            da = dicts[ai]
            for bi in range(ai + 1, len(alleles)):
                db = dicts[bi]
                common = da.keys() & db.keys()
                if len(common) < min_pairs:
                    out.append(np.nan)
                    continue
                eff = np.fromiter(
                    (db[k] - da[k] for k in common), dtype=float, count=len(common)
                )
                out.append(float(np.std(eff) / std_baseline))
    return out


def idiosyncratic_index(landscape, mutation, min_pairs: int = 3):
    """
    Calculates the idiosyncratic index for the fitness landscape proposed in [1].

    The idiosyncratic index of a specific genetic mutation quantifies the sensitivity
    of a specific mutation to idiosyncratic epistasis. It is defined as the
    variation in the fitness difference between genotypes that differ by the mutation,
    relative to the variation in the fitness difference between random genotype pairs.
    We compute this for the entire fitness landscape by averaging it across individual
    mutations.

    The index is typically in [0, 1] (0 = no idiosyncrasy); a mutation whose effect
    varies more across backgrounds than random genotype pairs can exceed 1.

    For more information, please refer to the original paper:

    [1] Daniel M. Lyons et al, "Idiosyncratic epistasis creates universals in mutational
    effects and evolutionary trajectories", Nat. Ecol. Evo., 2020.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    mutation : tuple(A, pos, B)
        A tuple containing:
        - A: The original variable value (allele) at the given position.
        - pos: The position in the configuration where the mutation occurs.
        - B: The new variable value (allele) after the mutation.

    min_pairs : int, default=3
        Minimum number of shared genetic backgrounds required to estimate the
        index. Mutations with fewer background-matched pairs yield an unstable
        effect-variance estimate and return NaN (so they are excluded from any
        landscape average rather than biasing it toward zero).

    Returns
    -------
    float
        The calculated idiosyncratic index, or NaN when it cannot be estimated
        (fewer than ``min_pairs`` shared backgrounds).
    """
    A, pos, B = mutation

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"]

    unique_alleles = X[pos].unique()
    if A not in unique_alleles:
        raise ValueError(
            f"Original allele '{A}' not found at position '{pos}'. Available: {unique_alleles}"
        )
    if B not in unique_alleles:
        raise ValueError(
            f"New allele '{B}' not found at position '{pos}'. Available: {unique_alleles}"
        )

    X_A = X[X[pos] == A]
    X_B = X[X[pos] == B]

    if X_A.empty or X_B.empty:
        warnings.warn(
            f"No genotypes found for allele '{A}' or '{B}' at position '{pos}'. Returning 0.0.",
            UserWarning,
        )
        return 0.0

    background_cols = [col for col in X.columns if col != pos]

    if not background_cols:
        X_A_backgrounds = pd.Series([tuple()] * len(X_A), index=X_A.index)
        X_B_backgrounds = pd.Series([tuple()] * len(X_B), index=X_B.index)
    else:
        X_A_backgrounds = X_A[background_cols].apply(tuple, axis=1)
        X_B_backgrounds = X_B[background_cols].apply(tuple, axis=1)

    df_A = pd.DataFrame({"background": X_A_backgrounds, "fitness_A": f.loc[X_A.index]})
    df_B = pd.DataFrame({"background": X_B_backgrounds, "fitness_B": f.loc[X_B.index]})

    df_A = df_A.drop_duplicates(subset="background", keep="first").set_index(
        "background"
    )
    df_B = df_B.drop_duplicates(subset="background", keep="first").set_index(
        "background"
    )

    df_merged = pd.merge(df_A, df_B, left_index=True, right_index=True, how="inner")

    if df_merged.empty:
        return _pythonize(np.nan)

    mutation_effects = df_merged["fitness_B"] - df_merged["fitness_A"]
    n_pairs = len(mutation_effects)

    if n_pairs < min_pairs:
        return _pythonize(np.nan)

    all_fitness_values = f.values
    if len(all_fitness_values) <= 1 or np.all(
        all_fitness_values == all_fitness_values[0]
    ):
        return 0.0

    # Random-pair baseline uses the exact closed form Var(diff) = 2*Var(f)
    # (i.i.d. draws): deterministic and avoids a near-zero denominator vs sampling.
    std_mutation_effect = np.std(mutation_effects)
    std_random_diff = np.sqrt(2.0) * np.std(all_fitness_values)

    idiosyncratic_val = std_mutation_effect / std_random_diff

    return _pythonize(idiosyncratic_val)


def global_idiosyncratic_index(landscape, n_jobs=-1, seed=None, min_pairs: int = 3):
    """
    Calculates the global idiosyncratic index for the entire fitness landscape using parallel processing.

    This function extends the individual mutation idiosyncratic index from Lyons et al. (2020)
    to provide a global measure by averaging across all possible mutations in the landscape.
    The global index quantifies the overall sensitivity of the landscape to idiosyncratic
    epistasis.

    The index is typically in [0, 1], with higher values indicating stronger idiosyncratic
    effects; individual mutations whose effects vary more than random genotype pairs can push
    the average above 1.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (all available cores).
    seed : int, optional
        Accepted for API consistency with other stochastic functions, but the
        index is computed deterministically (analytic random-pair baseline), so
        this has no effect on the result.
    min_pairs : int, default=3
        Minimum number of shared genetic backgrounds for a mutation to contribute
        (passed to :func:`idiosyncratic_index`).

    Returns
    -------
    float
        The overall idiosyncratic index (average across all mutations).

    References
    ----------
    .. [1] Daniel M. Lyons et al, "Idiosyncratic epistasis creates universals in mutational
       effects and evolutionary trajectories", Nat. Ecol. Evo., 2020.
    """
    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"].to_numpy(dtype=float)

    # Flat landscape: mirror idiosyncratic_index (every mutation 0.0) -> avg 0.0, not NaN.
    if len(f) <= 1 or np.all(f == f[0]):
        return _pythonize(0.0)
    std_baseline = float(np.sqrt(2.0) * np.std(f))

    # Sorted-allele codes (match original sorted iteration); memmapped to workers.
    Xcodes = np.column_stack(
        [pd.Categorical(X[c], categories=sorted(X[c].unique())).codes for c in X.columns]
    ).astype(np.int32)

    # Process-based (loky) parallelism over POSITIONS: each task computes all of a
    # position's allele-pair indices from a single shared background grouping
    # (vs. regrouping once per allele pair), then we flatten.
    per_pos = Parallel(n_jobs=n_jobs)(
        delayed(_idiosyncratic_position_worker)(Xcodes, f, std_baseline, j, min_pairs)
        for j in range(Xcodes.shape[1])
    )
    values = [v for sub in per_pos for v in sub]
    # Too-few-background mutations are NaN and excluded from the mean (padding
    # 0.0 would bias sparse landscapes toward zero idiosyncrasy).
    if not values or np.all(np.isnan(values)):
        return _pythonize(np.nan)
    return _pythonize(float(np.nanmean(values)))


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
    nodes_with_successors = int(np.count_nonzero(outdeg > 0))

    if nodes_with_successors < 2:
        warnings.warn(
            "Not enough nodes with successors to calculate correlation for diminishing returns.",
            UserWarning,
        )
        return np.nan

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
    nodes_with_predecessors = int(np.count_nonzero(indeg > 0))

    if nodes_with_predecessors < 2:
        warnings.warn(
            "Not enough nodes with predecessors to calculate correlation for increasing cost.",
            UserWarning,
        )
        return np.nan

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
