from scipy.stats import binomtest
from itertools import combinations
import warnings

import numpy as np
import pandas as pd

from ._utils import _pythonize, _pack_rows
import logging

logger = logging.getLogger(__name__)


def _row_group_ids(M):
    """Dense 0-based id per distinct row of a small-int matrix: fast int64 mixed-
    radix packing, with a byte-view ``np.unique`` fallback for wide / high-
    cardinality inputs (so grouping always succeeds)."""
    if M.shape[1] == 0:
        return np.zeros(M.shape[0], dtype=np.intp), 1
    ids, n = _pack_rows(M)
    if ids is not None:
        return ids, n
    Mc = np.ascontiguousarray(M)
    void = Mc.view([("", Mc.dtype)] * Mc.shape[1]).reshape(-1)
    uniq, inv = np.unique(void, return_inverse=True)
    return np.asarray(inv).reshape(-1).astype(np.intp), int(uniq.size)


def _mutation_effects_for_position(X, f_arr, f_std, position, test_type):
    """Fitness effects of every allele-pair mutation at one ``position``, sharing a
    single background grouping (genetic background = all other positions).

    Vectorised equivalent of the original per-pair ``apply(tuple)`` + merge: each
    (allele, background) is a unique genotype, so per-allele fitness is gathered
    into a background-indexed array and a pair's effect is ``f_A - f_B`` over their
    shared backgrounds. median / mean / binomtest are order-invariant, so values
    match the original exactly. Returns a list of per-pair result dicts.
    """
    pos_vals = X[position].to_numpy()
    bg_cols = [c for c in X.columns if c != position]
    if bg_cols:
        codes = np.column_stack(
            [pd.factorize(X[c])[0] for c in bg_cols]
        ).astype(np.int64)
        bg_ids, n_bg = _row_group_ids(codes)
    else:
        bg_ids = np.zeros(len(X), dtype=np.intp)
        n_bg = 1

    unique_values = sorted(pd.Series(pos_vals).dropna().unique())
    fit_by_val = {}
    for v in unique_values:
        arr = np.full(n_bg, np.nan)
        rows = np.flatnonzero(pos_vals == v)
        arr[bg_ids[rows]] = f_arr[rows]
        fit_by_val[v] = arr

    results = []
    for A, B in combinations(unique_values, 2):
        fa = fit_by_val[A]
        fb = fit_by_val[B]
        mask = ~(np.isnan(fa) | np.isnan(fb))  # shared genetic backgrounds
        diff = fa[mask] - fb[mask]             # f_A - f_B (== original fitness_1 - fitness_2)
        n_trials = int(diff.size)
        if n_trials == 0:
            median_effect = np.nan
            mean_effect = np.nan
            p_value, significant = np.nan, False
        else:
            median_effect = float(np.median(np.abs(diff))) / f_std
            mean_effect = float(diff.mean())
            if test_type == "positive":
                successes = int(np.count_nonzero(diff > 0))
            else:  # "negative"; validated by the caller
                successes = int(np.count_nonzero(diff < 0))
            test_result = binomtest(successes, n_trials, p=0.5, alternative="greater")
            p_value = test_result.pvalue
            significant = test_result.pvalue < 0.05
        results.append(_pythonize({
            "mutation_from": A,
            "mutation_to": B,
            "median_abs_effect": median_effect,
            "mean_effect": mean_effect,
            "p_value": p_value,
            "significant": significant,
        }))
    return results


def evolvability_enhancing_mutations(landscape, epsilon=0, auto_calculate=True):
    """
    Calculates the proportion of edges where the higher-fitness node connects to
    a neighborhood with higher mean fitness than the lower-fitness node.

    This metric quantifies the prevalence of potentially evolvability-enhancing (EE)
    mutations in the landscape, as described in Wagner (2023). An edge represents
    an EE mutation if the delta_mean_neighbor_fit (difference in mean neighbor fitness
    between the connected nodes) exceeds the specified epsilon threshold.

    Parameters
    ----------
    landscape : BaseLandscape
        The fitness landscape object.
    epsilon : float, default=0
        Tolerance threshold for detecting significant differences in mean neighbor fitness.
        Only edges with delta_mean_neighbor_fit > epsilon are counted as EE mutations.
    auto_calculate : bool, default=True
        If True, automatically computes neighbour fitness (via the
        landscape's .neighbor_fitness property) if needed.
        If False, raises an exception when neighbor fitness metrics are missing.

    Returns
    -------
    float
        The proportion of edges with delta_mean_neighbor_fit > epsilon.

    Raises
    ------
    RuntimeError
        If auto_calculate=False and neighbor fitness metrics haven't been calculated.

    References
    ----------
    .. [1] Wagner, A. The role of evolvability in the evolution of
          complex traits. Nat Rev Genet 24, 1-16 (2023).
          https://doi.org/10.1038/s41576-023-00559-0
    """
    landscape._check_built()

    if "delta_mean_neighbor_fit" not in landscape.graph.es.attributes():
        if auto_calculate:
            if landscape.verbose:
                logger.info("Neighbor fitness metrics not found. Computing them...")
            landscape.neighbor_fitness  # lazily computes mean/delta neighbor fitness
        else:
            raise RuntimeError(
                "Neighbor fitness metrics haven't been calculated. "
                "Either access landscape.neighbor_fitness first "
                "or set auto_calculate=True."
            )

    delta_values = landscape.graph.es["delta_mean_neighbor_fit"]
    total_edges = landscape.graph.ecount()

    if total_edges == 0:
        warnings.warn("No edges found in the landscape graph.", RuntimeWarning)
        return 0.0

    ee_count = sum(1 for delta in delta_values if delta > epsilon)
    ee_proportion = ee_count / total_edges

    return _pythonize(ee_proportion)


def neutrality(landscape, threshold: float = 0.01) -> float:
    """
    Calculate the neutrality index of the landscape using an igraph-based graph.
    It assesses the proportion of neighbors with fitness values within a given threshold,
    indicating the presence of neutral areas in the landscape.

    When the landscape has a plateau layer (``epsilon > 0``), neutral neighbors
    stored during construction are included alongside the graph-based neighbors.
    This ensures that equal-fitness pairs — which have no directed edge — are
    still counted toward the neutrality metric.

    Parameters
    ----------
    landscape : object
        An object which contains an igraph.Graph in its 'graph' attribute. It is assumed
        that each vertex of the graph has a 'fitness' attribute.
    threshold : float, default=0.01
        The fitness difference threshold for neighbors to be considered neutral.

    Returns
    -------
    neutrality : float
        The neutrality index, ranging from 0 to 1. A higher value indicates more neutrality.
    """
    g = landscape.graph
    neutral_nn = getattr(landscape, '_neutral_neighbors', None) or {}
    fitness_values = g.vs["fitness"]
    neutral_pairs = 0
    total_pairs = 0

    for v in range(g.vcount()):
        fitness = fitness_values[v]

        # Directed-graph neighbors (improving + worsening edges)
        graph_neighbors = set(g.neighbors(v))

        # Include neutral neighbors from plateau layer if available
        all_neighbors = graph_neighbors | set(neutral_nn.get(v, []))

        for neighbor in all_neighbors:
            neighbor_fitness = fitness_values[neighbor]
            if abs(fitness - neighbor_fitness) <= threshold:
                neutral_pairs += 1
            total_pairs += 1

    neutrality_val = neutral_pairs / total_pairs if total_pairs > 0 else 0

    return _pythonize(neutrality_val)


def single_mutation_effects(
    landscape, position: str, test_type: str = "positive", n_jobs: int = 1
) -> pd.DataFrame:
    """
    Assess the fitness effects of all possible mutations at a single position across all genetic backgrounds.

    Parameters
    ----------
    landscape : Landscape
        The Landscape object containing the data and graph.

    position : str
        The name of the position (variable) to assess mutations for.

    test_type : str, default='positive'
        The type of significance test to perform. Must be 'positive' or 'negative'.

    n_jobs : int, default=1
        The number of parallel jobs to run.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing mutation pairs, median absolute fitness effect,
        p-values, and significance flags.
    """

    if test_type not in ("positive", "negative"):
        raise ValueError("test_type must be 'positive' or 'negative'")

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"]
    # Vectorised over a shared background grouping; the per-pair work is now cheap,
    # so this runs serially (the previous per-pair joblib fan-out was net-negative).
    results = _mutation_effects_for_position(
        X, f.to_numpy(), f.std(), position, test_type
    )
    return pd.DataFrame(results)


def all_mutation_effects(
    landscape, test_type: str = "positive", n_jobs: int = 1
) -> pd.DataFrame:
    """
    Assess the fitness effects of all possible mutations across all positions in the landscape.

    Parameters
    ----------
    landscape : Landscape
        The Landscape object containing the data and graph.

    test_type : str, default='positive'
        The type of significance test to perform. Must be 'positive' or 'negative'.

    n_jobs : int, default=1
        The number of parallel jobs to run.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing, for each position and mutation pair, the median absolute fitness effect,
        p-values, and significance flags.
    """

    if test_type not in ("positive", "negative"):
        raise ValueError("test_type must be 'positive' or 'negative'")

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"]
    f_arr = f.to_numpy()
    f_std = f.std()
    # Compute the shared data once and run positions serially: each position's
    # vectorised computation is cheap, and the old per-position joblib fan-out
    # pickled the whole landscape per worker (net-negative; see ANALYSIS bench).
    frames = [
        pd.DataFrame(
            _mutation_effects_for_position(X, f_arr, f_std, position, test_type)
        )
        for position in X.columns
    ]
    return pd.concat(frames, ignore_index=True)
