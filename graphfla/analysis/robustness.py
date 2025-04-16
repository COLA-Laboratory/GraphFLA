from scipy.stats import binomtest
from itertools import combinations
from joblib import Parallel, delayed

import numpy as np
import pandas as pd


def calculate_evol_enhance(landscape, epsilon=0, auto_calculate=True):
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
        If True, automatically runs calculate_neighbor_fitness() if needed.
        If False, raises an exception when neighbor fitness metrics are missing.

    Returns
    -------
    dict
        A dictionary containing:
        - 'ee_proportion': The proportion of edges with delta_mean_neighbor_fit > epsilon
        - 'ee_count': The count of edges with delta_mean_neighbor_fit > epsilon
        - 'total_edges': The total number of edges in the landscape

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

    # Check if neighbor fitness has been calculated
    if "delta_mean_neighbor_fit" not in landscape.graph.es.attributes():
        if auto_calculate:
            if landscape.verbose:
                print(
                    "Neighbor fitness metrics not found. Running calculate_neighbor_fitness()..."
                )
            landscape.calculate_neighbor_fitness()
        else:
            raise RuntimeError(
                "Neighbor fitness metrics haven't been calculated. "
                "Either call landscape.calculate_neighbor_fitness() first "
                "or set auto_calculate=True."
            )

    # Get all delta_mean_neighbor_fit values
    delta_values = landscape.graph.es["delta_mean_neighbor_fit"]
    total_edges = landscape.graph.ecount()

    if total_edges == 0:
        if landscape.verbose:
            print("Warning: No edges found in the landscape graph.")
        return {"ee_proportion": 0.0, "ee_count": 0, "total_edges": 0}

    # Count edges where delta_mean_neighbor_fit > epsilon
    ee_count = sum(1 for delta in delta_values if delta > epsilon)
    ee_proportion = ee_count / total_edges

    return {
        "ee_proportion": ee_proportion,
        "ee_count": ee_count,
        "total_edges": total_edges,
    }


def neutrality(landscape, threshold: float = 0.01) -> float:
    """
    Calculate the neutrality index of the landscape using an igraph-based graph.
    It assesses the proportion of neighbors with fitness values within a given threshold,
    indicating the presence of neutral areas in the landscape.

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
    # Get the igraph graph object from the landscape.
    g = landscape.graph
    neutral_pairs = 0
    total_pairs = 0

    # Iterate over each vertex by its index.
    for v in range(g.vcount()):
        fitness = g.vs[v]["fitness"]  # Retrieve the fitness of the current vertex.

        # Iterate over all neighbors of the current vertex.
        for neighbor in g.neighbors(v):
            neighbor_fitness = g.vs[neighbor]["fitness"]
            # Count the pair as neutral if the fitness difference is within the threshold.
            if abs(fitness - neighbor_fitness) <= threshold:
                neutral_pairs += 1
            total_pairs += 1

    # Compute neutrality as the ratio of neutral neighbor pairs to the total number of pairs.
    neutrality = neutral_pairs / total_pairs if total_pairs > 0 else 0

    return neutrality


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

    def test_significance(series, test_type="positive"):
        if test_type == "positive":
            successes = (series > 0).sum()
            hypothesized_prob = 0.5
            alternative = "greater"
        elif test_type == "negative":
            successes = (series < 0).sum()
            hypothesized_prob = 0.5
            alternative = "greater"
        else:
            raise ValueError("test_type must be 'positive' or 'negative'")

        n_trials = len(series)
        if n_trials == 0:
            return np.nan, False

        test_result = binomtest(
            successes, n_trials, p=hypothesized_prob, alternative=alternative
        )
        significant = test_result.pvalue < 0.05

        return test_result.pvalue, significant

    def compute_mutation_effect(X, f, position, A, B, test_type):
        X1 = X[X[position] == A]
        X2 = X[X[position] == B]

        X1 = pd.Series(X1.drop(columns=[position]).apply(tuple, axis=1))
        X2 = pd.Series(X2.drop(columns=[position]).apply(tuple, axis=1))

        df1 = pd.concat([X1, f], axis=1, join="inner")
        df2 = pd.concat([X2, f], axis=1, join="inner")
        df1.set_index(0, inplace=True)
        df2.set_index(0, inplace=True)

        df_diff = pd.merge(
            df1, df2, left_index=True, right_index=True, suffixes=("_1", "_2")
        )
        df_diff.index = range(len(df_diff))
        diff = df_diff["fitness_1"] - df_diff["fitness_2"]

        median_effect = abs(diff).median() / f.std()
        p_value, significant = test_significance(diff, test_type)

        return {
            "mutation_from": A,
            "mutation_to": B,
            "median_abs_effect": median_effect,
            "mean_effect": diff.mean(),
            "p_value": p_value,
            "significant": significant,
        }

    data = landscape.get_data()
    X = data.iloc[:, : len(landscape.data_types)]
    f = data["fitness"]

    unique_values = X[position].dropna().unique()
    unique_values = sorted(unique_values)

    mutation_pairs = list(combinations(unique_values, 2))

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_mutation_effect)(X, f, position, A, B, test_type)
        for A, B in mutation_pairs
    )

    mutation_effects_df = pd.DataFrame(results)

    return mutation_effects_df


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

    def assess_position(position, test_type):
        return single_mutation_effects(
            landscape=landscape, position=position, test_type=test_type, n_jobs=1
        )

    data = landscape.get_data()
    X = data.iloc[:, : len(landscape.data_types)]

    positions = list(X.columns)

    all_mutation_effects = Parallel(n_jobs=n_jobs)(
        delayed(assess_position)(position, test_type) for position in positions
    )

    all_mutation_effects_df = pd.concat(all_mutation_effects, ignore_index=True)

    return all_mutation_effects_df
