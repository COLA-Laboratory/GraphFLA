from scipy.stats import binomtest
from itertools import combinations
from joblib import Parallel, delayed

import numpy as np
import pandas as pd


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
