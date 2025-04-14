from .distances import mixed_distance
from .algorithms import hill_climb, random_walk
from .utils import is_ancestor_fast
from scipy.stats import spearmanr, pearsonr, binomtest, ttest_1samp
from typing import Any, Tuple
from itertools import combinations, product
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from typing import Union

import numpy as np
import pandas as pd
import networkx as nx
import random
import warnings

### Future Work:
### 1. (YES) Distribution of Fitness Effects (DFE)
### 2. (YES) Mutational Robustness
### 3. (NO) Evolvability
###   - Mutational robustness (e.g., average fitness effect of neighbors, neutral neighborhood size) (Payne & Wagner 2014, Lauring et al.).
###   - How the DFE changes across genetic backgrounds (Johnson et al.).
###   - Evolvability-reducing (ER) mutations.
###   - Analysis of neutral networks and their properties (Lauring et al., Greenbury et al.).
### 4. Advanced Epistasis
###   - (YES) Diminishing Returns.
###   - (NO) Increasing Costs.
###   - (NO) Global epistasis.
###   - (NO) Idiosyncratic epistasis.
###   - (NO) Additive, multiplicative, and chimeric epistasis.
### 5. (NO) Higher-Order Epistasis
### 6. (YES) Evolutionary Accessibility & Predictability
### 7. (NO) Environment-Dependent Fitness Effects and Epistasis (GxE, GxGxE), this may include:
###   - Calculating and analyzing GxE and GxGxE interactions systematically. (Costanzo et al., Bank 2022).
###   - Implementing methods for differential interaction analysis (identifying modified, masked, novel interactions) as defined by Costanzo et al.
###   - Tools for modeling and predicting landscape changes across environmental gradients (Chen et al. 2022, Bank 2022), potentially using transformation models like those suggested by Li & Zhang (2018).
###   - Support for multi-objective optimization concepts like Pareto fronts to analyze trade-offs between different functions/phenotypes (Shoval et al.).
### 8. (YES) Landscape Modeling
### 9. Simulation
###   - (YES) Adaptive walk models (reedy, stochastic based on fixation probabilities like Kimura's, random walks on neutral networks).
###   - (YES) Population genetics models.
###   - (NO) Tools for predicting evolutionary trajectories and outcomes based on landscape topography (Lässig et al., de Visser & Krug 2014, Bank 2022).


def global_optima_accessibility(
    landscape, approximate: bool = False, n_samples: Union[int, float, None] = 0.2
) -> float:
    """
    Calculate or estimate the accessibility of the global optimum (GO).

    This metric represents the fraction of configurations in the landscape
    that can reach the global optimum via any monotonic, fitness-improving path.

    By default (`approximate=False`), it relies on the 'size_basin_first'
    attribute calculated by `determine_accessible_paths()`. If this
    attribute is not available (i.e., `determine_accessible_paths()` has
    not been run or `landscape._path_calculated` is False), this method will
    raise a RuntimeError unless `approximate` is set to True.

    If `approximate=True`, the accessibility is estimated by sampling
    a specified number or fraction (`n_samples`) of configurations and
    checking if the GO is reachable from each sample using graph traversal
    (specifically, checking if the GO is a successor via directed paths).

    Parameters
    ----------
    approximate : bool, default=False
        If True, estimate accessibility by sampling and graph traversal.
        If False, use the pre-calculated 'size_basin_first' attribute for the GO.
    n_samples : int or float, optional
        Specifies the number or fraction of configurations to sample for
        approximation. Required and must be set if `approximate=True`.
        - If int: The absolute number of configurations to sample (must be > 0).
        - If float: The fraction of total configurations to sample (must be in (0, 1]).
        Ignored if `approximate=False`.

    Returns
    -------
    float
        The fraction of configurations estimated or known to be able to
        reach the global optimum monotonically (value between 0.0 and 1.0).

    Raises
    ------
    RuntimeError
        If `approximate=False` and `determine_accessible_paths()` has not
        been successfully run (i.e., `landscape._path_calculated` is False).
        If the global optimum has not been determined.
        If the graph is not initialized.
    ValueError
        If `approximate=True` and `n_samples` is None or invalid (e.g., int <= 0,
        float not in (0, 1], or wrong type).
    """
    if landscape.graph is None:
        raise RuntimeError("Graph not initialized. Cannot calculate accessibility.")
    if landscape.go_index is None:
        # Attempt to determine GO if not already done
        landscape._determine_global_optimum()
        if landscape.go_index is None:  # Check again after attempting
            raise RuntimeError(
                "Global optimum could not be determined. Cannot calculate accessibility."
            )

    if landscape.n_configs is None or landscape.n_configs == 0:
        warnings.warn(
            "Landscape has 0 configurations. Accessibility is 0.", RuntimeWarning
        )
        return 0.0

    # --- Exact Calculation ---
    if not approximate:
        if not landscape._path_calculated:
            raise RuntimeError(
                "Exact global optima accessibility requires 'size_basin_first' data. "
                "Please run `landscape.determine_accessible_paths()` first, or set `approximate=True`."
            )

        try:
            # Get the pre-calculated size of the basin leading to the GO.
            # The count from determine_accessible_paths already includes the GO node itlandscape.
            go_node_data = landscape.graph.nodes[landscape.go_index]
            size_basin_go = go_node_data.get("size_basin_first", 0)

            if "size_basin_first" not in go_node_data and landscape.verbose:
                # This case indicates an issue if _path_calculated is True
                warnings.warn(
                    "Global optimum node missing 'size_basin_first' attribute "
                    "despite _path_calculated=True. Calculation might be inaccurate. Returning 0.",
                    RuntimeWarning,
                )
                return 0.0
            elif size_basin_go == 0 and landscape.verbose:
                warnings.warn(
                    "Calculated 'size_basin_first' for GO is 0.", RuntimeWarning
                )

            accessibility = size_basin_go / landscape.n_configs
            return accessibility

        except KeyError:
            # Should not happen if _path_calculated is True and GO index is valid
            raise RuntimeError(
                "Internal inconsistency: Global optimum index not found in graph nodes after check. "
                "Re-run `determine_accessible_paths()` or use `approximate=True`."
            )
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred accessing exact accessibility: {e}"
            )

    # --- Approximate Calculation ---
    else:
        if n_samples is None:
            raise ValueError(
                "If approximate=True, 'n_samples' must be provided (int > 0 or float (0, 1])."
            )

        num_to_sample: int
        if isinstance(n_samples, int):
            if n_samples <= 0:
                raise ValueError("If 'n_samples' is int, it must be > 0.")
            if n_samples > landscape.n_configs:
                warnings.warn(
                    f"'n_samples' ({n_samples}) > total configurations ({landscape.n_configs}). Sampling all."
                )
                num_to_sample = landscape.n_configs
            else:
                num_to_sample = n_samples
        elif isinstance(n_samples, float):
            if not 0.0 < n_samples <= 1.0:
                raise ValueError("If 'n_samples' is float, it must be in (0, 1].")
            num_to_sample = max(
                1, int(n_samples * landscape.n_configs)
            )  # Ensure at least 1 sample
        else:
            raise ValueError("'n_samples' must be an integer (>0) or a float (0, 1].")

        all_nodes = list(landscape.graph.nodes())
        if not all_nodes:  # Handle empty graph case edge case
            warnings.warn("Graph has no nodes to sample from.", RuntimeWarning)
            return 0.0

        # Ensure we don't try to sample more nodes than exist
        actual_sample_size = min(num_to_sample, len(all_nodes))
        if actual_sample_size < num_to_sample and landscape.verbose:
            print(
                f"  Reduced sample size to {actual_sample_size} due to available nodes."
            )

        if actual_sample_size == 0:  # If graph had nodes but num_to_sample ended up 0
            warnings.warn(
                "Calculated sample size is 0. Cannot approximate.", RuntimeWarning
            )
            return 0.0

        sample_nodes = random.sample(all_nodes, actual_sample_size)

        count_reaching_go = 0

        for start_node in sample_nodes:
            try:
                # Use the reachability check function
                if is_ancestor_fast(landscape.graph, start_node, landscape.go_index):
                    count_reaching_go += 1
            except Exception as e:
                # Handle potential errors during graph traversal for a specific node
                warnings.warn(
                    f"Reachability check failed for sample node {start_node}: {e}",
                    RuntimeWarning,
                )
                continue  # Skip this sample

        # Calculate approximation based on successful checks
        approx_accessibility = count_reaching_go / actual_sample_size

        return approx_accessibility


def fdc(
    landscape,
    method: str = "spearman",
) -> tuple:
    """
    Calculate the fitness distance correlation (FDC) of a landscape. This metric assesses how likely it is
    to encounter higher fitness values when moving closer to the global optimum.

    Parameters
    ----------
    method : str, one of {"spearman", "pearson"}, default="spearman"
        The correlation measure used to assess FDC.

    Returns
    -------
    (float, float) : tuple
        A tuple containing the FDC value and the p-value. The FDC value ranges from -1 to 1, where a value
        close to 1 indicates a positive correlation between fitness and distance to the global optimum.
    """

    data = landscape.get_data()

    if method == "spearman":
        correlation, p_value = spearmanr(data["dist_go"], data["fitness"])
    elif method == "pearson":
        correlation, p_value = pearsonr(data["dist_go"], data["fitness"])
    else:
        raise ValueError(
            f"Invalid method {method}. Please choose either 'spearman' or 'pearson'."
        )

    return correlation, p_value


def ffi(
    landscape, frac: float = 1, min_len: int = 3, method: str = "spearman"
) -> tuple:
    """
    Calculate the fitness flattening index (FFI) of the landscape. It assesses whether the
    landscape tends to be flatter around the global optimum by evaluating adaptive paths.

    Parameters
    ----------
    frac : float, default=1
        The fraction of adaptive paths to be assessed.

    min_len : int, default=3
        Minimum length of an adaptive path for it to be considered.

    method : str, one of {"spearman", "pearson"}, default="spearman"
        The correlation measure used to assess FFI.

    Returns
    -------
    tuple
        A tuple containing the FFI value and the p-value. The FFI value ranges from -1 to 1,
        where a value close to 1 indicates a flatter landscape around the global optimum.
    """

    def check_diminishing_differences(data, method):
        data.index = range(len(data))
        differences = data.diff().dropna()
        index = np.arange(len(differences))
        if method == "pearson":
            correlation, p_value = pearsonr(index, differences)
        elif method == "spearman":
            correlation, p_value = spearmanr(index, differences)
        else:
            raise ValueError(
                "Invalid method. Please choose either 'spearman' or 'pearson'."
            )
        return correlation, p_value

    data = landscape.get_data()
    fitness = data["fitness"]

    ffi_list = []
    p_values = []

    for i in data.index:
        lo, _, trace = hill_climb(
            landscape.graph, i, "delta_fit", verbose=0, return_trace=True
        )
        if len(trace) >= min_len and lo == landscape.go_index:
            fitnesses = fitness.loc[trace]
            ffi, p_value = check_diminishing_differences(fitnesses, method)
            ffi_list.append(ffi)
            p_values.append(p_value)

    ffi = pd.Series(ffi_list).mean()
    mean_p_value = pd.Series(p_values).mean()
    return ffi, mean_p_value


def autocorrelation(
    landscape, walk_length: int = 20, walk_times: int = 1000, lag: int = 1
) -> Tuple[float, float]:
    """
    A measure of landscape ruggedness. It operates by calculating the autocorrelation of
    fitness values over multiple random walks on a graph.

    Parameters:
    ----------
    walk_length : int, default=20
        The length of each random walk.

    walk_times : int, default=1000
        The number of random walks to perform.

    lag : int, default=1
        The distance lag used for calculating autocorrelation. See pandas.Series.autocorr.

    Returns:
    -------
    autocorr : Tuple[float, float]
        A tuple containing the mean and variance of the autocorrelation values.
    """

    corr_list = []
    nodes = list(landscape.graph.nodes())
    for _ in range(walk_times):
        random_node = random.choice(nodes)
        logger = random_walk(landscape.graph, random_node, "fitness", walk_length)
        autocorrelation = logger["fitness"].astype(float).autocorr(lag=lag)
        corr_list.append(autocorrelation)

    autocorr = pd.Series(corr_list).median()

    return autocorr, pd.Series(corr_list).var()


def neutrality(landscape, threshold: float = 0.01) -> float:
    """
    Calculate the neutrality index of the landscape. It assesses the proportion of neighbors
    with fitness values within a given threshold, indicating the presence of neutral areas in
    the landscape.

    Parameters
    ----------
    threshold : float, default=0.01
        The fitness difference threshold for neighbors to be considered neutral.

    Returns
    -------
    neutrality : float
        The neutrality index, which ranges from 0 to 1, where higher values indicate more
        neutrality in the landscape.
    """

    neutral_pairs = 0
    total_pairs = 0

    for node in landscape.graph.nodes:
        fitness = landscape.graph.nodes[node]["fitness"]
        for neighbor in landscape.graph.neighbors(node):
            neighbor_fitness = landscape.graph.nodes[neighbor]["fitness"]
            if abs(fitness - neighbor_fitness) <= threshold:
                neutral_pairs += 1
            total_pairs += 1

    neutrality = neutral_pairs / total_pairs if total_pairs > 0 else 0

    return neutrality


def ruggedness(landscape) -> float:
    """
    Calculate the ruggedness index of the landscape. It is defined as the ratio of the number
    of local optima to the total number of configurations.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    Returns
    -------
    float
        The ruggedness index, ranging from 0 to 1.
    """

    n_lo = landscape.n_lo
    n_configs = landscape.n_configs
    if n_configs == 0:
        return 0.0
    ruggedness = n_lo / n_configs

    return ruggedness


def basin_fit_corr(landscape, method: str = "spearman") -> tuple:
    """
    Calculate the correlation between the size of the basin of attraction and the fitness of local optima.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    method : str, one of {"spearman", "pearson"}, default="spearman"
        The correlation measure to use.

    Returns
    -------
    tuple
        A tuple containing the correlation coefficient and the p-value.
    """

    lo_data = landscape.get_data(lo_only=True)
    basin_sizes = lo_data["size_basin_best"]
    fitness_values = lo_data["fitness"]

    if method == "spearman":
        correlation, p_value = spearmanr(basin_sizes, fitness_values)
    elif method == "pearson":
        correlation, p_value = pearsonr(basin_sizes, fitness_values)
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'spearman' or 'pearson'.")

    return correlation, p_value


def gradient_intensity(landscape) -> float:
    """
    Calculate the gradient intensity of the landscape. It is defined as the average absolute
    fitness difference (delta_fit) across all edges.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    Returns
    -------
    float
        The gradient intensity.
    """

    total_edges = landscape.graph.number_of_edges()
    if total_edges == 0:
        return 0.0
    total_delta_fit = sum(
        abs(data.get("delta_fit", 0)) for _, _, data in landscape.graph.edges(data=True)
    )
    gradient = total_delta_fit / total_edges

    return gradient


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


def idiosyncratic_index(landscape, mutation):
    """
    Calculates the idiosyncratic index for the fitness landscape proposed in [1].

    The idiosyncratic index of a specific genetic mutation quantifies the sensitivity
    of a specific mutation to idiosyncratic epistasis. It is defined as the as the
    variation in the fitness difference between genotypes that differ by the mutation,
    relative to the variation in the fitness difference between random genotypes for
    the same number of genotype pairs. We compute this for the entire fitness landscape
    by averaging it across individual mutations.

    The idiosyncratic index for a landscape varies from 0 to 1, corresponding to the
    minimum and maximum levels of idiosyncrasy, respectively.

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

    Returns
    -------
    float
        The calculated idiosyncratic index.
    """
    A, pos, B = mutation

    data = landscape.get_data()
    X = data.iloc[:, : len(landscape.data_types)]
    f = data["fitness"]

    # Check if alleles A and B exist at the specified position
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
        print(
            f"Warning: No genotypes found for allele '{A}' or '{B}' at position '{pos}'. Returning 0.0."
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
        return 0.0

    mutation_effects = df_merged["fitness_B"] - df_merged["fitness_A"]
    n_pairs = len(mutation_effects)

    if n_pairs <= 1:
        return 0.0

    std_mutation_effect = np.std(mutation_effects)
    all_fitness_values = f.values

    if len(all_fitness_values) <= 1 or np.all(
        all_fitness_values == all_fitness_values[0]
    ):
        return 0.0

    rand_f1 = np.random.choice(all_fitness_values, size=n_pairs, replace=True)
    rand_f2 = np.random.choice(all_fitness_values, size=n_pairs, replace=True)

    random_diffs = rand_f1 - rand_f2

    std_random_diff = np.std(random_diffs)

    if std_random_diff == 0:
        return 0.0

    idiosyncratic_val = std_mutation_effect / std_random_diff

    return idiosyncratic_val


def pairwise_epistasis(X, f, pos1, pos2):
    """
    Assess the pairwise epistasis effects between all unique unordered mutations at two specified positions.

    Parameters
    ----------
    X : pd.DataFrame
        The genotype matrix where each column corresponds to a genetic position.

    f : pd.Series
        The fitness values corresponding to each genotype.

    pos1 : str
        The name of the first position to assess mutations for.

    pos2 : str
        The name of the second position to assess mutations for.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing mutation pairs, median absolute epistasis effect,
        p-values, and significance flags.
    """

    def get_diff(df1, df2, f):
        f.name = "fitness"
        df1 = pd.concat([df1, f], axis=1, join="inner")
        df2 = pd.concat([df2, f], axis=1, join="inner")
        df1.set_index(0, inplace=True)
        df2.set_index(0, inplace=True)

        df_diff = pd.merge(
            df1, df2, left_index=True, right_index=True, suffixes=("_1", "_2")
        )
        df_diff.index = range(len(df_diff))
        diff = df_diff["fitness_1"] - df_diff["fitness_2"]
        return diff

    def compute_epistasis(X, f, pos1, pos2, mut1, mut2):
        _, a, A = mut1
        _, b, B = mut2

        X_AB = X[(X[pos1] == A) & (X[pos2] == B)]
        X_ab = X[(X[pos1] == a) & (X[pos2] == b)]
        X_Ab = X[(X[pos1] == A) & (X[pos2] == b)]
        X_aB = X[(X[pos1] == a) & (X[pos2] == B)]

        X_AB = pd.Series(X_AB.drop(columns=[pos1, pos2]).apply(tuple, axis=1))
        X_ab = pd.Series(X_ab.drop(columns=[pos1, pos2]).apply(tuple, axis=1))
        X_Ab = pd.Series(X_Ab.drop(columns=[pos1, pos2]).apply(tuple, axis=1))
        X_aB = pd.Series(X_aB.drop(columns=[pos1, pos2]).apply(tuple, axis=1))

        f_AB_ab = get_diff(X_AB, X_ab, f)
        f_Ab_ab = get_diff(X_Ab, X_ab, f)
        f_aB_ab = get_diff(X_aB, X_ab, f)

        diff = f_AB_ab - (f_Ab_ab + f_aB_ab)

        if diff.empty:
            cohen_d = np.nan
            ttest_p = np.nan
            mean = np.nan
        else:
            cohen_d = abs(diff).median() / f.std()
            _, ttest_p = ttest_1samp(diff, 0)
            mean = diff.mean()

        return {
            "pos1": pos1,
            "mutation1_from": a,
            "mutation1_to": A,
            "pos2": pos2,
            "mutation2_from": b,
            "mutation2_to": B,
            "cohen_d": cohen_d,
            "ttest_p": ttest_p,
            "mean_diff": mean,
        }

    unique_vals1 = sorted(X[pos1].dropna().unique())
    unique_vals2 = sorted(X[pos2].dropna().unique())

    mutations1 = [(pos1, a, b) for a, b in combinations(unique_vals1, 2)]
    mutations2 = [(pos2, c, d) for c, d in combinations(unique_vals2, 2)]

    mutation_pairs = list(product(mutations1, mutations2))

    results = [
        compute_epistasis(X, f, pos1, pos2, mut1, mut2) for mut1, mut2 in mutation_pairs
    ]

    epistasis_df = pd.DataFrame(results)

    return epistasis_df


def all_pairwise_epistasis(X, f, n_jobs=1):
    """
    Compute and aggregate epistasis effects between all unique pairs of positions in the genotype matrix using parallel execution.

    Parameters
    ----------
    X : pd.DataFrame
        The genotype matrix where each column corresponds to a genetic position.
    f : pd.Series
        The fitness values corresponding to each genotype.
    n_jobs : int, default=1
        The number of parallel jobs to run. -1 means using all available cores.

    Returns
    -------
    pd.DataFrame
        An aggregated DataFrame containing average epistasis scores for each position pair.
    """

    positions = list(X.columns)
    position_pairs = list(combinations(positions, 2))

    detailed_results = Parallel(n_jobs=n_jobs)(
        delayed(pairwise_epistasis)(X, f, pos1, pos2) for pos1, pos2 in position_pairs
    )

    all_epistasis_df = pd.concat(detailed_results, ignore_index=True)

    aggregated = (
        all_epistasis_df.groupby(["pos1", "pos2"])
        .agg(
            average_cohen_d=("cohen_d", "median"),
            average_mean_diff=("mean_diff", "median"),
            most_significant_p=("ttest_p", "min"),
            total_mutation_pairs=("ttest_p", "count"),
        )
        .reset_index()
    )

    return aggregated


def r_s_ratio(landscape) -> float:
    """
    Calculate the roughness-to-slope (r/s) ratio of a fitness landscape.

    This metric quantifies the deviation from additivity by comparing the
    standard deviation of the residuals from a linear model fit (roughness)
    to the mean absolute additive coefficients (slope). Higher values
    indicate greater ruggedness and epistasis relative to the additive trend.

    Calculation follows definitions used in Rough Mount Fuji models and
    empirical landscape studies, e.g., [1]-[4].

    References
    ----------
    [1] I. Fragata et al., "Evolution in the light of fitness landscape
        theory," Trends Ecol. Evol., vol. 34, no. 1, pp. 69-82, Jan. 2019.
    [2] T. Aita, H. Uchiyama, T. Inaoka, M. Nakajima, T. Kokubo, and
        Y. Husimi, "Analysis of a local fitness landscape with a model of
        the rough Mount Fuji-type landscape," Biophys. Chem., vol. 88,
        no. 1-3, pp. 1-10, Dec. 2000.
    [3] A. Skwara et al., "Statistically learning the functional landscape
        of microbial communities," Nat. Ecol. Evol., vol. 7, no. 11,
        pp. 1823-1833, Nov. 2023.
    [4] C. Bank, R. T. Hietpas, J. D. Jensen, and D. N. A. Bolon, "A
        systematic survey of an intragenic epistatic landscape," Proc. Natl.
        Acad. Sci. USA, vol. 113, no. 50, pp. 14424-14429, Dec. 2016.

    Parameters
    ----------
    landscape : graphfal.landscape.Landscape
        A Landscape object instance. It must have attributes `n_vars`,
        `data_types`, and a method `get_data()` that returns a DataFrame
        containing the raw configurations and a 'fitness' column.

    Returns
    -------
    float
        The roughness-to-slope (r/s) ratio. Returns np.inf if the slope (s)
        is zero or very close to zero. Returns np.nan if the calculation fails.

    Raises
    ------
    ValueError
        If the landscape object is missing required attributes/methods or if
        data types are unsupported.
    """
    # 1. Get Data
    data = landscape.get_data()

    raw_X = data.iloc[:, : landscape.n_vars]
    fitness_values = data["fitness"].values
    data_types = landscape.data_types

    # 2. Prepare Numerical Genotype Representation for Additive Model
    numerical_X_cols = {}
    cols = list(data_types.keys())  # Maintain order

    for col in cols:
        dtype = data_types[col]
        if dtype == "boolean":
            # Convert boolean to 0/1 integer representation
            numerical_X_cols[col] = raw_X[col].astype(bool).astype(int)
        elif dtype == "categorical":
            # Use pandas Categorical codes (integer representation)
            numerical_X_cols[col] = pd.Categorical(raw_X[col]).codes
        elif dtype == "ordinal":
            # Use pandas Categorical codes (integer representation)
            numerical_X_cols[col] = pd.Categorical(raw_X[col], ordered=True).codes
        else:
            raise ValueError(
                f"Unsupported data type '{dtype}' for r/s ratio calculation in column '{col}'"
            )

    numerical_X = pd.DataFrame(numerical_X_cols, index=raw_X.index)
    X_fit = numerical_X.values  # Use numpy array for sklearn

    # Check for sufficient data
    n_samples, n_features = X_fit.shape
    if n_samples <= n_features:
        warnings.warn(
            f"Number of samples ({n_samples}) is less than or equal to "
            f"the number of features ({n_features}) after encoding. "
            "Linear regression might be underdetermined or unstable.",
            UserWarning,
        )

    # 3. Fit Additive (Linear) Model
    try:
        linear_model = LinearRegression(fit_intercept=True)
        linear_model.fit(X_fit, fitness_values)

        # 4. Calculate Slope (s)
        # Mean of absolute values of additive coefficients (betas)
        additive_coeffs = linear_model.coef_
        # potential case where n_features is 1
        if n_features == 1 and np.isscalar(additive_coeffs):
            slope_s = np.abs(additive_coeffs)
        elif n_features == 1 and isinstance(additive_coeffs, (np.ndarray, list)):
            slope_s = np.abs(additive_coeffs[0])
        elif n_features > 1:
            slope_s = np.mean(np.abs(additive_coeffs))
        else:  # n_features == 0 (should not happen if validation is correct)
            slope_s = 0

        if np.isclose(slope_s, 0):
            warnings.warn(
                "Slope 's' is zero or near zero. Landscape may be flat "
                "or purely epistatic according to the linear fit. Returning inf.",
                UserWarning,
            )
            return np.inf

        predicted_fitness = linear_model.predict(X_fit)
        residuals = fitness_values - predicted_fitness
        roughness_r = np.std(residuals, ddof=1 if n_samples > 1 else 0)

        r_s_ratio_value = roughness_r / slope_s

        return r_s_ratio_value

    except Exception as e:
        warnings.warn(
            f"Calculation of r/s ratio failed: {e}. Returning np.nan.", UserWarning
        )
        return np.nan


def _find_squares(landscape) -> list:
    """
    Identify all 4-node cycles (squares) in the landscape's graph. A square is a
    quadruplet of sequences that contain a wild type,  two of its one-mutant neighbours,
    and the double mutant that can be formed from the single mutants.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    Returns
    -------
    list of list of nodes
        A list where each element is a list of nodes forming a 4-node cycle (square).
    """
    graph = landscape.graph
    undir_graph = nx.to_undirected(graph)
    cycles = nx.cycle_basis(undir_graph)

    squares = []
    for cycle in cycles:
        if len(cycle) == 4:
            squares.append(cycle)
    return squares


def _assign_roles(landscape, squares):
    """
    Assign mutation roles to nodes within each 4-node cycle (square).

    This function categorizes nodes in each square based on their fitness values into
    wild type, one-mutants, and double mutant. Specifically, the sequence with the
    highest fitness is designated as the double mutant. The remaining roles can then
    be assigned accordingly based on mutations.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    squares : list of list of nodes
        A list of squares (4-node cycles) to analyze.

    Returns
    -------
    list of dict
        A list of dictionaries, each representing a square with assigned roles as well
        as the fitness values of each sequence.
    """
    squares_with_roles = []
    graph = landscape.graph
    for square in squares:
        fitness_values = {node: graph.nodes[node].get("fitness", 0) for node in square}
        double_mutant = max(fitness_values, key=fitness_values.get)
        neighbors = set(nx.subgraph(graph, square).predecessors(double_mutant))
        single_mutants = [node for node in square if node in neighbors]
        wild_type = next(
            node
            for node in square
            if node not in single_mutants and node != double_mutant
        )

        squares_with_roles.append(
            {
                "wild_type": wild_type,
                "single_mutant_1": single_mutants[0],
                "single_mutant_2": single_mutants[1],
                "double_mutant": double_mutant,
                "fitness_values": fitness_values,
            }
        )

    return squares_with_roles


def _mag_sign_epistasis(landscape, squares_with_roles):
    """
    Determine the prevalence of three types of epistasis in a fitness landscape. This is
    achieved by determining the proportion of squares exhibiting different types of epistasis:
    - Magnitude epistasis
    - Sign epistasis
    - Reciprocal sign epistasis

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    squares_with_roles : list of dict
        A list of squares with assigned mutation roles and fitness values.

    Returns
    -------
    dict
        A dictionary containing the proportion of squares with each type of epistasis:
        - "magnitude epistasis": Proportion showing magnitude epistasis.
        - "sign epistasis": Proportion showing sign epistasis.
        - "reciprocal sign epistasis": Proportion showing reciprocal sign epistasis.
    """
    idx_to_fitness = landscape.get_data()["fitness"].to_dict()
    df_squares = pd.DataFrame(squares_with_roles).iloc[:, :4]
    df_squares.columns = ["ab", "aB", "Ab", "AB"]

    for col in df_squares.columns:
        df_squares[col] = df_squares[col].replace(idx_to_fitness)

    mut1 = df_squares["aB"] - df_squares["ab"]
    mut2 = df_squares["AB"] - df_squares["Ab"]
    mut3 = df_squares["Ab"] - df_squares["ab"]
    mut4 = df_squares["AB"] - df_squares["aB"]

    magnitude = df_squares[(mut1 * mut2 > 0) & (mut3 * mut4 > 0)]
    perc_mag = len(magnitude) / len(df_squares)

    sign = df_squares[
        ((mut1 * mut2 < 0) & (mut3 * mut4 > 0))
        | ((mut1 * mut2 > 0) & (mut3 * mut4 < 0))
    ]
    perc_sign = len(sign) / len(df_squares)

    reci_sign = df_squares[(mut1 * mut2 < 0) & (mut3 * mut4 < 0)]
    perc_reci_sign = len(reci_sign) / len(df_squares)

    dict_epistasis = {
        "magnitude epistasis": perc_mag,
        "sign epistasis": perc_sign,
        "reciprocal sign epistasis": perc_reci_sign,
    }

    return dict_epistasis


def _pos_neg_epistasis(landscape, squares_with_roles):
    """
    Determine the prevalence of positive and negative epistasis in a fitness landscape.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    squares_with_roles : list of lists
        A list of squares with assigned mutation roles and fitness values.

    Returns
    -------
    dict
        A dictionary containing the proportions of:
        - "positive epistasis": Proportion of squares where the combined effect of two mutations
          is greater than the sum of their individual effects.
        - "negative epistasis": Proportion of squares where the combined effect of two mutations
          is less than or equal to the sum of their individual effects.
    """
    idx_to_fitness = landscape.get_data()["fitness"].to_dict()
    df_squares = pd.DataFrame(squares_with_roles).iloc[:, :4]
    df_squares.columns = ["ab", "aB", "Ab", "AB"]

    for col in df_squares.columns:
        df_squares[col] = df_squares[col].replace(idx_to_fitness)

    mut1 = df_squares["aB"] - df_squares["ab"]
    mut2 = df_squares["Ab"] - df_squares["ab"]
    mut3 = df_squares["AB"] - df_squares["ab"]

    positive = df_squares[mut3 > (mut1 + mut2)]
    perc_positive = len(positive) / len(df_squares)
    perc_negative = 1 - perc_positive

    dict_epistassi = {
        "positive epistasis": perc_positive,
        "negative epistasis": perc_negative,
    }

    return dict_epistassi


def classify_epistasis(landscape, type: str = "pos_neg") -> dict:
    """
    Classify the type of epistasis present in a given fitness landscape.

    This function analyzes a fitness landscape to determine whether it exhibits positive/negative
    epistasis or magnitude/sign/reciprocal sign epistasis. The classification is determined by
    identifying 4-node cycles (squares) and analyzing the relationships between mutations.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    type : str, optional (default="pos_neg")
        The type of epistasis to classify. Supported options are:
        - "pos_neg": Classifies positive and negative epistasis.
        - "mag_sign": Classifies magnitude epistasis, sign epistasis, and reciprocal sign epistasis.

    Returns
    -------
    dict
        A dictionary with the proportion of squares exhibiting the specified type(s) of epistasis:
        - If `type="pos_neg"`, the dictionary contains:
            - "positive epistasis": Proportion of squares with positive epistasis.
            - "negative epistasis": Proportion of squares with negative epistasis.
        - If `type="mag_sign"`, the dictionary contains:
            - "magnitude epistasis": Proportion of squares with magnitude epistasis.
            - "sign epistasis": Proportion of squares with sign epistasis.
            - "reciprocal sign epistasis": Proportion of squares with reciprocal sign epistasis.

    Raises
    ------
    ValueError
        If the `type` argument is not "pos_neg" or "mag_sign".
    """
    squares = _find_squares(landscape)
    squares_with_roles = _assign_roles(landscape, squares)
    if type == "pos_neg":
        dict_epistasis = _pos_neg_epistasis(landscape, squares_with_roles)
    elif type == "mag_sign":
        dict_epistasis = _mag_sign_epistasis(landscape, squares_with_roles)
    else:
        raise ValueError("Invalid type specified. Use 'pos_neg' or 'mag_sign'.")

    return dict_epistasis
