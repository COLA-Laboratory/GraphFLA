from .distances import mixed_distance
from .algorithms import hill_climb, random_walk
from scipy.stats import spearmanr, pearsonr, binomtest, ttest_1samp
from typing import Any, Tuple
from itertools import combinations, product
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import networkx as nx
import random


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


def basin_size_fit_corr(landscape, method: str = "spearman") -> tuple:
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

    lo_indices = landscape.lo_index
    if not lo_indices:
        raise ValueError("No local optima found in the landscape.")

    basin_sizes = []
    fitness_values = []
    for lo in lo_indices:
        basin_size = landscape.graph.nodes[lo].get("size_basin", 0)
        fitness = landscape.graph.nodes[lo].get("fitness", 0)
        basin_sizes.append(basin_size)
        fitness_values.append(fitness)

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
    raise NotImplementedError("This method is currently not implemented.")

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
    Calculate the roughness-to-slope (r/s) ratio of a fitness landscape. This metric quantifies
    the deviation from additivity, with higher values indicating greater ruggedness and epistasis.

    Parameters
    ----------
    landscape : object
        A landscape object with a `get_data()` method that provides sequence and fitness data.

    Returns
    -------
    float
        The roughness-to-slope (r/s) ratio.
    """
    data = landscape.get_data()

    fitness_column = 'fitness'

    encoded_sequences = pd.get_dummies(data.iloc[:,:landscape.n_vars])
    fitness_values = data[fitness_column]

    linear_model = LinearRegression()
    linear_model.fit(encoded_sequences, fitness_values)

    predicted_fitness = linear_model.predict(encoded_sequences)
    residuals = fitness_values - predicted_fitness

    roughness = np.sqrt(np.mean(residuals**2))
    slope = np.mean(np.abs(linear_model.coef_))
    r_s_ratio = roughness / slope

    return r_s_ratio


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
        wild_type = next(node for node in square if node not in single_mutants and node != double_mutant)

        squares_with_roles.append({
            "wild_type": wild_type,
            "single_mutant_1": single_mutants[0],
            "single_mutant_2": single_mutants[1],
            "double_mutant": double_mutant,
            "fitness_values": fitness_values,
        })
        
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
    df_squares = pd.DataFrame(squares_with_roles).iloc[:,:4]
    df_squares.columns = ["ab", "aB", "Ab", "AB"]
    
    for col in df_squares.columns:
        df_squares[col] = df_squares[col].replace(idx_to_fitness)
    
    mut1 = df_squares["aB"] - df_squares["ab"]
    mut2 = df_squares["AB"] - df_squares["Ab"]
    mut3 = df_squares["Ab"] - df_squares["ab"]
    mut4 = df_squares["AB"] - df_squares["aB"]

    magnitude = df_squares[(mut1 * mut2 > 0) & (mut3 * mut4 > 0)]
    perc_mag = len(magnitude) / len(df_squares)

    sign = df_squares[((mut1 * mut2 < 0) & (mut3 * mut4 > 0)) | ((mut1 * mut2 > 0) & (mut3 * mut4 < 0))]
    perc_sign = len(sign) / len(df_squares)

    reci_sign = df_squares[(mut1 * mut2 < 0) & (mut3 * mut4 < 0)]
    perc_reci_sign = len(reci_sign) / len(df_squares)

    dict_epistasis = {
        "magnitude epistasis" : perc_mag,
        "sign epistasis" : perc_sign,
        "reciprocal sign epistasis" : perc_reci_sign
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
    df_squares = pd.DataFrame(squares_with_roles).iloc[:,:4]
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
        "positive epistasis" : perc_positive,
        "negative epistasis" : perc_negative
    }

    return dict_epistassi
    
def classify_epistasis(
        landscape, 
        type: str="pos_neg"
    ) -> dict:
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


