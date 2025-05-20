import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau

from ..algorithms import hill_climb


def neighbor_fit_corr(landscape, auto_calculate=True, method="pearson"):
    """
    Calculates the correlation between a configuration's fitness and the mean fitness
    of its neighbors across the fitness landscape.

    This metric quantifies the extent to which fitter configurations tend to have
    neighbors with higher fitness values. A strong positive correlation suggests that
    higher-fitness configurations exist in higher-fitness regions of the landscape,
    indicating a structured landscape with potential fitness gradients.

    Parameters
    ----------
    landscape : BaseLandscape
        The fitness landscape object.
    auto_calculate : bool, default=True
        If True, automatically runs determine_neighbor_fitness() if needed.
        If False, raises an exception when neighbor fitness metrics are missing.
    method : str, default='pearson'
        The correlation method to use. Options are:
        - 'pearson': Standard correlation coefficient
        - 'spearman': Rank correlation
        - 'kendall': Kendall Tau correlation

    Returns
    -------
    dict
        A dictionary containing:
        - 'correlation': The correlation coefficient between fitness and mean neighbor fitness
        - 'p_value': The p-value of the correlation test
        - 'method': The correlation method used
        - 'n_nodes': The number of nodes used in the calculation
        - 'stats': Additional descriptive statistics

    Raises
    ------
    RuntimeError
        If auto_calculate=False and neighbor fitness metrics haven't been calculated.
    ValueError
        If an invalid correlation method is specified.

    Notes
    -----
    - Nodes with no neighbors (and thus NaN mean_neighbor_fit) are excluded
    - A positive correlation suggests that fitter configurations tend to exist in
      higher-fitness regions of the landscape
    - A negative correlation suggests the opposite pattern
    - No correlation suggests random distribution of fitness across the landscape
    """
    landscape._check_built()

    # Check if neighbor fitness has been calculated
    if "mean_neighbor_fit" not in landscape.graph.vs.attributes():
        if auto_calculate:
            if landscape.verbose:
                print(
                    "Neighbor fitness metrics not found. Running determine_neighbor_fitness()..."
                )
            landscape.determine_neighbor_fitness()
        else:
            raise RuntimeError(
                "Neighbor fitness metrics haven't been calculated. "
                "Either call landscape.determine_neighbor_fitness() first "
                "or set auto_calculate=True."
            )

    # Valid correlation methods
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError(
            f"Invalid correlation method: {method}. Choose from 'pearson', 'spearman', or 'kendall'"
        )

    # Extract fitness and mean neighbor fitness values
    fitness_values = landscape.graph.vs["fitness"]
    neighbor_fitness_values = landscape.graph.vs["mean_neighbor_fit"]

    data = pd.DataFrame(
        {"fitness": fitness_values, "mean_neighbor_fit": neighbor_fitness_values}
    )

    # Remove rows with NaN (nodes with no neighbors)
    data_clean = data.dropna()
    n_nodes = len(data_clean)
    n_excluded = len(data) - n_nodes

    if n_nodes == 0:
        if landscape.verbose:
            print(
                "Warning: No valid data for correlation calculation after removing NaNs."
            )
        return {
            "correlation": np.nan,
            "p_value": np.nan,
            "method": method,
            "n_nodes": 0,
            "n_excluded": n_excluded,
            "stats": {"fitness_mean": np.nan, "neighbor_fitness_mean": np.nan},
        }

    # Calculate correlation
    if method == "pearson":
        corr, _ = pearsonr(data_clean["fitness"], data_clean["mean_neighbor_fit"])
    elif method == "spearman":
        corr, _ = spearmanr(data_clean["fitness"], data_clean["mean_neighbor_fit"])
    else:  # kendall
        corr, _ = kendalltau(data_clean["fitness"], data_clean["mean_neighbor_fit"])

    return corr


def fitness_distance_corr(
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

    # Check if the landscape has dist_go calculated
    if "dist_go" not in landscape.graph.vs.attributes():
        # If dist_go is not available, calculate it
        landscape.determine_dist_to_go()

        # Check again in case calculation failed
        if "dist_go" not in landscape.graph.vs.attributes():
            raise RuntimeError(
                "Could not calculate distance to global optimum. Make sure the landscape "
                "has proper configuration data and a valid global optimum."
            )

    data = landscape.get_data()

    if method == "spearman":
        correlation, _ = spearmanr(data["dist_go"], data["fitness"])
    elif method == "pearson":
        correlation, _ = pearsonr(data["dist_go"], data["fitness"])
    else:
        raise ValueError(
            f"Invalid method {method}. Please choose either 'spearman' or 'pearson'."
        )

    return correlation


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

    for i in data.index:
        lo, _, trace = hill_climb(
            landscape.graph, i, "delta_fit", verbose=0, return_trace=True
        )
        if len(trace) >= min_len and lo == landscape.go_index:
            fitnesses = fitness.loc[trace]
            ffi, _ = check_diminishing_differences(fitnesses, method)
            ffi_list.append(ffi)

    ffi = pd.Series(ffi_list).mean()
    return ffi


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
    # Check if basins have been calculated
    if "size_basin_greedy" not in landscape.graph.vs.attributes():
        # If basin sizes are not available, calculate them
        if landscape.verbose:
            print("Basin sizes not found. Calculating basins of attraction...")
        landscape.determine_basin_of_attraction()

        # Check again in case calculation failed
        if "size_basin_greedy" not in landscape.graph.vs.attributes():
            raise RuntimeError(
                "Could not calculate basin sizes. Make sure the landscape "
                "has a valid graph structure for basin calculation."
            )

    lo_data = landscape.get_data(lo_only=True)
    basin_sizes = lo_data["size_basin_greedy"]
    fitness_values = lo_data["fitness"]

    if method == "spearman":
        corr_greedy, _ = spearmanr(basin_sizes, fitness_values)
    elif method == "pearson":
        corr_greedy, _ = pearsonr(basin_sizes, fitness_values)
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'spearman' or 'pearson'.")

    if "size_basin_accessible" in lo_data.columns:
        basin_sizes_accessible = lo_data["size_basin_accessible"]
        if method == "spearman":
            corr_accessible, _ = spearmanr(basin_sizes_accessible, fitness_values)
        elif method == "pearson":
            corr_accessible, _ = pearsonr(basin_sizes_accessible, fitness_values)

        return {
            "greedy": corr_greedy,
            "accessible": corr_accessible,
        }
    else:
        return {"greedy": corr_greedy}
