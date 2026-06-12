import numpy as np
import pandas as pd
import warnings
from scipy.stats import spearmanr, pearsonr, kendalltau
from typing import TYPE_CHECKING

from ..algorithms import HillClimb, SearchCache

if TYPE_CHECKING:
    from ..landscape.landscape import Landscape


from ._utils import _pythonize


def neighbor_fitness_correlation(landscape, auto_calculate=True, method="pearson"):
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
        If True, automatically computes neighbour fitness (via the
        landscape's .neighbor_fitness property) if needed.
        If False, raises an exception when neighbor fitness metrics are missing.
    method : str, default='pearson'
        The correlation method to use. Options are:
        - 'pearson': Standard correlation coefficient
        - 'spearman': Rank correlation
        - 'kendall': Kendall Tau correlation

    Returns
    -------
    float
        The correlation coefficient between fitness and mean neighbor fitness.
        Returns NaN if no valid data is available.

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

    if "mean_neighbor_fit" not in landscape.graph.vs.attributes():
        if auto_calculate:
            if landscape.verbose:
                print("Neighbor fitness metrics not found. Computing them...")
            landscape.neighbor_fitness  # lazily computes mean/delta neighbor fitness
        else:
            raise RuntimeError(
                "Neighbor fitness metrics haven't been calculated. "
                "Either access landscape.neighbor_fitness first "
                "or set auto_calculate=True."
            )

    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError(
            f"Invalid correlation method: {method}. Choose from 'pearson', 'spearman', or 'kendall'"
        )

    fitness_values = landscape.graph.vs["fitness"]
    neighbor_fitness_values = landscape.graph.vs["mean_neighbor_fit"]

    data = pd.DataFrame(
        {"fitness": fitness_values, "mean_neighbor_fit": neighbor_fitness_values}
    )

    data_clean = data.dropna()  # drop nodes with no neighbours (NaN fit)
    n_nodes = len(data_clean)

    if n_nodes == 0:
        if landscape.verbose:
            print(
                "Warning: No valid data for correlation calculation after removing NaNs."
            )
        return _pythonize(np.nan)

    if method == "pearson":
        corr, _ = pearsonr(data_clean["fitness"], data_clean["mean_neighbor_fit"])
    elif method == "spearman":
        corr, _ = spearmanr(data_clean["fitness"], data_clean["mean_neighbor_fit"])
    else:  # kendall
        corr, _ = kendalltau(data_clean["fitness"], data_clean["mean_neighbor_fit"])

    return _pythonize(corr)


def fdc(
    landscape,
    method: str = "spearman",
) -> float:
    """
    Calculate the fitness distance correlation (FDC) of a landscape. This metric assesses how likely it is
    to encounter higher fitness values when moving closer to the global optimum.

    Parameters
    ----------
    method : str, one of {"spearman", "pearson"}, default="spearman"
        The correlation measure used to assess FDC.

    Returns
    -------
    float
        The FDC value, ranging from -1 to 1. A value close to 1 indicates a
        positive correlation between fitness and distance to the global optimum
        (and a strongly negative value the usual "easy" gradient-toward-optimum
        structure under maximization).
    """

    if "dist_go" not in landscape.graph.vs.attributes():
        landscape.dist_to_go  # lazily compute distance to global optimum

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

    return _pythonize(correlation)


def fitness_flattening_index(
    landscape, min_len: int = 3, method: str = "spearman"
) -> float:
    """
    Calculate the fitness flattening index (FFI) of the landscape. It assesses whether the
    landscape tends to be flatter around the global optimum by evaluating adaptive paths.

    Parameters
    ----------
    min_len : int, default=3
        Minimum length of an adaptive path for it to be considered.

    method : str, one of {"spearman", "pearson"}, default="spearman"
        The correlation measure used to assess FFI.

    Returns
    -------
    float
        The FFI value, ranging from -1 to 1, where a value close to 1 indicates
        a flatter landscape around the global optimum.
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

    cache = SearchCache(landscape.graph)
    climber = HillClimb(cache)
    for i in data.index:
        result = climber.run(i)
        trace = result.path
        if len(trace) >= min_len and result.final == landscape.go_index:
            fitnesses = fitness.loc[trace]
            ffi, _ = check_diminishing_differences(fitnesses, method)
            ffi_list.append(ffi)

    ffi = pd.Series(ffi_list).mean()
    return _pythonize(ffi)


def basin_fitness_correlation(landscape, method: str = "spearman"):
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
    float
        The correlation coefficient between basin size and local-optimum fitness.
    """
    if "size_basin_greedy" not in landscape.graph.vs.attributes():
        if landscape.verbose:
            print("Basin sizes not found. Calculating basins of attraction...")
        landscape.basins  # lazily computes greedy basins (size_basin_greedy, ...)

        if "size_basin_greedy" not in landscape.graph.vs.attributes():
            raise RuntimeError(
                "Could not calculate basin sizes. Make sure the landscape "
                "has a valid graph structure for basin calculation."
            )

    lo_data = landscape.get_data(lo_only=True)
    basin_sizes = lo_data["size_basin_greedy"]
    fitness_values = lo_data["fitness"]

    if method == "spearman":
        corr, _ = spearmanr(basin_sizes, fitness_values)
    elif method == "pearson":
        corr, _ = pearsonr(basin_sizes, fitness_values)
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'spearman' or 'pearson'.")

    return _pythonize(corr)
