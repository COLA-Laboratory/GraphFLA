import numpy as np
import pandas as pd
import warnings
from scipy.stats import spearmanr, pearsonr, kendalltau

from ..algorithms import HillClimb, SearchCache

from ._utils import _pythonize
import logging

logger = logging.getLogger(__name__)


def neighbor_fitness_correlation(
    landscape, auto_calculate=True, method="pearson"
) -> float:
    r"""Return the correlation between fitness and mean neighbor fitness.

    This metric quantifies the extent to which fitter configurations tend to have
    neighbors with higher fitness values. A strong positive correlation suggests that
    higher-fitness configurations exist in higher-fitness regions of the landscape,
    indicating a structured landscape with potential fitness gradients.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    auto_calculate : bool, default=True
        Compute missing neighbor-fitness attributes through
        ``landscape.neighbor_fitness``. If False, require the cached attributes.
    method : {"pearson", "spearman", "kendall"}, default="pearson"
        Correlation coefficient to calculate.

    Returns
    -------
    correlation : float
        Correlation between fitness and mean neighbor fitness in [-1, 1].
        Configurations with missing values are excluded; return NaN if none
        remain or the correlation is undefined.

    Raises
    ------
    RuntimeError
        If auto_calculate=False and neighbor fitness metrics haven't been calculated.
    ValueError
        If the method is invalid or Pearson correlation has fewer than two
        valid fitness/neighbor-fitness pairs.

    Notes
    -----
    Positive correlation means fitter configurations tend to have fitter
    neighbors; negative correlation indicates the opposite association.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import neighbor_fitness_correlation
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> round(neighbor_fitness_correlation(landscape), 3)
    -0.169
    """
    landscape._check_built()

    if "mean_neighbor_fit" not in landscape.graph.vs.attributes():
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
        warnings.warn(
            "No valid data for correlation calculation after removing NaNs.",
            RuntimeWarning,
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
    r"""Return the correlation between fitness and distance to the global optimum.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape. Distances to its selected global optimum are
        computed lazily when absent.
    method : {"spearman", "pearson"}, default="spearman"
        Correlation coefficient to calculate.

    Returns
    -------
    correlation : float
        Correlation in [-1, 1], or NaN when undefined. Under maximization,
        a negative correlation means fitness tends to increase toward the
        selected global optimum.

    Raises
    ------
    ValueError
        If the method is invalid or Pearson correlation has fewer than two
        configurations.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import fdc
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> round(fdc(landscape), 3)
    -0.949
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
    r"""Return the mean fitness-increment trend along greedy adaptive paths.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    min_len : int, default=3
        Minimum number of configurations, including the starting configuration,
        in a greedy path. Only paths ending at the selected global optimum count.
    method : {"spearman", "pearson"}, default="spearman"
        Correlation coefficient to calculate.

    Returns
    -------
    index : float
        Mean correlation of step index with successive signed fitness changes.
        Under maximization, a negative value means gains tend to decrease along
        paths. Returns NaN if no eligible path has a defined correlation.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import fitness_flattening_index
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 3, 2, 4], verbose=False)
    >>> round(fitness_flattening_index(landscape), 3)
    -1.0
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


def basin_fitness_correlation(landscape, method: str = "spearman") -> float:
    r"""Return the correlation between greedy basin size and local-optimum fitness.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.

    method : {"spearman", "pearson"}, default="spearman"
        The correlation measure to use.

    Returns
    -------
    correlation : float
        Correlation between greedy basin size and local-optimum fitness in
        [-1, 1], or NaN when undefined. Basins are computed lazily when absent.

    Raises
    ------
    ValueError
        If the method is invalid or Pearson correlation has fewer than two
        local optima.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import basin_fitness_correlation
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 3, 2, 1], verbose=False)
    >>> round(basin_fitness_correlation(landscape), 3)
    1.0
    """
    if "size_basin_greedy" not in landscape.graph.vs.attributes():
        if landscape.verbose:
            logger.info("Basin sizes not found. Calculating basins of attraction...")
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
