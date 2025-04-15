import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr

from ..algorithms import hill_climb_igraph


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
        lo, _, trace = hill_climb_igraph(
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
    basin_sizes = lo_data["size_basin_greedy"]
    fitness_values = lo_data["fitness"]

    if method == "spearman":
        correlation, p_value = spearmanr(basin_sizes, fitness_values)
    elif method == "pearson":
        correlation, p_value = pearsonr(basin_sizes, fitness_values)
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'spearman' or 'pearson'.")

    return correlation, p_value
