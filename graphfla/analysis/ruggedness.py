import numpy as np
import pandas as pd

import random

from ..algorithms import RandomWalk, SearchCache
from ._roughness import roughness_slope_ratio


from ._utils import _pythonize



def local_optima_ratio(landscape) -> float:
    """
    The most intuitive measure of landscape ruggedness. It is based on the ratio
    of the number of local optima to the total number of configurations in the landscape.

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
        # Undefined on an empty landscape (no configurations to count).
        return float("nan")
    ruggedness = n_lo / n_configs

    return _pythonize(ruggedness)


def autocorrelation(
    landscape,
    walk_length: int = 20,
    walk_times: int = 1000,
    lag: int = 1,
    seed: int = None,
) -> float:
    """
    A measure of landscape ruggedness. It operates by calculating the autocorrelation of
    fitness values over multiple random walks on a graph.

    Parameters:
    ----------
    landscape : Landscape
        The fitness landscape object.

    walk_length : int, default=20
        The length of each random walk.

    walk_times : int, default=1000
        The number of random walks to perform.

    lag : int, default=1
        The distance lag used for calculating autocorrelation.

    seed : int, optional
        Seed for a local RNG, making the set of random walks reproducible. If
        None (default), the global ``random`` state is used.

    References:
    ----------
    [1] E. Weinberger, "Correlated and Uncorrelated Fitness Landscapes and How to Tell
        the Difference", Biol. Cybern. 63, 325-336 (1990).

    Returns:
    -------
    float
        The lag-``lag`` autocorrelation of fitness, pooled across all random
        walks under a single grand mean. Returns NaN if no walk yields more
        than ``lag`` steps.
    """
    # Pool lagged products under one grand mean: per-walk centering biases the
    # estimate toward zero; pooling is the unbiased estimator (Weinberger 1990).
    rand = random.Random(seed) if seed is not None else random
    series = []
    cache = SearchCache(landscape.graph)
    for _ in range(walk_times):
        random_node = rand.randrange(0, landscape.n_configs)
        walk_seed = rand.getrandbits(32) if seed is not None else None
        result = RandomWalk(cache, length=walk_length, seed=walk_seed).run(random_node)
        if len(result.path) > lag:
            series.append(cache.fitness[result.path].astype(float))

    if not series:
        return _pythonize(np.nan)

    grand_mean = np.concatenate(series).mean()
    num = 0.0
    den = 0.0
    for x in series:
        xc = x - grand_mean
        num += float(np.dot(xc[: len(xc) - lag], xc[lag:]))
        den += float(np.dot(xc, xc))

    return _pythonize(num / den if den else np.nan)


def gradient_intensity(landscape) -> float:
    """
    Calculate the gradient intensity of the landscape using igraph. It is
    defined as the average absolute fitness difference (delta_fit) across all edges.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object.

    Returns
    -------
    float
        The gradient intensity.
    """

    graph = landscape.graph
    total_edges = graph.ecount()
    if total_edges == 0:
        # Undefined with no edges (no fitness gradients to average).
        return float("nan")

    # delta_fit defaults to 0 when the attribute is missing on an edge
    delta_fits = [abs(edge.attributes().get("delta_fit", 0)) for edge in graph.es]
    total_delta_fit = sum(delta_fits)
    fitness = landscape.graph.vs["fitness"]

    mean_fitness = pd.Series(fitness).mean()
    if mean_fitness == 0:
        # Normalisation by mean fitness is undefined when the mean is zero.
        return float("nan")
    gradient = (total_delta_fit / total_edges) / mean_fitness
    return _pythonize(gradient)


def r_s_ratio(landscape) -> float:
    r"""Return the roughness-to-slope ratio of an additive least-squares fit.

    Parameters
    ----------
    landscape : Landscape
        Built landscape. Each retained configuration has equal weight; its
        stored ``fitness`` is the objective value, with no log transformation.
        Boolean variables use 0/1 coding. Categorical variables use one
        indicator per observed state except the first in pandas category
        order (normally sorted labels). Ordinal variables use the integer
        ranks established during construction, or pandas category order if
        construction codes are unavailable. Constant variables are excluded.

    Returns
    -------
    ratio : float
        Residual root mean square divided by the mean absolute coefficient,
        excluding the intercept. The mean weights encoded columns equally,
        not original variables. Returns zero, up to rounding error, for an
        exact additive fit with nonzero slope. Returns ``numpy.inf`` when
        slope is at most ``1e-12`` times the objective range. Returns
        ``numpy.nan`` for empty or constant data, an unidentifiable fit,
        a failed numerical solve, or a model exceeding the workspace limit.

    Raises
    ------
    RuntimeError
        If the landscape has not been built.
    ValueError
        If a variable type is unsupported or an objective value is nonfinite.

    Warns
    -----
    UserWarning
        If the ratio is undefined, the slope is numerically zero, or the fit
        is saturated (no residual degrees of freedom). A fit also returns
        NaN with a warning if its quadratic QR workspace exceeds 64 MiB.

    See Also
    --------
    higher_order_epistasis : Variance explained by interactions up to an order.

    Notes
    -----
    With an intercept, fit :math:`\hat f_i=b_0+\sum_j b_j x_{ij}` by ordinary
    least squares, then calculate :math:`r=\sqrt{\sum_i(f_i-\hat f_i)^2/N}`
    and :math:`s=\sum_j|b_j|/p` [1]_. Missing configurations are not imputed;
    graph edges and the optimization direction do not enter the fit.

    For categorical variables with more than two states, changing the
    reference state can change ``s`` and the ratio. For ordinal variables,
    nonlinear effects of a single variable also contribute to ``r``.
    Compare values only under consistent coding and objective scales.

    References
    ----------
    .. [1] I. G. Szendro et al., "Quantitative analyses of empirical fitness
       landscapes," J. Stat. Mech. P01005 (2013), Eqs. (3)-(5).
       https://doi.org/10.1088/1742-5468/2013/01/P01005

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import r_s_ratio
    >>> landscape = BooleanLandscape().build_from_data(
    ...     [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 2, 4], verbose=False)
    >>> round(r_s_ratio(landscape), 3)
    0.125
    """
    return roughness_slope_ratio(landscape)
