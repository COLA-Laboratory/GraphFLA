import numpy as np
import pandas as pd

import random
from typing import Optional

from ..algorithms import RandomWalk, SearchCache
from ._roughness import roughness_slope_ratio


from ._utils import _pythonize



def local_optima_ratio(landscape) -> float:
    r"""Return the number of local-optimum plateaus per configuration.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.

    Returns
    -------
    ratio : float
        ``landscape.n_lo / landscape.n_configs``, or NaN on an empty landscape.
        The numerator counts local-optimum plateaus, whereas the denominator
        counts configurations. A multi-member peak plateau counts once.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import local_optima_ratio
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> local_optima_ratio(landscape)
    0.25
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
    seed: Optional[int] = None,
) -> float:
    r"""Return the pooled fitness autocorrelation along random walks.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.
    walk_length : int, default=20
        Maximum number of visited configurations, including the starting node.
    walk_times : int, default=1000
        Number of random walks, each starting at a uniformly sampled node.
    lag : int, default=1
        Separation between fitness observations along a walk.
    seed : int or None, default=None
        Seed for local random walks. An integer makes the sample reproducible;
        None uses the global Python ``random`` state.

    Returns
    -------
    correlation : float
        Lagged fitness correlation pooled under one grand mean. Returns NaN if
        no walk has more than ``lag`` observations or pooled variance is zero.

    References
    ----------
    .. [1] E. Weinberger, "Correlated and Uncorrelated Fitness Landscapes and How
       to Tell the Difference", Biol. Cybern. 63, 325-336 (1990).

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import autocorrelation
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> round(autocorrelation(landscape, walk_times=8, seed=0), 3)
    -0.082
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
        if len(result['path']) > lag:
            series.append(cache.fitness[result['path']].astype(float))

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
    r"""Return the mean absolute edge fitness change divided by mean fitness.

    Parameters
    ----------
    landscape : Landscape
        Built fitness landscape.

    Returns
    -------
    intensity : float
        Mean absolute ``delta_fit`` over graph edges, divided by mean fitness.
        Missing edge ``delta_fit`` attributes contribute zero. Returns NaN if
        there are no edges or mean fitness is zero. The sign follows mean fitness.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import gradient_intensity
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0, 1, 2, 4], verbose=False)
    >>> round(gradient_intensity(landscape), 3)
    1.143
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
    graphfla.exceptions.NotBuiltError
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
    walsh_hadamard : Coefficients and variance explained through each order.

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
