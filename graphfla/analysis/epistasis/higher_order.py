"""Compatibility access to the integrated Walsh-Hadamard order summary."""

import logging
import warnings

from sklearn.utils import Bunch

from ._walsh import _analyze, _positive_integer

logger = logging.getLogger(__name__)


def higher_order_epistasis(
    landscape, max_order=None, verbose=False, n_jobs=1, *, order=None, **kwargs
):
    """Return cumulative and incremental fit quality through each order.

    Pass the result of :func:`walsh_hadamard` to reuse its summary without
    encoding or fitting again. Passing a landscape uses the same fitting
    pipeline, without constructing a coefficient table.

    Parameters
    ----------
    landscape : Landscape or sklearn.utils.Bunch
        Built landscape or the result returned by :func:`walsh_hadamard`.
    max_order : int or None, default=None
        Maximum order to report. None uses two for a landscape, or all computed
        orders for an existing result. Zero reports only the constant baseline.
        For a result, cannot exceed its computed maximum; filtering does not
        change that result's highest-order model variance spectrum.
    verbose : bool, default=False
        Log that a landscape is being fitted. Kept for call compatibility.
    n_jobs : int, default=1
        Parallel Lasso CV jobs when fitting a landscape. An existing result
        requires the default because no fitting takes place.
    order : int or None, default=None
        Deprecated alias for ``max_order``. Do not supply both names.
    **kwargs : dict
        Fitting options forwarded to the Walsh-Hadamard pipeline: ``method``,
        ``alpha``, ``cv``, ``random_state``, ``max_cells``, ``chunk_size``,
        ``max_iter`` and ``tol``. Not accepted with an existing result.

    Returns
    -------
    order_summary : pandas.DataFrame
        The same order-summary schema documented by :func:`walsh_hadamard`.
        ``r2`` is cumulative training fit; ``delta_r2`` is the gain over the
        preceding order. These are not held-out scores. The order-zero row is
        the constant baseline. Constant fitness gives NaN R-squared values.
        For rank-deficient OLS, fitted values and scores are still returned,
        but the unidentified model variance fractions are NaN. No coefficients
        are returned by this function.

    Raises
    ------
    ValueError
        If parameters or data are invalid, the resource limit is exceeded, or
        fitting options are supplied with an existing result.

    Warns
    -----
    FutureWarning
        If the deprecated ``order`` keyword is used.
    UserWarning
        If fitness is constant and R-squared is undefined.

    See Also
    --------
    walsh_hadamard : Compute coefficients and this summary together.

    Notes
    -----
    Nested models are refitted using one shared design. OLS scores depend on
    the fitted interaction space, not its choice of reference basis. Adding
    order k measures improvement conditional on all lower orders; it does not
    establish causal interaction or statistical significance. Lasso gains may
    be negative. Its CV selects penalties, not an out-of-sample R-squared.

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import walsh_hadamard, higher_order_epistasis
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [0., 1., 1., 2.], verbose=False
    ... )
    >>> result = walsh_hadamard(landscape)
    >>> summary = higher_order_epistasis(result)  # no second fit
    >>> summary.delta_r2.round(6).tolist()
    [0.0, 1.0, 0.0]
    """
    if order is not None:
        if max_order is not None:
            raise ValueError("Specify only max_order; order is a deprecated alias.")
        warnings.warn(
            "The order parameter is deprecated; use max_order instead.",
            FutureWarning,
            stacklevel=2,
        )
        max_order = order
    if (
        isinstance(landscape, Bunch)
        and {"coefficients", "order_summary", "fit_info"} <= landscape.keys()
    ):
        if kwargs or n_jobs != 1:
            raise ValueError(
                "Fitting parameters cannot be changed when passing an existing result."
            )
        computed = landscape.fit_info["max_order"]
        maximum = computed if max_order is None else max_order
        _positive_integer(maximum, "max_order", minimum=0)
        if maximum > computed:
            raise ValueError(
                f"max_order={maximum} exceeds the computed maximum {computed}."
            )
        return landscape.order_summary.loc[
            landscape.order_summary.order <= maximum
        ].copy()
    if verbose:
        logger.info("Fitting nested Walsh-Hadamard models.")
    result = _analyze(
        landscape,
        max_order=2 if max_order is None else max_order,
        n_jobs=n_jobs,
        require_coefficients=False,
        **kwargs,
    )
    return result.order_summary
