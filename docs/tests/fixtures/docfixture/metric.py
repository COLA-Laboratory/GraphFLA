def metric(
    values, /, threshold: float = 0.5, *, label: str = "base", **kwargs
) -> float:
    """A summary that belongs to the API.

    EXTENDED_DESCRIPTION_SENTINEL belongs in the detailed explanation.

    .. deprecated:: 2.0
        Use a newer entry point when available.

    Parameters
    ----------
    values : list[float]
        SOURCE_PARAMETER_V1.
    threshold : float, optional
        Threshold for evaluation.
    label : str, optional
        Label attached to the result.
    **kwargs
        Additional evaluation settings.

    Other Parameters
    ----------------
    scale : float
        Optional scaling factor.

    Returns
    -------
    score : float
        SOURCE_RETURN_V1.

    Raises
    ------
    ValueError
        If values are empty.

    Warns
    -----
    RuntimeWarning
        If the scale is extreme.

    Warnings
    --------
    WARNING_SENTINEL: interpret thresholds carefully.

    See Also
    --------
    stream : Yield individual values.

    Notes
    -----
    NOTE_SENTINEL follows the definition in [1]_.
    Inline math :math:`x^2` remains mathematical notation.

    .. math::

        y = x^2 + 1

    References
    ----------
    .. [1] REFERENCE_SENTINEL. A fixture citation.

    Examples
    --------
    >>> 2 + 3
    5
    """
    return sum(values)


def stream(limit: int = 3):
    """Yield values from a bounded stream.

    Parameters
    ----------
    limit : int, optional
        Number of values.

    Yields
    ------
    int
        YIELD_SENTINEL.

    Receives
    --------
    int
        RECEIVE_SENTINEL.
    """
    yield from range(limit)
