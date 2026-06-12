from __future__ import annotations

from .landscape import Landscape


class OrdinalLandscape(Landscape):
    """A specialized landscape class for ordinal configuration spaces.

    Each configuration is a vector of *ordered* discrete values — for
    example, dose levels ``{0, 1, 2, 3}``, hyperparameter tiers, Likert
    scales, or any discrete-but-ordered factor.

    Unlike a categorical landscape, the neighborhood of an ordinal
    configuration is **±1 step on the ordinal scale at one position** —
    i.e., Manhattan-distance-1 on each axis — and distances to the global
    optimum are computed using Manhattan rather than Hamming distance.
    This matches the standard definition used in the ordinal-landscape
    literature.

    The constructor takes no shape arguments: the number of levels for
    each variable is auto-detected from the data during
    :meth:`build_from_data` (it follows ``pandas.Categorical(...).codes``,
    so the natural order of integer-coded values is preserved).

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape
        seeks higher fitness values. If False, it seeks lower values.

    Notes
    -----
    - The ``"active"`` neighborhood strategy is the default for ordinal
      landscapes (via :attr:`_default_neighborhood_strategy`); it calls the
      :class:`~graphfla._neighbors.OrdinalNeighborGenerator`, which enforces
      the correct **±1 step (Manhattan-1)** semantics. The Hamming-based
      ``"pairwise"``/``"broadcast"`` strategies would instead treat *any*
      single-position change as adjacent — including pairs many steps apart
      on the ordinal scale — so :meth:`build_from_data` warns if you request
      them on an ordinal landscape.
    - If your data uses non-integer ordered values (e.g., strings such as
      ``"low"``, ``"mid"``, ``"high"``), pandas will fall back to
      lexicographic sorting, which is almost never what you want. Either
      pre-encode the values as integers or pass each column as a
      ``pandas.Categorical(col, ordered=True, categories=[...])`` with
      the explicit order.
    - For a *mixed* landscape combining ordinal variables with boolean
      or categorical ones, use the generic
      :class:`~graphfla.landscape.Landscape` with ``kind="default"`` and
      supply an explicit ``data_types`` dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ...     "dose_A": [0, 0, 1, 1, 2, 2],
    ...     "dose_B": [0, 1, 0, 1, 0, 1],
    ... })
    >>> f = pd.Series([0.1, 0.3, 0.5, 0.4, 0.7, 0.6])
    >>> landscape = OrdinalLandscape(maximize=True).build_from_data(X, f, verbose=False)
    """

    #: Default to "active" so OrdinalNeighborGenerator enforces ±1-step
    #: (Manhattan-1) adjacency; the base build_from_data consults this attr.
    _default_neighborhood_strategy = "active"

    def __init__(self, maximize: bool = True):
        """Initialize an ordinal landscape.

        Parameters
        ----------
        maximize : bool, default=True
            Determines the optimization direction.
        """
        super().__init__(kind="ordinal", maximize=maximize)
