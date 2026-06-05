from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

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
    - If your data uses non-integer ordered values (e.g., strings such as
      ``"low"``, ``"mid"``, ``"high"``), pandas will fall back to
      lexicographic sorting, which is almost never what you want. Either
      pre-encode the values as integers or pass each column as a
      ``pandas.Categorical(col, ordered=True, categories=[...])`` with
      the explicit order.
    - For a *mixed* landscape combining ordinal variables with boolean
      or categorical ones, use the generic
      :class:`~graphfla.landscape.Landscape` with ``type="default"`` and
      supply an explicit ``data_types`` dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ...     "dose_A": [0, 0, 1, 1, 2, 2],
    ...     "dose_B": [0, 1, 0, 1, 0, 1],
    ... })
    >>> f = pd.Series([0.1, 0.3, 0.5, 0.4, 0.7, 0.6])
    >>> landscape = OrdinalLandscape(maximize=True)
    >>> landscape.build_from_data(X, f, verbose=False)
    """

    def __init__(self, maximize: bool = True):
        """Initialize an ordinal landscape.

        Parameters
        ----------
        maximize : bool, default=True
            Determines the optimization direction.
        """
        super().__init__(type="ordinal", maximize=maximize)

    def build_from_data(
        self,
        X: Any,
        f: Union[pd.Series, list, np.ndarray],
        data_types: Optional[Dict[str, str]] = None,
        epsilon: float = 0,
        calculate_basins: bool = False,
        calculate_paths: bool = False,
        calculate_distance: bool = False,
        calculate_neighbor_fit: bool = False,
        tau: Optional[float] = None,
        filter_mode: str = "any",
        n_edit: int = 1,
        neighborhood_strategy: str = "active",
        verbose: Optional[bool] = True,
        accessible_paths_max_configs: int = 200_000,
        force_accessible_paths: bool = False,
    ) -> None:
        """Construct the landscape graph and properties from configuration data.

        This is a thin wrapper around
        :meth:`Landscape.build_from_data` that fixes
        ``neighborhood_strategy="active"`` by default.

        Why the override? The ``"pairwise"`` and ``"broadcast"`` strategies
        rely on **Hamming distance** to identify neighbors, which on an
        ordinal landscape would treat every pair differing at one position
        as adjacent — including pairs many steps apart on the ordinal
        scale. The ``"active"`` strategy, in contrast, calls the
        :class:`~graphfla._neighbors.OrdinalNeighborGenerator`, which
        enforces the correct **±1 step (Manhattan-1)** semantics.

        Users who explicitly need a Hamming-style neighborhood on ordinal
        data can still pass ``neighborhood_strategy="pairwise"`` or
        ``"broadcast"``; the parameter is preserved for full flexibility.
        See :meth:`Landscape.build_from_data` for the complete parameter
        reference.
        """
        return super().build_from_data(
            X=X,
            f=f,
            data_types=data_types,
            epsilon=epsilon,
            calculate_basins=calculate_basins,
            calculate_paths=calculate_paths,
            calculate_distance=calculate_distance,
            calculate_neighbor_fit=calculate_neighbor_fit,
            tau=tau,
            filter_mode=filter_mode,
            n_edit=n_edit,
            neighborhood_strategy=neighborhood_strategy,
            verbose=verbose,
            accessible_paths_max_configs=accessible_paths_max_configs,
            force_accessible_paths=force_accessible_paths,
        )
