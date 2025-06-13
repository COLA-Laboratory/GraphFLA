from .sequence import Landscape


class BooleanLandscape(Landscape):
    """A specialized landscape class for boolean configuration spaces.

    This class represents fitness landscapes where each configuration is a bit string
    (sequence of 0s and 1s), which is common in many optimization problems like
    NK landscapes, MAXSAT, and binary encoding of combinatorial problems.

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> X_data = pd.DataFrame({'var_0': [0, 0, 1, 1], 'var_1': [0, 1, 0, 1]})
    >>> f_data = pd.Series([1.0, 2.0, 3.0, 2.5])
    >>> landscape = BooleanLandscape().build_from_data(X_data, f_data)
    """

    def __init__(self, maximize: bool = True):
        """Initialize a boolean landscape.

        Parameters
        ----------
        maximize : bool, default=True
            Determines the optimization direction.
        """
        super().__init__(type="boolean", maximize=maximize)
