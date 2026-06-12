from __future__ import annotations

from .sequence import SequenceLandscape
from .._data import RNA_ALPHABET


class RNALandscape(SequenceLandscape):
    """A specialized landscape class for RNA sequence configuration spaces.

    This class represents fitness landscapes where each configuration is an RNA sequence
    using the standard RNA alphabet (A, C, G, U).

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.
    """

    def __init__(self, maximize: bool = True):
        super().__init__(alphabet=RNA_ALPHABET, maximize=maximize, kind="rna")
