from .sequence import SequenceLandscape

DNA_ALPHABET = ["A", "C", "G", "T"]


class DNALandscape(SequenceLandscape):
    """A specialized landscape class for DNA sequence configuration spaces.

    This class represents fitness landscapes where each configuration is a DNA sequence
    using the standard DNA alphabet (A, C, G, T).

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.
    """

    def __init__(self, maximize: bool = True):
        super().__init__(alphabet=DNA_ALPHABET, maximize=maximize)
