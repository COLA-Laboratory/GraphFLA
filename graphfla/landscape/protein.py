from .sequence import SequenceLandscape

PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


class ProteinLandscape(SequenceLandscape):
    """A specialized landscape class for protein sequence configuration spaces.

    This class represents fitness landscapes where each configuration is a protein sequence
    using the standard protein alphabet (20 amino acids).

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.
    """

    def __init__(self, maximize: bool = True):
        super().__init__(
            alphabet=PROTEIN_ALPHABET,
            maximize=maximize,
        )
