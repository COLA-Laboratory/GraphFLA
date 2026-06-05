from .landscape import Landscape
from typing import List


class SequenceLandscape(Landscape):
    """A specialized landscape class for sequence-based configuration spaces.

    This class represents fitness landscapes where each configuration is a sequence
    of characters from a defined alphabet. Examples include DNA/RNA sequences,
    protein sequences, or any other discrete sequence space.

    Parameters
    ----------
    alphabet : list[str]
        The set of valid characters that can appear in the sequences.
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.
    """

    def __init__(
        self,
        alphabet: List[str],
        maximize: bool = True,
    ):
        from .._data import SequenceHandler
        from .._neighbors import SequenceNeighborGenerator

        handler = SequenceHandler(alphabet)
        neighbor_generator = SequenceNeighborGenerator(len(alphabet))

        # Register the alphabet-specific strategies on this instance only (no
        # global class-registry mutation), via the base constructor.
        type_key = f"sequence_{id(alphabet)}"
        super().__init__(
            type=type_key,
            maximize=maximize,
            input_handler=handler,
            neighbor_generator=neighbor_generator,
        )
        self.alphabet = alphabet
