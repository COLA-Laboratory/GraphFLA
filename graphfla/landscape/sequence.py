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

        type_key = f"sequence_{id(alphabet)}"
        Landscape.register_input_handler(type_key, handler)
        Landscape.register_neighbor_generator(type_key, neighbor_generator)

        super().__init__(type=type_key, maximize=maximize)
        self.alphabet = alphabet
