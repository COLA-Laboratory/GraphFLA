from __future__ import annotations

from typing import List

from .landscape import Landscape


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
        kind: str = "sequence",
    ):
        from .._data import SequenceHandler
        from .._neighbors import SequenceNeighborGenerator

        handler = SequenceHandler(alphabet)
        neighbor_generator = SequenceNeighborGenerator(len(alphabet))

        # Register the alphabet-specific strategies on this instance only (no
        # global class-registry mutation), via the base constructor. All sequence
        # landscapes share the constant 'sequence' registry key -- the per-instance
        # registry isolates the alphabet -- while ``kind`` carries the semantic
        # identity ('dna'/'rna'/'protein'/'sequence') used by the discriminators.
        super().__init__(
            kind=kind,
            maximize=maximize,
            input_handler=handler,
            neighbor_generator=neighbor_generator,
            strategy_key="sequence",
        )
        self.alphabet = alphabet
