from __future__ import annotations

import logging
from typing import List, Optional

from .landscape import Landscape
from ..exceptions import InvalidParameterError

logger = logging.getLogger(__name__)


class _DeferredSequenceStrategy:
    """Placeholder registered when a :class:`SequenceLandscape` is created without
    an alphabet. The real handler/generator replace it in ``build_from_data``
    once the alphabet has been inferred from the data; touching it before then
    is a bug, so every attribute access raises."""

    def __getattr__(self, name):
        raise RuntimeError(
            "SequenceLandscape alphabet has not been inferred yet; call "
            "build_from_data() before using the landscape."
        )


class SequenceLandscape(Landscape):
    """A specialized landscape class for sequence-based configuration spaces.

    Each configuration is a sequence of symbols from a shared alphabet (DNA/RNA,
    protein, or any discrete-symbol sequence).

    Parameters
    ----------
    alphabet : list[str], optional
        The valid symbols sequences may contain. If omitted (``None``, the
        default), the alphabet is **inferred automatically from the data** when
        the landscape is built -- the sorted set of symbols observed across all
        positions. Pass it explicitly only to pin a particular symbol set/order
        (e.g. to include symbols absent from this specific sample).
    maximize : bool, default=True
        Optimization direction (True seeks higher fitness).
    """

    def __init__(
        self,
        alphabet: Optional[List[str]] = None,
        maximize: bool = True,
        kind: str = "sequence",
    ):
        from .._data import SequenceHandler
        from .._neighbors import SequenceNeighborGenerator

        if alphabet is None:
            # Defer: register placeholders so the base registry check passes; the
            # real strategies are installed from the data in build_from_data().
            handler = _DeferredSequenceStrategy()
            neighbor_generator = _DeferredSequenceStrategy()
        else:
            alphabet = list(alphabet)
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
        # ``None`` until inferred; build_from_data fills it in for the deferred case.
        self.alphabet = alphabet

    def build_from_data(self, X, f, **kwargs):
        """Build the landscape, inferring the alphabet from *X* when it was not
        supplied at construction (see :class:`SequenceLandscape`)."""
        if self.alphabet is None:
            from .._data import SequenceHandler
            from .._neighbors import SequenceNeighborGenerator

            alphabet = self._infer_alphabet(X)
            self.alphabet = alphabet
            self._input_handlers["sequence"] = SequenceHandler(alphabet)
            self._neighbor_generators["sequence"] = SequenceNeighborGenerator(
                len(alphabet)
            )
            if kwargs.get("verbose", True):
                logger.info(
                    f"Inferred sequence alphabet ({len(alphabet)} symbols): "
                    f"{' '.join(alphabet)}"
                )
        return super().build_from_data(X, f, **kwargs)

    @staticmethod
    def _infer_alphabet(X) -> List[str]:
        """The sorted set of symbols present in *X* -- characters for sequence
        strings, cell values for a per-position table. Mirrors how
        ``SequenceHandler`` interprets each input form."""
        import numpy as np
        import pandas as pd

        syms: set = set()

        def _add_value(v):
            if not pd.isna(v):
                syms.add(str(v))

        def _add_chars(v):
            if not pd.isna(v):
                syms.update(str(v))

        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                for v in pd.unique(X[col].to_numpy()):
                    _add_value(v)
        elif isinstance(X, pd.Series):
            for v in X.to_numpy():
                _add_chars(v)
        elif isinstance(X, np.ndarray):
            if X.ndim >= 2:
                for v in np.unique(X):
                    _add_value(v)
            else:
                for v in X:
                    _add_chars(v)
        elif isinstance(X, (list, tuple)):
            for row in X:
                if isinstance(row, (list, tuple, np.ndarray)):
                    for v in row:
                        _add_value(v)
                else:
                    _add_chars(row)
        else:
            raise InvalidParameterError(
                f"Cannot infer a sequence alphabet from X of type "
                f"{type(X).__name__!r}; pass alphabet= explicitly."
            )

        syms.discard("")
        if not syms:
            raise InvalidParameterError(
                "Could not infer a non-empty alphabet from the data; pass "
                "alphabet= explicitly."
            )
        return sorted(syms)
