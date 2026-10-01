"""Column sources decoded only after construction selects the final nodes."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrameAttributes:
    frame: pd.DataFrame

    def items(self, indices=None):
        frame = self.frame if indices is None else self.frame.iloc[indices]
        for column in frame.columns:
            yield str(column), frame[column].to_numpy(copy=False)


@dataclass(frozen=True)
class SequenceAttributes:
    codes: np.ndarray
    positions: np.ndarray
    background: str
    alphabet: Tuple[str, ...]

    def items(self, indices=None):
        codes = self.codes if indices is None else self.codes[indices]
        alphabet = np.asarray(self.alphabet, dtype=object)
        variable = dict(zip(self.positions, range(len(self.positions))))
        for position, symbol in enumerate(self.background):
            if position in variable:
                values = alphabet[codes[:, variable[position]]].tolist()
            else:
                values = [symbol] * len(codes)
            yield f"pos_{position}", values
