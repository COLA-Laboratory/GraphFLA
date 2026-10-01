"""Direct encoding of ASCII sequences without a full per-site DataFrame."""

import numpy as np
import pandas as pd

from .attributes import SequenceAttributes
from .handlers import SequenceHandler
from ._validation import PreparedData


def prepare_sequences(handler, X, f):
    """Return compact sequence data, or None for the general preparation path."""
    if type(handler) is not SequenceHandler:
        return None
    if not isinstance(X, (list, tuple, pd.Series, np.ndarray)) or len(X) == 0:
        return None
    if isinstance(X, np.ndarray) and X.ndim != 1:
        return None
    sequences = list(X)
    if not all(isinstance(s, str) for s in sequences):
        return None
    length = len(sequences[0])
    if length == 0 or any(len(s) != length for s in sequences):
        return None
    alphabet = handler.alphabet
    if not alphabet or len(set(alphabet)) != len(alphabet):
        return None
    if any(len(s) != 1 or not s.isascii() for s in alphabet):
        return None
    joined = "".join(sequences).upper()
    if not joined.isascii() or len(joined) != len(sequences) * length:
        return None
    if not isinstance(f, (list, np.ndarray, pd.Series)) or np.iscomplexobj(f):
        return None
    try:
        fitness = np.asarray(f, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if fitness.ndim != 1 or not np.isfinite(fitness).all():
        return None

    rows = np.frombuffer(joined.encode("ascii"), dtype=np.uint8).reshape(-1, length)
    lookup = np.full(256, 255, dtype=np.uint8)
    lookup[[ord(s) for s in alphabet]] = np.arange(len(alphabet), dtype=np.uint8)
    seen = np.zeros(256, dtype=bool)
    seen[rows.ravel()] = True
    if np.any(lookup[seen] == 255):
        return None
    positions = np.flatnonzero(rows.min(axis=0) != rows.max(axis=0))
    if not positions.size:
        return None
    codes = np.ascontiguousarray(lookup[rows[:, positions]])
    keep = ~pd.DataFrame(codes, copy=False).duplicated().to_numpy()
    if not keep.all():
        codes = codes[keep]
        fitness = fitness[keep]
    columns = [f"pos_{p}" for p in positions]
    maxima = codes.max(axis=0)
    return PreparedData(
        attributes=SequenceAttributes(
            codes, positions, joined[:length], tuple(alphabet)
        ),
        fitness=fitness,
        data_types=dict.fromkeys(columns, "categorical"),
        n_vars=len(positions),
        config_dict={
            i: {"type": "categorical", "max": int(value)}
            for i, value in enumerate(maxima)
        },
        configs_array=codes,
        configs_index=np.flatnonzero(keep),
    )
