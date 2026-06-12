"""Validation helpers, constants, and the PreparedData container.

Leaf module for the data subpackage: alphabet / allowed-type constants, the
small validation + cleaning utilities shared by the handlers and the
pipeline, and the encoded-dataset container.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd


# ===================================================================
# Constants
# ===================================================================

ALLOWED_DATA_TYPES = {"boolean", "categorical", "ordinal"}
DNA_ALPHABET = ["A", "C", "G", "T"]
RNA_ALPHABET = ["A", "C", "G", "U"]
PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


# ===================================================================
# Data container
# ===================================================================


def configs_series_from_array(
    configs_array: np.ndarray, index: Any = None
) -> pd.Series:
    """Build the per-row configuration tuple ``Series`` from the numeric matrix.

    The result is *byte-for-byte* what :func:`encode_data` historically produced
    eagerly: an ``object`` ``Series`` whose values are ``tuple`` objects of plain
    Python ``int`` (one element per encoded variable), indexed by ``index``
    (the caller's encoded-frame index; defaults to a ``RangeIndex``).

    Construction mirrors the previous in-line code exactly -- a single C-level
    ``ndarray.tolist()`` followed by ``map(tuple, ...)`` -- so the tuples and
    their element types match for every landscape type (boolean / sequence /
    ordinal share this path). It is factored out here so the ``Series`` can be
    produced *lazily*, on first access, rather than always during construction:
    the numeric ``configs_array`` (not this tuple ``Series``) is what every
    neighbour-construction strategy consumes, so most builds never need it.
    """
    config_objects = np.fromiter(
        map(tuple, configs_array.tolist()),
        dtype=object,
        count=len(configs_array),
    )
    return pd.Series(config_objects, index=index, copy=False)


@dataclass(frozen=True)
class PreparedData:
    """Encoded configuration data plus the metadata needed by ``Landscape``.

    The per-row configuration tuple ``Series`` is exposed lazily via the
    :attr:`configs` property (built from :attr:`configs_array` on demand) rather
    than stored eagerly -- it is a pure downstream artifact and the numeric
    ``configs_array`` is what graph construction actually consumes.
    """

    data_for_attributes: pd.DataFrame
    data_types: Dict[str, str]
    n_vars: int
    config_dict: Dict[int, Dict[str, int]]
    configs_array: np.ndarray
    configs_index: Any = None

    @property
    def configs(self) -> pd.Series:
        """The per-row configuration tuple ``Series`` (built on access)."""
        return configs_series_from_array(self.configs_array, self.configs_index)


# ===================================================================
# Helpers (private)
# ===================================================================


def _apply_mask(values, mask):
    """Apply a boolean mask while preserving the input container type."""
    mask = np.asarray(mask, dtype=bool)

    if isinstance(values, (pd.Series, pd.DataFrame)):
        filtered = values.loc[mask]
        filtered.reset_index(drop=True, inplace=True)
        return filtered

    if isinstance(values, np.ndarray):
        return values[mask]

    if isinstance(values, list):
        return [value for value, keep in zip(values, mask) if keep]

    if isinstance(values, tuple):
        return tuple(value for value, keep in zip(values, mask) if keep)

    return np.asarray(values)[mask]


def _validate_bitstrings_fast(bitstrings, bit_length):
    """Vectorized validation/decoding of a list of equal-length 0/1 bitstrings.

    Returns the flat ``uint8`` array of ASCII byte codes for
    ``"".join(bitstrings)`` when *every* string has length ``bit_length`` and
    contains only ``'0'``/``'1'`` (ASCII); otherwise returns ``None`` to signal
    that the caller should fall back to the element-by-element loop (which
    reproduces the exact original error: the first offending index and, for a
    bad character, the ``invalid_chars`` set).

    All elements are assumed to already be Python ``str`` (the caller checks
    this up-front via ``isinstance``).
    """
    joined = "".join(bitstrings)
    # Length mismatch makes the flat join un-reshapeable; defer to the slow loop.
    if len(joined) != bit_length * len(bitstrings):
        return None
    try:
        byte_codes = np.frombuffer(joined.encode("ascii"), dtype=np.uint8)
    except UnicodeEncodeError:
        # Non-ASCII: let the slow loop raise the original error.
        return None
    if byte_codes.size:
        # ord('0') == 48, ord('1') == 49; anything else is invalid.
        uniq = np.unique(byte_codes)
        if uniq[0] < 48 or uniq[-1] > 49:
            return None
    return byte_codes


def _warn_if_missing(
    X_in: pd.DataFrame,
    f_in: pd.Series,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Detect missing values in X or f and warn; do not alter rows.

    Rows with NaNs are left in place so callers can impute, drop, or
    otherwise handle them explicitly. :func:`encode_data` raises if NaNs
    remain at encoding time.
    """
    if verbose:
        print(" - Checking for missing values...")

    X_out = X_in

    if X_out.isnull().values.any():
        nan_rows_X = X_out.isnull().any(axis=1)
        n_nan_X = int(nan_rows_X.sum())
        nan_cols = X_out.columns[X_out.isnull().any()].tolist()
        msg = (
            f"Found {n_nan_X} row(s) with NaN in feature matrix X (affected columns "
            f"include {nan_cols[:15]}{'...' if len(nan_cols) > 15 else ''}). "
            "Rows were not dropped; impute or drop NaNs before building, or "
            "encode_data will raise."
        )
        warnings.warn(msg, UserWarning)
        if verbose:
            print(f"   - {msg}")

    f_aligned = f_in.loc[X_out.index]
    nan_mask_f = f_aligned.isnull()
    if nan_mask_f.any():
        n_missing_f = int(nan_mask_f.sum())
        msg = (
            f"Found {n_missing_f} row(s) with NaN in fitness f (for rows in X). "
            "Rows were not dropped; fix fitness values before building, or "
            "encode_data will raise."
        )
        warnings.warn(msg, UserWarning)
        if verbose:
            print(f"   - {msg}")

    return X_out, f_aligned


def _drop_duplicates(
    X_in: pd.DataFrame,
    f_in: pd.Series,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove duplicate configurations, keeping the first occurrence."""
    if verbose:
        print(" - Handling duplicate configurations...")

    initial_count = len(X_in)
    mask_duplicates = X_in.duplicated(keep="first")
    num_removed = mask_duplicates.sum()

    if num_removed > 0:
        if verbose:
            print(
                f"   - Found {num_removed} duplicate configurations in X. Keeping first occurrence."
            )
        X_out = X_in.loc[~mask_duplicates]
        f_out = f_in.loc[~mask_duplicates]
    else:
        if verbose:
            print("   - No duplicate configurations found.")
        X_out, f_out = X_in, f_in

    final_count = len(X_out)
    if verbose and initial_count != final_count:
        print(
            f"   - Duplicate handling complete. Kept {final_count}/{initial_count} configurations."
        )

    return X_out, f_out


def _invariant_columns(X: pd.DataFrame) -> List[str]:
    """Return the columns of *X* that take at most one distinct value.

    Exactly equivalent to ``X.nunique().index[X.nunique() <= 1].tolist()`` --
    the column names (in ``X``-column order) for which ``Series.nunique()``
    (which drops NaN) is ``<= 1`` -- but avoids the full per-column
    ``factorize``/``unique`` machinery for *categorical* columns. For those the
    distinct-non-NaN count is read directly off the integer category codes (NaN
    is code ``-1``), which is an O(n) min/max scan instead of an O(n log n)
    hash/sort and is dramatically cheaper when there are many wide categorical
    columns (the high-dimensional sequence case). Non-categorical columns fall
    back to ``Series.nunique()`` so the result is identical for every dtype.
    """
    invariant = []
    for col in X.columns:
        series = X[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            codes = series.cat.codes.to_numpy()
            if codes.size == 0:
                invariant.append(col)
                continue
            mn = codes.min()
            if mn >= 0:
                # No NaN present: invariant iff a single distinct code.
                if mn == codes.max():
                    invariant.append(col)
            else:
                # NaN present (code -1): count distinct among the non-NaN codes,
                # matching ``nunique(dropna=True)``.
                nonneg = codes[codes >= 0]
                if nonneg.size == 0 or nonneg.min() == nonneg.max():
                    invariant.append(col)
        elif series.nunique() <= 1:
            invariant.append(col)
    return invariant


def _build_config_dict(
    data_types: Dict[str, str], data_encoded: pd.DataFrame
) -> Dict[int, Dict[str, int]]:
    """Describe the encoded search space for neighborhood generation."""
    config_dict = {}
    max_encoded = data_encoded.max()
    for i, (col, dtype) in enumerate(data_types.items()):
        max_val = 1 if dtype == "boolean" else max_encoded[col]
        config_dict[i] = {"type": dtype, "max": int(max_val)}
    return config_dict
