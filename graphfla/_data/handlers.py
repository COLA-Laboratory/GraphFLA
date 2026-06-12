"""Input handlers: validate and standardize raw configuration data.

Strategy classes (boolean / sequence / ordinal / mixed-default) plus the
per-type parsing routines they delegate to.
"""

from typing import Any, Dict, List, Tuple, Union, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ._validation import ALLOWED_DATA_TYPES, _validate_bitstrings_fast
import logging

logger = logging.getLogger(__name__)


# ===================================================================
# Input handlers
# ===================================================================


@runtime_checkable
class InputHandler(Protocol):
    """Protocol defining the interface for type-specific data preparation."""

    def prepare(
        self, X: Any, f: Union[pd.Series, list, np.ndarray], verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """
        Prepare input data for a specific landscape type.

        Parameters
        ----------
        X : Any
            Input configuration data.
        f : Union[pd.Series, list, np.ndarray]
            Fitness values corresponding to configurations.
        verbose : bool
            Whether to print processing information.

        Returns
        -------
        X_processed : DataFrame
            Processed feature data.
        f_processed : Series
            Processed fitness values.
        data_types : dict
            Dictionary mapping column names to data types.
        dimension : int
            Dimensionality of the data (e.g., sequence length, bit length).
        """
        ...


class BooleanHandler:
    """Input handler for boolean (bitstring) configuration spaces."""

    def prepare(
        self,
        X: Union[List[Any], pd.DataFrame, np.ndarray, pd.Series],
        f: Union[pd.Series, list, np.ndarray],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """Prepare boolean data."""
        X_df, bool_data_types, bit_len = _parse_boolean_input(
            X_input=X, verbose=verbose
        )

        if isinstance(f, (list, np.ndarray)):
            f_series = pd.Series(f, name="fitness")
        elif isinstance(f, pd.Series):
            f_series = f.copy()
            f_series.name = "fitness"
        else:
            raise TypeError(
                f"Input f must be a pandas Series, list, or numpy ndarray, got {type(f)}."
            )

        X_df.reset_index(drop=True, inplace=True)
        f_series.reset_index(drop=True, inplace=True)
        f_series.index = X_df.index

        return X_df, f_series, bool_data_types, bit_len


class SequenceHandler:
    """Input handler for discrete-alphabet sequence spaces."""

    def __init__(self, alphabet: List[str]):
        """Initialize with the appropriate alphabet."""
        self.alphabet = alphabet

    def prepare(
        self,
        X: Union[List[str], pd.Series, np.ndarray, pd.DataFrame],
        f: Union[pd.Series, list, np.ndarray],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """Prepare sequence data."""
        self._validate_alphabet(X)

        X_df, seq_data_types, seq_len = _parse_sequence_input(
            X_input=X,
            alphabet=self.alphabet,
            class_name=f"Sequence ({self.alphabet[0]}{self.alphabet[-1]})",
            validated=True,
            verbose=verbose,
        )

        if isinstance(f, (list, np.ndarray)):
            f_series = pd.Series(f, name="fitness")
        elif isinstance(f, pd.Series):
            f_series = f.copy()
            f_series.name = "fitness"
        else:
            raise TypeError(
                f"Input f must be a pandas Series, list, or numpy ndarray, got {type(f)}."
            )

        X_df.reset_index(drop=True, inplace=True)
        f_series.reset_index(drop=True, inplace=True)
        f_series.index = X_df.index

        return X_df, f_series, seq_data_types, seq_len

    def _validate_alphabet(
        self, X: Union[List[str], pd.Series, np.ndarray, pd.DataFrame]
    ) -> None:
        """Raise if any value in X is not among the specified alphabet.

        Only genuinely out-of-alphabet characters raise. Partial coverage --
        valid data that exercises only some of the alphabet (a DMS subset, a
        reduced ProteinGym landscape, a toy DNA example) -- is normal and is NOT
        flagged.
        """
        valid_chars = set(self.alphabet)

        if isinstance(X, (list, tuple, pd.Series)):
            if self._used_chars_fast(X, valid_chars) is not None:
                return  # fast path: every character is within the alphabet
            for idx, seq in enumerate(X):
                if isinstance(seq, str):
                    invalid_chars = set(seq.upper()) - valid_chars
                    if invalid_chars:
                        raise ValueError(
                            f"Input X values at index {idx} contain {', '.join(invalid_chars)}, "
                            f"which is not among specified alphabet: {self.alphabet}"
                        )

        elif isinstance(X, pd.DataFrame):
            for col in X.columns:
                for idx, val in enumerate(X[col]):
                    if val is not None and str(val).upper() not in valid_chars:
                        raise ValueError(
                            f"Input X values at position ({idx}, {col}) contain '{val}', "
                            f"which is not among specified alphabet: {self.alphabet}"
                        )

        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                if self._used_chars_fast(X, valid_chars) is not None:
                    return
                for idx, seq in enumerate(X):
                    if isinstance(seq, str):
                        invalid_chars = set(seq.upper()) - valid_chars
                        if invalid_chars:
                            raise ValueError(
                                f"Input X values at index {idx} contain {', '.join(invalid_chars)}, "
                                f"which is not among specified alphabet: {self.alphabet}"
                            )
            else:
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        val = X[i, j]
                        if val is not None and str(val).upper() not in valid_chars:
                            raise ValueError(
                                f"Input X values at position ({i}, {j}) contain '{val}', "
                                f"which is not among specified alphabet: {self.alphabet}"
                            )

    @staticmethod
    def _used_chars_fast(X, valid_chars):
        """Return the set of uppercased characters used across a 1-D string input.

        This is a vectorized equivalent of the per-sequence ``used_chars``
        accumulation in :meth:`_validate_alphabet` for the common case where
        *every* element is a (Python ``str``) sequence and all characters are
        ASCII and within the alphabet.

        Returns
        -------
        set or None
            The set of distinct uppercased characters when the fast path
            applies and every character is valid; otherwise ``None`` to signal
            that the caller should fall back to the element-by-element loop
            (which reproduces the exact error message, non-str skipping, or
            non-ASCII handling).
        """
        if isinstance(X, pd.Series):
            values = X.to_numpy()
        else:
            values = X
        try:
            # join raises on mixed (non-str) input; fall back to the slow loop.
            joined = "".join(values)
        except TypeError:
            return None
        if not joined:
            return set()
        try:
            byte_codes = np.frombuffer(joined.upper().encode("ascii"), dtype=np.uint8)
        except UnicodeEncodeError:
            # Non-ASCII: defer to the slow loop for the original error message.
            return None
        # Distinct bytes via a 256-slot boolean scatter: one linear pass, no
        # sort -- faster than np.unique on the (potentially huge) byte buffer.
        seen = np.zeros(256, dtype=bool)
        seen[byte_codes] = True
        used = {chr(code) for code in np.flatnonzero(seen).tolist()}
        if used - valid_chars:
            # Invalid char present; let the slow loop pinpoint it and raise.
            return None
        return used


class OrdinalHandler:
    """Input handler for ordinal (integer-coded ordered) configuration spaces.

    Every column of the input is treated as an *ordered* discrete variable;
    the natural neighborhood is a ±1 step on the ordinal scale (handled by
    :class:`OrdinalNeighborGenerator`).
    """

    def prepare(
        self,
        X: Union[List[Any], pd.DataFrame, np.ndarray, pd.Series],
        f: Union[pd.Series, list, np.ndarray],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """Prepare ordinal data."""
        X_df, ord_data_types, n_vars = _parse_ordinal_input(
            X_input=X, verbose=verbose
        )

        if isinstance(f, (list, np.ndarray)):
            f_series = pd.Series(f, name="fitness")
        elif isinstance(f, pd.Series):
            f_series = f.copy()
            f_series.name = "fitness"
        else:
            raise TypeError(
                f"Input f must be a pandas Series, list, or numpy ndarray, got {type(f)}."
            )

        X_df.reset_index(drop=True, inplace=True)
        f_series.reset_index(drop=True, inplace=True)
        f_series.index = X_df.index

        return X_df, f_series, ord_data_types, n_vars


class DefaultHandler:
    """Input handler for mixed data types (boolean / categorical / ordinal)."""

    def prepare(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        f: Union[pd.Series, list, np.ndarray],
        data_types: Dict[str, str],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """
        Prepare data with explicit data types.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input configuration data.
        f : Series, list, or ndarray
            Fitness values.
        data_types : dict
            Dictionary mapping column names to data types.
        verbose : bool
            Whether to print processing information.

        Returns
        -------
        X_processed : DataFrame
        f_processed : Series
        data_types_validated : dict
        n_vars : int
        """
        if isinstance(X, np.ndarray):
            try:
                columns = [f"var_{i}" for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=columns)
            except Exception as e:
                raise TypeError(
                    f"Could not convert input X (ndarray) to DataFrame: {e}"
                )
            # Remap integer data_types keys (e.g. {0: 'boolean'}) to the var_i
            # names so callers needn't know the internal column naming.
            if isinstance(data_types, dict) and data_types and all(
                isinstance(k, (int, np.integer)) for k in data_types.keys()
            ):
                data_types = {f"var_{k}": v for k, v in data_types.items()}
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            raise TypeError(
                f"Input X must be a pandas DataFrame or numpy ndarray, got {type(X)}."
            )

        if isinstance(f, (list, np.ndarray)):
            try:
                f_series = pd.Series(f, name="fitness")
            except Exception as e:
                raise TypeError(
                    f"Could not convert input f (list/ndarray) to Series: {e}"
                )
        elif isinstance(f, pd.Series):
            f_series = f.copy()
            f_series.name = "fitness"
        else:
            raise TypeError(
                f"Input f must be a pandas Series, list, or numpy ndarray, got {type(f)}."
            )

        data_types_validated = self._validate_data_types(X_df, data_types, verbose)

        X_df.reset_index(drop=True, inplace=True)
        f_series.reset_index(drop=True, inplace=True)
        f_series.index = X_df.index
        return X_df, f_series, data_types_validated, len(data_types_validated)

    def _validate_data_types(
        self, X: pd.DataFrame, data_types: Dict[str, str], verbose: bool
    ) -> Dict[str, str]:
        """Validate the data_types dictionary against X's columns."""
        if verbose:
            logger.info(" - Validating data types dictionary...")

        if not isinstance(data_types, dict):
            raise TypeError(f"data_types must be a dictionary, got {type(data_types)}.")

        x_cols = set(X.columns)
        dt_keys = set(data_types.keys())

        if x_cols != dt_keys:
            missing_in_dt = x_cols - dt_keys
            extra_in_dt = dt_keys - x_cols
            error_msg = "Mismatch between X columns and data_types keys:"
            if missing_in_dt:
                error_msg += f"\n  - Columns in X missing from data_types: {sorted(list(missing_in_dt))}"
            if extra_in_dt:
                error_msg += f"\n  - Keys in data_types not found in X columns: {sorted(list(extra_in_dt))}"
            raise ValueError(error_msg)

        invalid_types = {}
        for key, type_val in data_types.items():
            if type_val not in ALLOWED_DATA_TYPES:
                invalid_types[key] = type_val

        if invalid_types:
            raise ValueError(
                f"Invalid data types found in data_types dictionary: {invalid_types}. "
                f"Allowed types are: {ALLOWED_DATA_TYPES}."
            )

        validated_dt = {col: data_types[col] for col in X.columns}

        if verbose:
            logger.info("   - Data types dictionary validation successful.")

        return validated_dt


def _parse_boolean_input(
    X_input: Union[List[Any], pd.DataFrame, np.ndarray, pd.Series],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str], int]:
    """Validate and standardise boolean input into a DataFrame of 0/1 columns.

    Handles various input formats:
    - List/Series/Array of bitstrings (e.g., ``['010', '110']``)
    - List/Tuple of Lists/Tuples of 0/1 (e.g., ``[[0, 1, 0], [1, 1, 0]]``)
    - Pandas DataFrame or NumPy array containing 0/1 or True/False.

    Returns
    -------
    tuple[DataFrame, dict[str, str], int]
        Standardised DataFrame, data-types dictionary, and bit length.
    """
    if verbose:
        logger.info("Preparing Boolean input...")

    if not hasattr(X_input, "__len__") or len(X_input) == 0:
        raise ValueError("Input configuration data `X` cannot be empty.")

    X_df = None
    bit_length = -1

    is_sequence_of_strings = False
    is_sequence_of_sequences = False
    try:
        first_element = (
            X_input[X_input.columns[0]]
            if isinstance(X_input, pd.DataFrame)
            else X_input[0]
        )
        if isinstance(first_element, str):
            is_sequence_of_strings = True
        elif isinstance(first_element, (list, tuple, np.ndarray)):
            if all(isinstance(val, (int, bool, np.integer)) for val in first_element):
                is_sequence_of_sequences = True
        elif isinstance(X_input, np.ndarray) and X_input.dtype.kind in ("U", "S"):
            is_sequence_of_strings = True
    except (IndexError, TypeError):
        pass

    # --- Format 1: bitstrings ---
    if is_sequence_of_strings:
        if verbose:
            logger.info("Detected bitstring sequence format input.")
        bitstrings = list(X_input)
        if not all(isinstance(s, str) for s in bitstrings):
            raise TypeError(
                "If X is a sequence of strings, all elements must be strings."
            )
        if not bitstrings:
            raise ValueError("Input bitstring sequence is empty.")

        bit_length = len(bitstrings[0])
        if bit_length == 0:
            raise ValueError("Bitstrings cannot be empty.")

        # Fast path: validate/decode uniform-length 0/1 bitstrings in one
        # vectorized pass; None means fall back to the loop for the exact error.
        byte_codes = _validate_bitstrings_fast(bitstrings, bit_length)
        if byte_codes is None:
            for i, bstr in enumerate(bitstrings):
                if len(bstr) != bit_length:
                    raise ValueError(
                        f"All bitstrings must have the same length (expected {bit_length}, got {len(bstr)} for string {i})."
                    )
                if not all(c in "01" for c in bstr):
                    invalid_chars = set(bstr) - set("01")
                    raise ValueError(
                        f"Bitstring {i} contains invalid characters: {invalid_chars}. Only '0' and '1' allowed."
                    )
            byte_codes = np.frombuffer(
                "".join(bitstrings).encode("ascii"), dtype=np.uint8
            )

        bit_array = byte_codes.reshape(len(bitstrings), bit_length) - ord("0")
        X_df = pd.DataFrame(bit_array, copy=False)

    # --- Format 2: sequences of 0/1 ---
    elif is_sequence_of_sequences:
        if verbose:
            logger.info("Detected sequence of 0/1 lists/tuples format input.")
        sequences = list(X_input)
        if not sequences:
            raise ValueError("Input sequence is empty.")

        try:
            processed_sequences = [[int(val) for val in seq] for seq in sequences]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert inner sequences to integers: {e}") from e

        bit_length = len(processed_sequences[0])
        if bit_length == 0:
            raise ValueError("Inner sequences cannot be empty.")

        data = []
        for i, seq in enumerate(processed_sequences):
            if len(seq) != bit_length:
                raise ValueError(
                    f"All inner sequences must have the same length (expected {bit_length}, got {len(seq)} for sequence {i})."
                )
            if not all(bit in [0, 1] for bit in seq):
                invalid_vals = set(seq) - {0, 1}
                raise ValueError(
                    f"Sequence {i} contains invalid values: {invalid_vals}. Only 0 or 1 allowed."
                )
            data.append(seq)

        X_df = pd.DataFrame(data)

    # --- Format 3: DataFrame or ndarray ---
    elif isinstance(X_input, (pd.DataFrame, np.ndarray)):
        if verbose:
            logger.info("Detected DataFrame/ndarray format input.")

        if isinstance(X_input, np.ndarray):
            try:
                X_df = pd.DataFrame(X_input)
            except Exception as e:
                raise TypeError(f"Could not convert NumPy array to DataFrame: {e}") from e
        else:
            X_df = X_input.copy()

        if X_df.empty:
            raise ValueError("Input DataFrame/ndarray is empty.")

        bit_length = X_df.shape[1]
        if bit_length == 0:
            raise ValueError("Input DataFrame/ndarray cannot have zero columns.")

        try:
            X_df = X_df.replace({True: 1, False: 0})
            X_df = X_df.astype(int)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert DataFrame content to integer 0/1: {e}. Ensure input contains only boolean-like values (0, 1, True, False)."
            )

        if not X_df.isin([0, 1]).all().all():
            problem_val = None
            for col in X_df.columns:
                bad_rows = X_df[~X_df[col].isin([0, 1])]
                if not bad_rows.empty:
                    problem_val = bad_rows.iloc[0][col]
                    break
            raise ValueError(
                f"Input data contains values other than 0 or 1 (or True/False). Found: {problem_val}"
            )

    else:
        raise TypeError(
            f"Unsupported input type for X: {type(X_input)}. Expected List/Series/ndarray of bitstrings, "
            "sequence of 0/1 sequences, or DataFrame/ndarray of 0/1/True/False."
        )

    if X_df is None or X_df.empty:
        raise ValueError("Could not process input X into a DataFrame.")
    if bit_length <= 0:
        raise ValueError("Could not determine a valid bit length.")

    X_df.columns = [f"bit_{i}" for i in range(bit_length)]
    data_types = {col: "boolean" for col in X_df.columns}

    if verbose:
        logger.info(f"Boolean input preparation complete. Detected bit length: {bit_length}.")
    return X_df, data_types, bit_length


def _parse_ordinal_input(
    X_input: Union[List[Any], pd.DataFrame, np.ndarray, pd.Series],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str], int]:
    """Validate and standardise ordinal input into a DataFrame of ordered columns.

    Handles three common input formats:

    - ``list[list[int]]`` / ``list[tuple[int]]`` — one row per configuration.
    - ``numpy.ndarray`` of shape ``(n_samples, n_vars)``.
    - ``pandas.DataFrame`` — columns are variables.

    The columns must contain *orderable* values. Integer-coded levels
    (``0, 1, 2, ...``) are the recommended representation. Pandas
    ``Categorical(..., ordered=True)`` with an explicit category order is
    also accepted; in that case the saved order is preserved.

    .. warning::

        If you pass non-integer values (e.g., strings such as ``"low"``,
        ``"mid"``, ``"high"``) without an explicit ordered Categorical,
        pandas will fall back to lexicographic sorting, which is almost
        never the order you want. Either pre-convert to integers or use
        ``pd.Categorical(col, ordered=True, categories=[...])`` to fix the
        intended order.

    Returns
    -------
    tuple[DataFrame, dict[str, str], int]
        Standardised DataFrame, ``{col: 'ordinal', ...}`` dict, and the
        number of variables.
    """
    if verbose:
        logger.info("Preparing Ordinal input...")

    if not hasattr(X_input, "__len__") or len(X_input) == 0:
        raise ValueError("Input configuration data `X` cannot be empty.")

    if isinstance(X_input, np.ndarray):
        if X_input.ndim != 2:
            raise ValueError(
                f"Ordinal input ndarray must be 2-D (n_samples, n_vars); "
                f"got shape {X_input.shape}."
            )
        try:
            X_df = pd.DataFrame(X_input)
        except Exception as e:
            raise TypeError(f"Could not convert NumPy array to DataFrame: {e}") from e
    elif isinstance(X_input, pd.DataFrame):
        X_df = X_input.copy()
    elif isinstance(X_input, (list, tuple)):
        if not X_input:
            raise ValueError("Input sequence is empty.")
        first = X_input[0]
        if not isinstance(first, (list, tuple, np.ndarray)):
            raise TypeError(
                "If X is a list/tuple, each element must itself be a "
                "list/tuple/ndarray representing one configuration."
            )
        row_lens = {len(r) for r in X_input}
        if len(row_lens) != 1:
            raise ValueError(
                f"All inner sequences must have the same length; "
                f"found lengths {sorted(row_lens)}."
            )
        try:
            X_df = pd.DataFrame(list(X_input))
        except Exception as e:
            raise TypeError(f"Could not convert list input to DataFrame: {e}") from e
    else:
        raise TypeError(
            f"Unsupported input type for X: {type(X_input)}. Expected "
            "DataFrame, 2-D ndarray, or list of lists/tuples."
        )

    if X_df.empty:
        raise ValueError("Input X is empty after parsing.")
    n_vars = X_df.shape[1]
    if n_vars == 0:
        raise ValueError("Input X must have at least one column.")

    if all(isinstance(c, (int, np.integer)) for c in X_df.columns):
        X_df.columns = [f"var_{i}" for i in range(n_vars)]

    data_types = {col: "ordinal" for col in X_df.columns}

    if verbose:
        logger.info(
            f"Ordinal input preparation complete. Detected {n_vars} variable(s)."
        )
    return X_df, data_types, n_vars


def _parse_sequence_input(
    X_input: Union[List[str], pd.Series, np.ndarray, pd.DataFrame],
    alphabet: List[str],
    class_name: str = "SequenceLandscape",
    validated: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str], int]:
    """Validate and standardise sequence input into a categorical DataFrame.

    Returns
    -------
    tuple[DataFrame, dict[str, str], int]
        Standardised DataFrame, data-types dictionary, and sequence length.
    """
    if verbose:
        logger.info(f"Preparing sequence input for {class_name}...")

    if not hasattr(X_input, "__len__") or len(X_input) == 0:
        raise ValueError("Input configuration data `X` cannot be empty.")

    X_df = None
    seq_len = -1
    valid_chars = set(alphabet)
    already_categorical = False

    # --- Format 1: list/series/array of strings ---
    is_sequence_format = False
    if isinstance(X_input, (list, tuple, pd.Series, np.ndarray)):
        try:
            first_element = X_input[0]
            if isinstance(first_element, str):
                is_sequence_format = True
            elif isinstance(X_input, np.ndarray) and X_input.dtype.kind in ("U", "S"):
                is_sequence_format = True
        except (IndexError, TypeError):
            pass

    if is_sequence_format:
        if verbose:
            logger.info("Detected sequence string format input.")
        sequences = list(X_input)
        if not all(isinstance(s, str) for s in sequences):
            raise TypeError(
                "If X is a list/Series/array of strings, all elements must be strings."
            )
        if not sequences:
            raise ValueError("Input sequence list is empty.")
        seq_len = len(sequences[0])
        if seq_len == 0:
            raise ValueError("Sequences cannot be empty strings.")
        n_seq = len(sequences)
        # Fast path (high-dim): uppercase the whole corpus in one C-level
        # "".join(...).upper() instead of the per-sequence loop. Require every
        # length == seq_len (not just the total) so a violation drops to the
        # loop below, which raises the exact original error.
        lengths = [len(s) for s in sequences]
        # seq_len == lengths[0], so all-equal lengths implies all == seq_len.
        uniform_length = min(lengths) == max(lengths)
        joined_upper = None
        if uniform_length:
            joined_upper = "".join(sequences).upper()
            if not validated and not set(joined_upper).issubset(valid_chars):
                # Invalid char somewhere; let the loop raise the per-index error.
                joined_upper = None
        if joined_upper is None:
            validated_sequences = []
            for i, seq in enumerate(sequences):
                seq_upper = seq.upper()
                if len(seq_upper) != seq_len:
                    raise ValueError(
                        f"All sequences must have the same length (expected {seq_len}, got {len(seq_upper)} for sequence {i})."
                    )
                if not validated and not set(seq_upper).issubset(valid_chars):
                    invalid_chars = set(seq_upper) - valid_chars
                    raise ValueError(
                        f"Sequence {i} contains invalid characters: {invalid_chars}. Allowed: {alphabet}"
                    )
                validated_sequences.append(seq_upper)
            joined_upper = "".join(validated_sequences)
        # Build categoricals straight from ASCII bytes via a char->code lookup,
        # skipping the S1->U1 widening and per-column astype. Byte-for-byte
        # identical to DataFrame(...).astype(CategoricalDtype(alphabet)): codes
        # are positions in alphabet, off-alphabet maps to -1 (caught by the null
        # check below). LUT is sized to the final code dtype (int8 covers
        # DNA/RNA/protein) to avoid a wide int64 intermediate; one transpose to
        # contiguous gives each from_codes an already-contiguous row.
        byte_array = np.frombuffer(
            joined_upper.encode("ascii"), dtype=np.uint8
        ).reshape(n_seq, seq_len)
        codes_dtype = np.int8 if len(alphabet) <= np.iinfo(np.int8).max else np.int64
        code_lut = np.full(256, -1, dtype=codes_dtype)
        for _code, _ch in enumerate(alphabet):
            code_lut[ord(_ch)] = _code
        codes_2d = code_lut[byte_array]
        codes_by_col = np.ascontiguousarray(codes_2d.T)  # (seq_len, n_seq)
        cat_dtype = pd.CategoricalDtype(categories=alphabet, ordered=False)
        X_df = pd.DataFrame(
            {
                f"pos_{i}": pd.Categorical.from_codes(codes_by_col[i], dtype=cat_dtype)
                for i in range(seq_len)
            },
            copy=False,
        )
        already_categorical = True

    # --- Format 2: DataFrame or ndarray ---
    elif isinstance(X_input, (pd.DataFrame, np.ndarray)):
        if verbose:
            logger.info("Detected DataFrame/ndarray format input.")
        if isinstance(X_input, np.ndarray):
            X_df = pd.DataFrame(X_input).astype(str).apply(lambda col: col.str.upper())
        else:
            X_df = X_input.copy().astype(str).apply(lambda col: col.str.upper())
        if X_df.empty:
            raise ValueError("Input DataFrame/ndarray is empty.")
        seq_len = X_df.shape[1]
        if seq_len == 0:
            raise ValueError("Input DataFrame/ndarray cannot have zero columns.")
        if not validated:
            for col in X_df.columns:
                unique_vals = set(X_df[col].dropna().unique())
                if not unique_vals.issubset(valid_chars):
                    invalid_chars = unique_vals - valid_chars
                    raise ValueError(
                        f"Column '{col}' contains invalid characters: {invalid_chars}. Allowed: {alphabet}"
                    )
        if isinstance(X_input, np.ndarray):
            X_df.columns = [f"pos_{i}" for i in range(seq_len)]
    else:
        raise TypeError(
            f"Unsupported input type for X: {type(X_input)}. Expected List/Series/ndarray of strings, or DataFrame/ndarray."
        )

    if X_df is None or X_df.empty:
        raise ValueError("Could not process input X into a DataFrame.")
    if not already_categorical:
        cat_dtype = pd.CategoricalDtype(categories=alphabet, ordered=False)
        X_df = X_df.astype(cat_dtype)
    for col in X_df.columns:
        if X_df[col].isnull().any():
            raise ValueError(
                f"Invalid characters found in column '{col}' after categorical conversion. Expected characters from: {alphabet}"
            )

    data_types = {col: "categorical" for col in X_df.columns}
    if verbose:
        logger.info("Sequence input preparation complete.")
    return X_df, data_types, seq_len
