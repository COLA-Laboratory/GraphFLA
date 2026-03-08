"""Data preparation and encoding for fitness landscape construction.

This module provides:

1. **Input handlers** — strategy classes that validate and standardize
   raw configuration data (boolean, sequence, or mixed-type) into
   DataFrames suitable for encoding.

2. **Pipeline functions** — the processing steps used by
   ``Landscape.build_from_data``: filtering, preparation, cleaning,
   and encoding.

3. **PreparedData** — a container for the fully encoded dataset and
   its associated metadata.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)
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
        """Check that all values in X conform to the alphabet."""
        valid_chars = set(self.alphabet)
        used_chars = set()

        if isinstance(X, (list, tuple, pd.Series)):
            for idx, seq in enumerate(X):
                if isinstance(seq, str):
                    seq_chars = set(seq.upper())
                    invalid_chars = seq_chars - valid_chars
                    if invalid_chars:
                        raise ValueError(
                            f"Input X values at index {idx} contain {', '.join(invalid_chars)}, "
                            f"which is not among specified alphabet: {self.alphabet}"
                        )
                    used_chars.update(seq_chars)

        elif isinstance(X, pd.DataFrame):
            for col in X.columns:
                for idx, val in enumerate(X[col]):
                    if val is not None:
                        val_upper = str(val).upper()
                        if val_upper not in valid_chars:
                            raise ValueError(
                                f"Input X values at position ({idx}, {col}) contain '{val}', "
                                f"which is not among specified alphabet: {self.alphabet}"
                            )
                        used_chars.add(val_upper)

        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                for idx, seq in enumerate(X):
                    if isinstance(seq, str):
                        seq_chars = set(seq.upper())
                        invalid_chars = seq_chars - valid_chars
                        if invalid_chars:
                            raise ValueError(
                                f"Input X values at index {idx} contain {', '.join(invalid_chars)}, "
                                f"which is not among specified alphabet: {self.alphabet}"
                            )
                        used_chars.update(seq_chars)
            else:
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        val = X[i, j]
                        if val is not None:
                            val_upper = str(val).upper()
                            if val_upper not in valid_chars:
                                raise ValueError(
                                    f"Input X values at position ({i}, {j}) contain '{val}', "
                                    f"which is not among specified alphabet: {self.alphabet}"
                                )
                            used_chars.add(val_upper)

        unused_chars = valid_chars - used_chars
        if unused_chars:
            warnings.warn(
                f"The following characters appear in the alphabet but are missing from input X: "
                f"{', '.join(sorted(unused_chars))}. "
                f"This might indicate you're using an incorrect alphabet for your data.",
                UserWarning,
            )


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
            print(" - Validating data types dictionary...")

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
            print("   - Data types dictionary validation successful.")

        return validated_dt


# ===================================================================
# Data container
# ===================================================================


@dataclass(frozen=True)
class PreparedData:
    """Encoded configuration data plus the metadata needed by ``Landscape``."""

    data_for_attributes: pd.DataFrame
    data_types: Dict[str, str]
    n_vars: int
    configs: pd.Series
    config_dict: Dict[int, Dict[str, int]]
    configs_array: np.ndarray


# ===================================================================
# Pipeline — public API
# ===================================================================


def filter_data(X, f, maximize, tau, filter_mode, verbose):
    """Filter raw configuration data by the functional threshold."""
    if tau is not None:
        if filter_mode == "any":
            if verbose:
                print(
                    f" - Applying functional threshold filter "
                    f"(tau={tau})..."
                )

            initial_count = len(f)
            fitness_values = f.to_numpy(copy=False) if isinstance(f, pd.Series) else np.asarray(f)

            if maximize:
                mask = fitness_values >= tau
                comparison_op = ">="
            else:
                mask = fitness_values <= tau
                comparison_op = "<="

            X = _apply_mask(X, mask)
            f = _apply_mask(f, mask)

            final_count = len(f)
            removed_count = initial_count - final_count

            if verbose:
                opposite_op = "<" if comparison_op[0] == ">" else ">"
                opposite_op += "=" if len(comparison_op) > 1 else ""

                print(
                    f"   - Removed {removed_count} configurations with fitness "
                    f"{opposite_op} {tau}"
                )
                print(f"   - Kept {final_count}/{initial_count} configurations")

            if final_count == 0:
                raise ValueError(
                    f"All configurations removed by functional threshold filter "
                    f"(tau={tau})"
                )

    return X, f


def prepare_data(handler, X, f, data_types=None, verbose=True):
    """Dispatch to the appropriate handler based on its type.

    Parameters
    ----------
    handler : InputHandler
        The handler instance selected for the current landscape type.
    X : Any
        Raw configuration data.
    f : Series, list, or ndarray
        Fitness values.
    data_types : dict or None
        Required when *handler* is a ``DefaultHandler``; ignored otherwise.
    verbose : bool
        Whether to print processing information.

    Returns
    -------
    X_processed : DataFrame
    f_processed : Series
    data_types : dict
    n_vars : int
    """
    if isinstance(handler, DefaultHandler):
        if not isinstance(data_types, dict):
            raise ValueError(
                "Data_types must be a dictionary, e.g., "
                "{'var_0': 'boolean', 'var_1': 'categorical'}"
                f"got {type(data_types)}."
            )
        return handler.prepare(X, f, data_types, verbose=verbose)
    else:
        return handler.prepare(X, f, verbose=verbose)


def clean_data(
    X_in: pd.DataFrame,
    f_in: pd.Series,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean unusable rows and duplicate configurations."""
    if verbose:
        print(" - Handling missing values and duplicates...")

    X_clean, f_clean = _drop_missing(X_in, f_in, verbose=verbose)
    if len(X_clean) == 0:
        raise ValueError("All data removed during missing value handling.")

    X_final, f_final = _drop_duplicates(X_clean, f_clean, verbose=verbose)
    if len(X_final) == 0:
        raise ValueError("All data removed after handling duplicates.")

    return X_final, f_final


def encode_data(
    X: pd.DataFrame,
    f: pd.Series,
    data_types: Dict[str, str],
    verbose: bool = False,
) -> PreparedData:
    """Encode configurations and assemble metadata for graph construction."""
    if verbose:
        print("Preparing data for landscape construction (encoding variables)...")

    prepared_data_types = data_types.copy()
    nunique = X.nunique()
    invariant_cols = nunique.index[nunique <= 1].tolist()

    if invariant_cols:
        X = X.drop(columns=invariant_cols)
        prepared_data_types = {
            key: value
            for key, value in prepared_data_types.items()
            if key not in invariant_cols
        }
        if verbose:
            print(
                f" - Removed {len(invariant_cols)} invariant variable(s); "
                f"{len(prepared_data_types)} variable(s) remaining."
            )

    encoded_columns = {}
    for col, dtype in prepared_data_types.items():
        col_data = X[col]
        if dtype == "boolean":
            encoded_columns[col] = col_data.astype(bool).astype(int)
        elif dtype == "categorical":
            if isinstance(col_data.dtype, pd.CategoricalDtype):
                encoded_columns[col] = col_data.cat.codes
            else:
                encoded_columns[col] = pd.factorize(col_data)[0]
        elif dtype == "ordinal":
            if isinstance(col_data.dtype, pd.CategoricalDtype) and col_data.cat.ordered:
                encoded_columns[col] = col_data.cat.codes
            else:
                encoded_columns[col] = pd.Categorical(col_data, ordered=True).codes
        else:
            raise ValueError(
                f"Unsupported data type '{dtype}' encountered during encoding."
            )

    X_encoded = pd.DataFrame(encoded_columns, index=X.index, copy=False)
    encoded_array = X_encoded.to_numpy(copy=False)
    max_code = int(encoded_array.max()) if encoded_array.size else 0
    if max_code <= np.iinfo(np.uint8).max:
        encoded_dtype = np.uint8
    elif max_code <= np.iinfo(np.uint16).max:
        encoded_dtype = np.uint16
    elif max_code <= np.iinfo(np.uint32).max:
        encoded_dtype = np.uint32
    else:
        encoded_dtype = np.uint64

    configs_array = np.ascontiguousarray(encoded_array, dtype=encoded_dtype)
    config_objects = np.fromiter(
        (tuple(row.tolist()) for row in configs_array),
        dtype=object,
        count=len(configs_array),
    )
    configs = pd.Series(config_objects, index=X_encoded.index, copy=False)

    if configs.duplicated().any():
        warnings.warn(
            "Duplicate encoded configurations found after validation. "
            "This might indicate issues in the duplicate handling step.",
            RuntimeWarning,
        )

    config_dict = _build_config_dict(prepared_data_types, X_encoded)

    data_for_attributes = X.reset_index(drop=True)
    data_for_attributes["fitness"] = f.to_numpy(copy=False)

    return PreparedData(
        data_for_attributes=data_for_attributes,
        data_types=prepared_data_types,
        n_vars=len(prepared_data_types),
        configs=configs,
        config_dict=config_dict,
        configs_array=configs_array,
    )


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
        print("Preparing Boolean input...")

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
            print("Detected bitstring sequence format input.")
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

        validated_bitstrings = []
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
            validated_bitstrings.append(bstr)

        bit_array = np.frombuffer(
            "".join(validated_bitstrings).encode("ascii"), dtype=np.uint8
        ).reshape(len(validated_bitstrings), bit_length) - ord("0")
        X_df = pd.DataFrame(bit_array, copy=False)

    # --- Format 2: sequences of 0/1 ---
    elif is_sequence_of_sequences:
        if verbose:
            print("Detected sequence of 0/1 lists/tuples format input.")
        sequences = list(X_input)
        if not sequences:
            raise ValueError("Input sequence is empty.")

        try:
            processed_sequences = [[int(val) for val in seq] for seq in sequences]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert inner sequences to integers: {e}")

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
            print("Detected DataFrame/ndarray format input.")

        if isinstance(X_input, np.ndarray):
            try:
                X_df = pd.DataFrame(X_input)
            except Exception as e:
                raise TypeError(f"Could not convert NumPy array to DataFrame: {e}")
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
        print(f"Boolean input preparation complete. Detected bit length: {bit_length}.")
    return X_df, data_types, bit_length


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
        print(f"Preparing sequence input for {class_name}...")

    if not hasattr(X_input, "__len__") or len(X_input) == 0:
        raise ValueError("Input configuration data `X` cannot be empty.")

    X_df = None
    seq_len = -1
    valid_chars = set(alphabet)

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
            print("Detected sequence string format input.")
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
        sequence_array = np.frombuffer(
            "".join(validated_sequences).encode("ascii"), dtype="S1"
        ).reshape(len(validated_sequences), seq_len).astype("U1")
        X_df = pd.DataFrame(
            sequence_array,
            columns=[f"pos_{i}" for i in range(seq_len)],
            copy=False,
        )

    # --- Format 2: DataFrame or ndarray ---
    elif isinstance(X_input, (pd.DataFrame, np.ndarray)):
        if verbose:
            print("Detected DataFrame/ndarray format input.")
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
    cat_dtype = pd.CategoricalDtype(categories=alphabet, ordered=False)
    X_df = X_df.astype(cat_dtype)
    for col in X_df.columns:
        if X_df[col].isnull().any():
            raise ValueError(
                f"Invalid characters found in column '{col}' after categorical conversion. Expected characters from: {alphabet}"
            )

    data_types = {col: "categorical" for col in X_df.columns}
    if verbose:
        print("Sequence input preparation complete.")
    return X_df, data_types, seq_len


def _drop_missing(
    X_in: pd.DataFrame,
    f_in: pd.Series,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Handle missing values by dropping incomplete rows."""
    if verbose:
        print(" - Handling missing values...")

    X_out, f_out = X_in, f_in
    initial_count = len(X_out)

    if X_out.isnull().values.any():
        nan_rows_X = X_out.isnull().any(axis=1)
        n_removed_X = nan_rows_X.sum()
        if verbose:
            print(
                f"   - Found {n_removed_X} rows with NaN in X (features). Removing them."
            )
        X_out = X_out.loc[~nan_rows_X]
        f_out = f_out.loc[~nan_rows_X]
        if len(X_out) == 0:
            warnings.warn("All rows removed due to NaNs in X.", RuntimeWarning)
            return X_out, f_out

    nan_mask_f = f_out.isnull()
    if nan_mask_f.any():
        n_missing_f = nan_mask_f.sum()
        if verbose:
            print(
                f"   - Found {n_missing_f} rows with NaN in f (fitness). Removing them."
            )
        X_out = X_out.loc[~nan_mask_f]
        f_out = f_out.loc[~nan_mask_f]

    final_count = len(X_out)
    if verbose and initial_count != final_count:
        print(
            f"   - Missing value handling complete. Kept {final_count}/{initial_count} configurations."
        )

    f_out = f_out.loc[X_out.index]
    return X_out, f_out


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


__all__ = [
    "ALLOWED_DATA_TYPES",
    "DNA_ALPHABET",
    "RNA_ALPHABET",
    "PROTEIN_ALPHABET",
    "PreparedData",
    "InputHandler",
    "BooleanHandler",
    "SequenceHandler",
    "DefaultHandler",
    "filter_data",
    "prepare_data",
    "clean_data",
    "encode_data",
]
