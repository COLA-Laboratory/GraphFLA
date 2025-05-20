from typing import Protocol, Tuple, Dict, List, Union, Any, runtime_checkable
import pandas as pd
import numpy as np

ALLOWED_DATA_TYPES = {"boolean", "categorical", "ordinal"}
DNA_ALPHABET = ["A", "C", "G", "T"]
RNA_ALPHABET = ["A", "C", "G", "U"]
PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


@runtime_checkable
class DataPreprocessor(Protocol):
    """Protocol defining the interface for data preprocessing."""

    def preprocess(
        self, X: Any, f: Union[pd.Series, list, np.ndarray], verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """
        Preprocess input data for a specific landscape type.

        Parameters
        ----------
        X : Any
            Input configuration data
        f : Union[pd.Series, list, np.ndarray]
            Fitness values corresponding to configurations
        verbose : bool
            Whether to print processing information

        Returns
        -------
        X_processed : DataFrame
            Processed feature data
        f_processed : Series
            Processed fitness values
        data_types : dict
            Dictionary mapping column names to data types
        dimension : int
            Dimensionality of the data (e.g., sequence length, bit length)
        """
        ...


class BooleanPreprocessor:
    """Preprocessor for boolean data."""

    def preprocess(
        self,
        X: Union[List[Any], pd.DataFrame, np.ndarray, pd.Series],
        f: Union[pd.Series, list, np.ndarray],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """Preprocess boolean data."""
        # Use the existing _preprocess_boolean_input function
        X_df, bool_data_types, bit_len = _preprocess_boolean_input(
            X_input=X, verbose=verbose
        )

        # Convert fitness to Series if needed
        if isinstance(f, (list, np.ndarray)):
            f_series = pd.Series(f, name="fitness").copy()
        elif isinstance(f, pd.Series):
            f_series = f.copy()
            f_series.name = "fitness"
        else:
            raise TypeError(
                f"Input f must be a pandas Series, list, or numpy ndarray, got {type(f)}."
            )

        # Ensure X and f have matching indices
        X_df.reset_index(drop=True, inplace=True)
        f_series.reset_index(drop=True, inplace=True)
        f_series.index = X_df.index

        return X_df, f_series, bool_data_types, bit_len


class SequencePreprocessor:
    """Preprocessor for sequence data."""

    def __init__(self, alphabet: List[str]):
        """Initialize with the appropriate alphabet."""
        self.alphabet = alphabet

    def preprocess(
        self,
        X: Union[List[str], pd.Series, np.ndarray, pd.DataFrame],
        f: Union[pd.Series, list, np.ndarray],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """Preprocess sequence data."""
        # Validate that all values in X are in the alphabet
        self._validate_alphabet_conformity(X)

        # Use the existing _preprocess_sequence_input function
        X_df, seq_data_types, seq_len = _preprocess_sequence_input(
            X_input=X,
            alphabet=self.alphabet,
            class_name=f"Sequence ({self.alphabet[0]}{self.alphabet[-1]})",
            verbose=verbose,
        )

        # Convert fitness to Series if needed
        if isinstance(f, (list, np.ndarray)):
            f_series = pd.Series(f, name="fitness").copy()
        elif isinstance(f, pd.Series):
            f_series = f.copy()
            f_series.name = "fitness"
        else:
            raise TypeError(
                f"Input f must be a pandas Series, list, or numpy ndarray, got {type(f)}."
            )

        # Ensure X and f have matching indices
        X_df.reset_index(drop=True, inplace=True)
        f_series.reset_index(drop=True, inplace=True)
        f_series.index = X_df.index

        return X_df, f_series, seq_data_types, seq_len

    def _validate_alphabet_conformity(
        self, X: Union[List[str], pd.Series, np.ndarray, pd.DataFrame]
    ) -> None:
        """Validates that all values in X conform to the specified alphabet and warns about unused alphabet characters."""
        valid_chars = set(self.alphabet)
        used_chars = set()

        # Handle different input types
        if isinstance(X, (list, tuple, pd.Series)):
            # For sequence format (list of strings)
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
            # For tabular format
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
            # For numpy array
            if X.ndim == 1:
                # 1D array (like a list of sequences)
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
                # 2D array (tabular format)
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

        # Check for unused alphabet characters
        unused_chars = valid_chars - used_chars
        if unused_chars:
            import warnings

            warnings.warn(
                f"The following characters appear in the alphabet but are missing from input X: "
                f"{', '.join(sorted(unused_chars))}. "
                f"This might indicate you're using an incorrect alphabet for your data.",
                UserWarning,
            )


class DefaultPreprocessor:
    """Default preprocessor for mixed data types."""

    def preprocess(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        f: Union[pd.Series, list, np.ndarray],
        data_types: Dict[str, str],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], int]:
        """
        Preprocess data with explicit data types.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input configuration data
        f : Series, list, or ndarray
            Fitness values
        data_types : dict
            Dictionary mapping column names to data types
        verbose : bool
            Whether to print processing information

        Returns
        -------
        X_processed : DataFrame
            Processed feature data
        f_processed : Series
            Processed fitness values
        data_types_validated : dict
            Validated data types dictionary
        n_vars : int
            Number of variables in the data
        """
        # Convert X to DataFrame if ndarray
        if isinstance(X, np.ndarray):
            try:
                columns = [f"var_{i}" for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=columns).copy()
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

        # Convert f to Series
        if isinstance(f, (list, np.ndarray)):
            try:
                f_series = pd.Series(f, name="fitness").copy()
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

        # Validate data_types dictionary against X columns
        data_types_validated = self._validate_data_types(X_df, data_types, verbose)

        # Align indices
        X_df.reset_index(drop=True, inplace=True)
        f_series.reset_index(drop=True, inplace=True)
        f_series.index = X_df.index
        return X_df, f_series, data_types_validated, len(data_types_validated)

    def _validate_data_types(
        self, X: pd.DataFrame, data_types: Dict[str, str], verbose: bool
    ) -> Dict[str, str]:
        """Validates the data_types dictionary against X's columns."""
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

    def _validate_data_types_dict(
        self, X_in: pd.DataFrame, dt_in: Dict[str, str]
    ) -> Dict[str, str]:
        """Validates the data_types dictionary against X's columns."""
        if self.verbose:
            print(" - Validating data types dictionary...")

        if not isinstance(dt_in, dict):
            raise TypeError(f"data_types must be a dictionary, got {type(dt_in)}.")

        x_cols = set(X_in.columns)
        dt_keys = set(dt_in.keys())

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
        for key, type_val in dt_in.items():
            if type_val not in ALLOWED_DATA_TYPES:
                invalid_types[key] = type_val

        if invalid_types:
            raise ValueError(
                f"Invalid data types found in data_types dictionary: {invalid_types}. "
                f"Allowed types are: {ALLOWED_DATA_TYPES}."
            )

        validated_dt = {col: dt_in[col] for col in X_in.columns}

        if self.verbose:
            print("   - Data types dictionary validation successful.")

        return validated_dt


def _preprocess_boolean_input(
    X_input: Union[List[Any], pd.DataFrame, np.ndarray, pd.Series],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str], int]:
    """
    Validates and standardizes boolean input into a DataFrame with integer 0/1 columns.

    Handles various input formats:
    - List/Series/Array of bitstrings (e.g., ['010', '110'])
    - List/Tuple of Lists/Tuples of 0/1 (e.g., [[0, 1, 0], [1, 1, 0]])
    - Pandas DataFrame or NumPy array containing 0/1 or True/False.

    Parameters
    ----------
    X_input : Union[List[Any], pd.DataFrame, np.ndarray, pd.Series]
        The raw boolean configuration data.
    verbose : bool, default=True
        Whether to print processing information.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, str], int]
        - Standardized DataFrame with integer 0/1 values.
        - Dictionary of data types ({'bit_0': 'boolean', ...}).
        - Detected bit length.

    Raises
    ------
    ValueError
        If input is empty, inconsistent, or contains invalid values/formats.
    TypeError
        If the input type is unsupported.
    """
    if verbose:
        print("Preprocessing Boolean input...")

    if not hasattr(X_input, "__len__") or len(X_input) == 0:
        raise ValueError("Input configuration data `X` cannot be empty.")

    X_df = None
    bit_length = -1

    # Detect format
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
            # Further check if elements are likely 0/1
            if all(isinstance(val, (int, bool, np.integer)) for val in first_element):
                is_sequence_of_sequences = True
        elif isinstance(X_input, np.ndarray) and X_input.dtype.kind in ("U", "S"):
            is_sequence_of_strings = True  # Array of strings
    except (IndexError, TypeError):
        pass  # Will be handled by DataFrame/ndarray check or raise error

    # Format 1: List/Series/Array of Strings (Bitstring Format)
    if is_sequence_of_strings:
        if verbose:
            print("Detected bitstring sequence format input.")
        bitstrings = list(X_input)  # Convert Series/Array to list
        if not all(isinstance(s, str) for s in bitstrings):
            raise TypeError(
                "If X is a sequence of strings, all elements must be strings."
            )
        if not bitstrings:
            raise ValueError("Input bitstring sequence is empty.")

        bit_length = len(bitstrings[0])
        if bit_length == 0:
            raise ValueError("Bitstrings cannot be empty.")

        data = []
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
            data.append([int(bit) for bit in bstr])

        X_df = pd.DataFrame(data)

    # Format 2: List/Tuple of Lists/Tuples/Arrays (0/1 Sequence Format)
    elif is_sequence_of_sequences:
        if verbose:
            print("Detected sequence of 0/1 lists/tuples format input.")
        sequences = list(X_input)  # Ensure it's a list
        if not sequences:
            raise ValueError("Input sequence is empty.")

        try:
            # Attempt to convert inner sequences to lists of ints for consistency check
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
            data.append(seq)  # Already a list of 0/1 ints

        X_df = pd.DataFrame(data)

    # Format 3: DataFrame or Ndarray (Tabular Format)
    elif isinstance(X_input, (pd.DataFrame, np.ndarray)):
        if verbose:
            print("Detected DataFrame/ndarray format input.")

        if isinstance(X_input, np.ndarray):
            # Convert numpy array to DataFrame, attempt flexible type handling
            try:
                X_df = pd.DataFrame(X_input)
            except Exception as e:
                raise TypeError(f"Could not convert NumPy array to DataFrame: {e}")
        else:  # Is already a DataFrame
            X_df = X_input.copy()

        if X_df.empty:
            raise ValueError("Input DataFrame/ndarray is empty.")

        bit_length = X_df.shape[1]
        if bit_length == 0:
            raise ValueError("Input DataFrame/ndarray cannot have zero columns.")

        # Validate and convert contents to 0/1 integers
        try:
            # Replace True/False with 1/0 if they exist
            X_df = X_df.replace({True: 1, False: 0})
            # Attempt conversion to int, raising error if non-numeric remain
            X_df = X_df.astype(int)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert DataFrame content to integer 0/1: {e}. Ensure input contains only boolean-like values (0, 1, True, False)."
            )

        # Check if all values are now 0 or 1
        if not X_df.isin([0, 1]).all().all():
            # Find example problematic value
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

    # Final checks and setup
    if X_df is None or X_df.empty:
        raise ValueError(
            "Could not process input X into a DataFrame."
        )  # Should not happen if logic above is correct
    if bit_length <= 0:
        raise ValueError("Could not determine a valid bit length.")  # Should not happen

    # Assign standard column names
    X_df.columns = [f"bit_{i}" for i in range(bit_length)]

    # Create data_types dictionary
    data_types = {col: "boolean" for col in X_df.columns}

    if verbose:
        print(
            f"Boolean input preprocessing complete. Detected bit length: {bit_length}."
        )
    return X_df, data_types, bit_length


def _preprocess_sequence_input(
    X_input: Union[List[str], pd.Series, np.ndarray, pd.DataFrame],
    alphabet: List[str],
    class_name: str = "SequenceLandscape",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str], int]:
    """
    Validates and standardizes sequence input (strings or tabular)
    into a DataFrame with categorical columns ordered by the alphabet.
    (Implementation retained from previous step - assumed correct)
    """
    # ... (implementation from previous step) ...
    if verbose:
        print(f"Preprocessing sequence input for {class_name}...")

    if not hasattr(X_input, "__len__") or len(X_input) == 0:
        raise ValueError("Input configuration data `X` cannot be empty.")

    X_df = None
    seq_len = -1
    valid_chars = set(alphabet)

    # Format 1: List/Series/Array of Strings (Sequence Format)
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
            if not set(seq_upper).issubset(valid_chars):
                invalid_chars = set(seq_upper) - valid_chars
                raise ValueError(
                    f"Sequence {i} contains invalid characters: {invalid_chars}. Allowed: {alphabet}"
                )
            validated_sequences.append(seq_upper)
        X_df = pd.DataFrame([list(seq) for seq in validated_sequences])
        X_df.columns = [f"pos_{i}" for i in range(seq_len)]

    # Format 2: DataFrame or Ndarray (Tabular Format)
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

    # Enforce Categorical Order
    if X_df is None or X_df.empty:
        raise ValueError("Could not process input X into a DataFrame.")
    for col in X_df.columns:
        X_df[col] = pd.Categorical(X_df[col], categories=alphabet, ordered=False)
        if X_df[col].isnull().any():
            raise ValueError(
                f"Invalid characters found in column '{col}' after categorical conversion. Expected characters from: {alphabet}"
            )

    # Create data_types dictionary
    data_types = {col: "categorical" for col in X_df.columns}
    if verbose:
        print("Sequence input preprocessing complete.")
    return X_df, data_types, seq_len
