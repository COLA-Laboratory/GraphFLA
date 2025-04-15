import pandas as pd
import numpy as np
from typing import List, Any, Dict, Tuple, Union, Optional
import warnings

from ..distances import hamming_distance
from ..base import BaseLandscape


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
    data_types = {str(col): "boolean" for col in X_df.columns}

    if verbose:
        print(
            f"Boolean input preprocessing complete. Detected bit length: {bit_length}."
        )
    return X_df, data_types, bit_length


class BooleanLandscape(BaseLandscape):
    """Represents fitness landscapes defined over Boolean (binary) spaces.

    This class specializes `BaseLandscape` for configurations represented as
    binary strings or arrays of 0s and 1s. It handles specific input formats
    for Boolean data and uses Hamming distance as the default distance metric.
    Neighbors are defined by single bit flips (n_edit=1).

    Parameters
    ----------
    verbose : bool, default=True
        Controls the verbosity of the output during landscape construction
        and analysis.

    Attributes
    ----------
    bit_length : int or None
        The length of the binary strings (number of variables) in the landscape.
        Determined during `from_data`.
    graph : networkx.DiGraph or None
        The directed graph representing the fitness landscape. Nodes are
        integer indices, edges connect single-bit-flip neighbors pointing
        towards higher fitness (if `maximize=True`). Populated after `from_data`.
    configs : pandas.Series or None
        Maps node index to the integer-encoded binary tuple (e.g., (0, 1, 0)).
        Populated after `from_data`.
    config_dict : dict or None
        Describes the encoding for bit positions (all 'boolean' with 'max'=1).
        Populated after `from_data`.
    data_types : dict or None
        Inferred data types, typically {'bit_0': 'boolean', ...}. Populated
        after `from_data`.
    n_configs : int or None
        Number of unique binary configurations. Populated after `from_data`.
    n_vars : int or None
        Length of the binary strings (`bit_length`). Populated after `from_data`.
    n_edges : int or None
        Number of edges (single bit flips leading to higher fitness).
        Populated after `from_data`.
    n_lo : int or None
        Number of local optima. Populated after graph analysis.
    lo_index : list[int] or None
        Indices of local optima. Populated after graph analysis.
    go_index : int or None
        Index of the global optimum. Populated after graph analysis.
    go : dict or None
        Attributes of the global optimum node. Populated after graph analysis.
    basin_index : dict[int, int] or None
        Mapping from node index to its basin's local optimum index. Populated
        if `calculate_basins=True`.
    lon : networkx.DiGraph or None
        Local Optima Network graph, if calculated.
    has_lon : bool
        Flag indicating if the LON has been calculated.
    maximize : bool
        Whether the objective is to maximize fitness. Set during `from_data`.
    verbose : bool
        Verbosity level.
    _is_built : bool
        Internal flag for build status.

    See Also
    --------
    BaseLandscape : The base class for all landscapes.
    Landscape : Factory class to create appropriate landscape types.
    """

    bit_length: Optional[int] = None

    def from_data(
        self,
        X: Union[List[Any], pd.DataFrame, np.ndarray, pd.Series],
        f: Union[pd.Series, list, np.ndarray],
        *,
        maximize: bool = True,
        epsilon: Union[float, str] = "auto",
        calculate_basins: bool = True,
        calculate_paths: bool = False,
        impute: bool = False,
        impute_model: Optional[Any] = None,
        verbose: Optional[bool] = None,
        n_edit: int = 1,
    ) -> "BooleanLandscape":
        """Construct the Boolean landscape graph and properties from data.

        This method specializes `BaseLandscape.from_data` for Boolean data.
        It accepts binary configurations in various formats (e.g., list of
        bitstrings ['010', '110'], list of lists [[0, 1, 0], [1, 1, 0]],
        DataFrame/ndarray of 0/1 or True/False). It validates and standardizes
        the input, infers the 'boolean' `data_types`, and determines the
        `bit_length` before calling the base class construction method.

        Parameters
        ----------
        X : list, pandas.DataFrame, numpy.ndarray, or pandas.Series
            Boolean configuration data. Can be bitstrings, sequences of 0/1,
            or tabular data with 0/1 or True/False values. Assumes all
            configurations have the same length.
        f : pandas.Series, list, or numpy.ndarray
            Fitness values corresponding to each configuration in `X`. Must have
            the same length as `X`.
        maximize : bool, default=True
            Determines the optimization direction (maximize=True for peaks,
            False for valleys).
        epsilon : float or 'auto', default='auto'
            Tolerance for floating point comparisons.
        calculate_basins : bool, default=True
            If True, calculates basins of attraction.
        calculate_paths : bool, default=False
            If True, calculates accessible paths (ancestors). Skipped for large
            landscapes.
        impute : bool, default=False
            If True, imputes missing fitness values using `X`. Requires
            scikit-learn or `impute_model`.
        impute_model : object, optional
            Custom model for fitness imputation if `impute=True`.
        verbose : bool, optional
            Overrides the instance's verbosity setting.
        n_edit : int, default=1
            Edit distance for neighborhood definition. For `BooleanLandscape`,
            this is typically fixed to 1 (single bit flips). Values other than
            1 might be ignored or raise warnings.

        Returns
        -------
        self : BooleanLandscape
            The populated landscape instance.

        Raises
        ------
        ValueError
            If input data is invalid (e.g., inconsistent lengths, non-binary
            values, empty data).
        RuntimeError
            If the landscape has already been built.
        TypeError
            If input `X` format is unsupported.
        ImportError
            If `impute=True` and scikit-learn is not installed, unless an
            `impute_model` is provided.
        """
        effective_verbose = verbose if verbose is not None else self.verbose

        # Preprocess boolean input into a standardized DataFrame format
        # and infer data types ('boolean' for all columns).
        X_df, bool_data_types, bit_len = _preprocess_boolean_input(
            X_input=X, verbose=effective_verbose
        )
        self.bit_length = bit_len  # Store the bit length

        # Call the BaseLandscape.from_data method with processed inputs
        super().from_data(
            X=X_df,  # Pass the standardized DataFrame
            f=f,
            data_types=bool_data_types,  # Pass the inferred boolean types
            maximize=maximize,
            epsilon=epsilon,
            calculate_basins=calculate_basins,
            calculate_paths=calculate_paths,
            impute=impute,
            impute_model=impute_model,
            verbose=verbose,  # Pass original verbose override
            n_edit=n_edit,  # Pass n_edit
        )
        return self

    def _generate_neighbors(
        self, config: Tuple, config_dict: Dict, n_edit: int
    ) -> List[Tuple]:
        """Generates neighbors for a Boolean configuration (single bit flips).

        Overrides `BaseLandscape._generate_neighbors`. For Boolean landscapes,
        neighbors are defined as configurations reachable by flipping a single
        bit (0 to 1, or 1 to 0). Only `n_edit=1` is supported.

        Parameters
        ----------
        config : tuple
            The integer-encoded binary tuple (e.g., (0, 1, 0)).
        config_dict : dict
            Ignored in this implementation.
        n_edit : int
            Edit distance. Only `n_edit=1` is supported.

        Returns
        -------
        list[tuple]
            List of neighboring integer-encoded binary tuples.
        """
        if n_edit != 1:
            if self.verbose:
                warnings.warn(
                    f"{self.__class__.__name__}._generate_neighbors uses n_edit=1 "
                    f"for single bit flips. Received n_edit={n_edit}. Returning no neighbors.",
                    UserWarning,
                )
            return []

        neighbors = []
        current_config_list = list(config)  # Convert tuple to list for modification
        num_bits = len(current_config_list)  # Should equal self.bit_length

        for i in range(num_bits):  # Iterate through each bit position
            neighbor_list = current_config_list.copy()
            # Flip the bit at position i
            neighbor_list[i] = 1 - neighbor_list[i]
            neighbors.append(tuple(neighbor_list))  # Convert back to tuple
        return neighbors

    def _get_default_distance_metric(self):
        """Returns Hamming distance as the default for Boolean strings."""
        return hamming_distance
