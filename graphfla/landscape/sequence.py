import pandas as pd
import numpy as np
from typing import List, Any, Dict, Tuple, Union, Optional
import warnings

from ..distances import hamming_distance
from ..base import BaseLandscape

# --- Constants ---
ALLOWED_DATA_TYPES = {"boolean", "categorical", "ordinal"}
DNA_ALPHABET = ["A", "C", "G", "T"]
RNA_ALPHABET = ["A", "C", "G", "U"]
PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


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
    data_types = {str(col): "categorical" for col in X_df.columns}
    if verbose:
        print("Sequence input preprocessing complete.")
    return X_df, data_types, seq_len


class SequenceLandscape(BaseLandscape):
    """Represents fitness landscapes defined over sequence spaces (DNA, RNA, protein).

    This class extends `BaseLandscape` to handle sequence-specific data,
    such as DNA strings, RNA strings, or protein amino acid sequences. It
    infers the necessary 'categorical' `data_types` based on the provided
    sequence alphabet and standardizes sequence input formats. It typically uses
    Hamming distance as the default distance metric and assumes neighbors are
    defined by single point mutations (n_edit=1).

    Concrete subclasses like `DNALandscape`, `RNALandscape`, and
    `ProteinLandscape` define the appropriate `ALPHABET`.

    Parameters
    ----------
    verbose : bool, default=True
        Controls the verbosity of the output during landscape construction
        and analysis.

    Attributes
    ----------
    ALPHABET : list[str]
        Class attribute. Must be defined by subclasses (e.g., `DNA_ALPHABET`).
        Specifies the allowed characters in the sequences.
    sequence_length : int or None
        The length of the sequences in the landscape. Determined during
        `from_data`.
    graph : networkx.DiGraph or None
        The directed graph representing the fitness landscape. Nodes are
        integer indices, edges connect single-mutation neighbors pointing
        towards higher fitness (if `maximize=True`). Populated after
        `from_data`.
    configs : pandas.Series or None
        Maps node index to the integer-encoded sequence tuple. Populated after
        `from_data`.
    config_dict : dict or None
        Describes the encoding for sequence positions (all typically
        'categorical' with 'max' based on alphabet size). Populated after
        `from_data`.
    data_types : dict or None
        Inferred data types, typically {'pos_0': 'categorical', ...}.
        Populated after `from_data`.
    n_configs : int or None
        Number of unique sequences in the landscape. Populated after
        `from_data`.
    n_vars : int or None
        Length of the sequences (`sequence_length`). Populated after
        `from_data`.
    n_edges : int or None
        Number of edges (single mutations leading to higher fitness).
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
    _seq_map : dict[str, int]
        Internal mapping from sequence character to integer encoding.
    _seq_map_rev : dict[int, str]
        Internal mapping from integer encoding back to sequence character.
    _is_built : bool
        Internal flag for build status.

    See Also
    --------
    BaseLandscape : The base class for all landscapes.
    DNALandscape : Landscape specific to DNA sequences.
    RNALandscape : Landscape specific to RNA sequences.
    ProteinLandscape : Landscape specific to protein sequences.
    Landscape : Factory class to create appropriate landscape types.

    References
    ----------
    .. [Papkou 2023] Papkou, A. et al. A rugged yet easily navigable
       fitness landscape. Science 382, eadh3860 (2023). (Example of DHFR landscape)
    .. [Pitt 2010] Pitt, J. N. & Ferré-D'Amaré, A. R. Rapid construction
       of empirical RNA fitness landscapes. Science 330, 376-379 (2010).
    .. [Poelwijk 2019] Poelwijk, F. J. et al. Learning the pattern of epistasis
       linking genotype and phenotype in a protein. Nat Commun 10, 4213 (2019).
    """

    ALPHABET: List[str] = []
    _seq_map: Dict[str, int] = {}
    _seq_map_rev: Dict[int, str] = {}
    sequence_length: Optional[int] = None

    def from_data(
        self,
        X: Union[List[str], pd.Series, np.ndarray, pd.DataFrame],
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
    ) -> "SequenceLandscape":
        """Construct the sequence landscape graph and properties from data.

        This method specializes `BaseLandscape.from_data` for sequence data.
        It accepts sequences in various formats (list of strings, Series, etc.),
        validates them against the subclass's `ALPHABET`, encodes them, and
        infers the appropriate `data_types` before calling the base class
        construction method.

        Parameters
        ----------
        X : list[str], pandas.Series, numpy.ndarray, or pandas.DataFrame
            Sequence data. Can be a list/Series of sequences (strings), a
            NumPy array of sequences, or a DataFrame where columns represent
            sequence positions. Assumes all sequences have the same length.
        f : pandas.Series, list, or numpy.ndarray
            Fitness values corresponding to each sequence in `X`. Must have the
            same length as `X`.
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
            Edit distance for neighborhood definition. For `SequenceLandscape`,
            this is typically fixed to 1 (single point mutations). Values
            other than 1 might be ignored or raise warnings depending on the
            implementation of `_generate_neighbors`.

        Returns
        -------
        self : SequenceLandscape
            The populated landscape instance.

        Raises
        ------
        NotImplementedError
            If the `ALPHABET` class attribute is not defined in a subclass.
        ValueError
            If input data is invalid (e.g., inconsistent sequence lengths,
            characters outside the alphabet).
        RuntimeError
            If the landscape has already been built.
        TypeError
            If input types are incorrect.
        ImportError
            If `impute=True` and scikit-learn is not installed, unless an
            `impute_model` is provided.
        """
        if not self.ALPHABET:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define the ALPHABET class attribute."
            )

        effective_verbose = verbose if verbose is not None else self.verbose

        # Preprocess sequences into a standardized DataFrame format
        # and infer data types (all categorical for sequences).
        X_df, sequence_data_types, seq_len = _preprocess_sequence_input(
            X_input=X,
            alphabet=self.ALPHABET,
            class_name=self.__class__.__name__,
            verbose=effective_verbose,
        )
        self.sequence_length = seq_len
        # Create internal mapping from characters to integers (0, 1, ...)
        self._seq_map = {char: i for i, char in enumerate(self.ALPHABET)}
        self._seq_map_rev = {i: char for char, i in self._seq_map.items()}

        # Call the BaseLandscape.from_data method with processed inputs
        super().from_data(
            X=X_df,
            f=f,
            data_types=sequence_data_types,  # Use inferred types
            maximize=maximize,
            epsilon=epsilon,
            calculate_basins=calculate_basins,
            calculate_paths=calculate_paths,
            impute=impute,
            impute_model=impute_model,
            verbose=verbose,
            n_edit=n_edit,  # Pass n_edit, though _generate_neighbors might ignore > 1
        )
        return self

    def _generate_neighbors(
        self, config: Tuple, config_dict: Dict, n_edit: int
    ) -> List[Tuple]:
        """Generates neighbors for a sequence (single point mutations).

        Overrides `BaseLandscape._generate_neighbors`. For sequence landscapes,
        neighbors are typically defined as sequences reachable by a single
        nucleotide or amino acid substitution (n_edit=1). This implementation
        ignores `config_dict` as the alphabet is class-defined.

        Parameters
        ----------
        config : tuple
            The integer-encoded sequence tuple.
        config_dict : dict
            Ignored in this implementation.
        n_edit : int
            Edit distance. Only `n_edit=1` is supported.

        Returns
        -------
        list[tuple]
            List of neighboring integer-encoded sequence tuples.
        """
        # Sequence landscapes typically only consider single point mutations.
        if n_edit != 1:
            if self.verbose:
                warnings.warn(
                    f"{self.__class__.__name__}._generate_neighbors typically uses "
                    f"n_edit=1. Received {n_edit}. Returning no neighbors.",
                    UserWarning,
                )
            return []  # Return empty list if n_edit is not 1

        neighbors = []
        current_config_list = list(config)
        num_vars = len(current_config_list)  # Should equal self.sequence_length
        num_chars_in_alphabet = len(self.ALPHABET)

        for i in range(num_vars):  # Iterate through each position
            original_encoded_char = current_config_list[i]
            # Try substituting with every other character in the alphabet
            for new_encoded_char in range(num_chars_in_alphabet):
                if new_encoded_char != original_encoded_char:
                    # Create the new neighbor sequence
                    neighbor_list = current_config_list.copy()
                    neighbor_list[i] = new_encoded_char
                    neighbors.append(tuple(neighbor_list))
        return neighbors

    def _get_default_distance_metric(self):
        """Returns Hamming distance as the default for sequences."""
        return hamming_distance


class DNALandscape(SequenceLandscape):
    """A fitness landscape defined over DNA sequences.

    This class specializes `SequenceLandscape` for DNA, using the alphabet
    ['A', 'C', 'G', 'T']. It inherits all methods and attributes from
    `SequenceLandscape` and `BaseLandscape`.

    Parameters
    ----------
    verbose : bool, default=True
        Controls the verbosity of the output.

    Attributes
    ----------
    ALPHABET : list[str]
        Defined as ['A', 'C', 'G', 'T'].
    (Other attributes inherited from SequenceLandscape and BaseLandscape)

    Examples
    --------
    >>> from landscape_lib import Landscape # Assuming library structure
    >>> dna_sequences = ["ACG", "AGT", "AAG", "ACT"]
    >>> fitness_values = [1.5, 2.1, 0.8, 1.9]
    >>> landscape = Landscape.from_data(dna_sequences, fitness_values, data_types="dna")
    >>> print(landscape.sequence_length)
    3
    >>> print(landscape.get_data().head()) # doctest: +SKIP
           fitness pos_0 pos_1 pos_2 sequence  ...
    0      1.5     A     C     G      ACG     ...
    1      2.1     A     G     T      AGT     ...
    2      0.8     A     A     G      AAG     ...
    3      1.9     A     C     T      ACT     ...
    """

    ALPHABET = DNA_ALPHABET


class RNALandscape(SequenceLandscape):
    """A fitness landscape defined over RNA sequences.

    This class specializes `SequenceLandscape` for RNA, using the alphabet
    ['A', 'C', 'G', 'U']. It inherits all methods and attributes from
    `SequenceLandscape` and `BaseLandscape`.

    Parameters
    ----------
    verbose : bool, default=True
        Controls the verbosity of the output.

    Attributes
    ----------
    ALPHABET : list[str]
        Defined as ['A', 'C', 'G', 'U'].
    (Other attributes inherited from SequenceLandscape and BaseLandscape)
    """

    ALPHABET = RNA_ALPHABET


class ProteinLandscape(SequenceLandscape):
    """A fitness landscape defined over protein amino acid sequences.

    This class specializes `SequenceLandscape` for proteins, using the standard
    20 amino acid single-letter codes as the alphabet. It inherits all methods
    and attributes from `SequenceLandscape` and `BaseLandscape`.

    Parameters
    ----------
    verbose : bool, default=True
        Controls the verbosity of the output.

    Attributes
    ----------
    ALPHABET : list[str]
        Defined as the 20 standard amino acid codes:
        ['A', 'C', 'D', ..., 'Y'].
    (Other attributes inherited from SequenceLandscape and BaseLandscape)
    """

    ALPHABET = PROTEIN_ALPHABET
