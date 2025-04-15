import pandas as pd
import numpy as np
from typing import List, Any, Dict, Union, Optional

from ..base import BaseLandscape
from .sequence import DNALandscape, RNALandscape, ProteinLandscape
from .boolean import BooleanLandscape


class Landscape:
    """Factory class for creating specific fitness landscape instances.

    This class acts as a convenient entry point for creating landscape objects.
    Instead of directly instantiating `BaseLandscape` or its subclasses, users
    should typically use the `Landscape.from_data` class method. This method
    inspects the `data_types` argument and automatically chooses and constructs
    the most appropriate landscape class (`DNALandscape`, `RNALandscape`,
    `ProteinLandscape`, `BooleanLandscape`, or the generic `BaseLandscape`).

    Direct instantiation of the `Landscape` class itself is disabled.

    See Also
    --------
    BaseLandscape : The base class from which specific landscapes inherit.
    SequenceLandscape : Base class for sequence-based landscapes.
    DNALandscape : Landscape for DNA sequences.
    RNALandscape : Landscape for RNA sequences.
    ProteinLandscape : Landscape for protein sequences.
    BooleanLandscape : Landscape for binary sequences.

    References
    ----------
    The concept of fitness landscapes is foundational in evolutionary biology
    and related fields. Key ideas include:
    .. [Wright 1932] Wright, S. The roles of mutation, inbreeding,
       crossbreeding and selection in evolution. Proc. Sixth Intl. Cong. Genetics
       1, 356-366 (1932).
    .. [Kauffman 1993] Kauffman, S. A. The Origins of Order: Self-Organization
       and Selection in Evolution. Oxford University Press (1993). (NK Landscapes)
    .. [Gavrilets 2004] Gavrilets, S. Fitness Landscapes and the Origin of Species.
       Princeton University Press (2004). (Holey Landscapes, Speciation)
    .. [Poelwijk 2007] Poelwijk, F. J. et al. Empirical fitness landscapes
       reveal accessible evolutionary paths. Nature 445, 383-386 (2007).
    .. [de Visser 2014] de Visser, J. A. G. M. & Krug, J. Empirical fitness
       landscapes and the predictability of evolution. Nat Rev Genet 15,
       480-490 (2014). (Review)
    .. [Fragata 2018] Fragata, I. et al. Evolution in the light of fitness
       landscape theory. Trends Ecol Evol 34, 69-82 (2018). (Review)
    .. [Bank 2022] Bank, C. Epistasis and Adaptation on Fitness Landscapes.
       Annu. Rev. Ecol. Evol. Syst. 53, 457-479 (2022). (Review)

    Examples
    --------
    >>> from landscape_lib import Landscape
    >>> import pandas as pd

    >>> # Example 1: DNA Landscape
    >>> dna_seqs = ["ATT", "AGT", "GGT", "GTT"]
    >>> dna_fitness = [0.5, 0.8, 0.2, 0.9]
    >>> dna_landscape = Landscape.from_data(dna_seqs, dna_fitness, data_types="dna")
    >>> print(type(dna_landscape))
    <class 'landscape_lib.landscape.DNALandscape'>
    >>> dna_landscape.describe()
    --- Landscape Summary ---
    Class: DNALandscape
    Built: True
    Variables (n_vars): 3
    Configurations (n_configs): 4
    Connections (n_edges): 3
    Local Optima (n_lo): 1
    Global Optimum Index: 3
    Maximize Fitness: True
    Basins Calculated: True
    Paths Calculated: False
    LON Calculated: False
    ---

    >>> # Example 2: Boolean Landscape
    >>> bool_configs = ["00", "01", "10", "11"]
    >>> bool_fitness = [1, 3, 2, 4]
    >>> bool_landscape = Landscape.from_data(bool_configs, bool_fitness, data_types="boolean")
    >>> print(type(bool_landscape))
    <class 'landscape_lib.landscape.BooleanLandscape'>
    >>> print(bool_landscape.go) # doctest: +SKIP
    {'fitness': 4, 'bit_0': 1, 'bit_1': 1, 'out_degree': 0, 'in_degree': 2, 'is_lo': True, 'basin_index': 3, 'size_basin_greedy': 4, 'radius_basin_greedy': 1, 'dist_go': 0.0, 'bitstring': '11'}

    >>> # Example 3: Generic Categorical/Ordinal Landscape
    >>> X_generic = pd.DataFrame({'feat1': ['A', 'A', 'B', 'B'], 'feat2': [1, 2, 1, 2]})
    >>> f_generic = [10, 12, 15, 11]
    >>> types_generic = {'feat1': 'categorical', 'feat2': 'ordinal'}
    >>> generic_landscape = Landscape.from_data(X_generic, f_generic, data_types=types_generic)
    >>> print(type(generic_landscape))
    <class 'landscape_lib.landscape.BaseLandscape'>
    """

    def __init__(self):
        """Direct instantiation is disabled. Use Landscape.from_data()."""
        raise NotImplementedError(
            "Cannot directly instantiate Landscape. Use the "
            "Landscape.from_data() class method to create an instance of the "
            "appropriate landscape type (e.g., DNALandscape, BooleanLandscape, "
            "BaseLandscape)."
        )

    @classmethod
    def from_data(
        cls,
        X: Union[pd.DataFrame, np.ndarray, List[str], List[Any], pd.Series],
        f: Union[pd.Series, list, np.ndarray],
        data_types: Union[Dict[str, str], str],
        *,
        maximize: bool = True,
        epsilon: Union[float, str] = "auto",
        calculate_basins: bool = True,
        calculate_paths: bool = False,
        impute: bool = False,
        impute_model: Optional[Any] = None,
        verbose: bool = True,
        n_edit: int = 1,
    ) -> BaseLandscape:
        """Constructs and returns a specific Landscape instance based on data.

        This factory method determines the appropriate landscape class
        (e.g., `DNALandscape`, `BooleanLandscape`, `BaseLandscape`) based on the
        `data_types` parameter and constructs an instance using the provided
        configuration data `X` and fitness values `f`.

        Parameters
        ----------
        X : pandas.DataFrame, numpy.ndarray, list[str], list[Any], or pandas.Series
            Configuration data (genotypes). The format depends on the landscape
            type specified by `data_types`.
            - For `data_types` dict: Typically pd.DataFrame or np.ndarray where
              columns correspond to keys in `data_types`.
            - For "dna", "rna", "protein": Typically list/Series of strings, or
              DataFrame/ndarray representing sequences.
            - For "boolean": Typically list/Series of bitstrings, list of lists
              of 0/1, or DataFrame/ndarray of 0/1 or True/False.
        f : pandas.Series, list, or numpy.ndarray
            Fitness values corresponding to each configuration in `X`. Must
            have the same length as `X`.
        data_types : dict[str, str] or str
            Specifies the type of landscape and how to interpret `X`.
            - If dict: Used for `BaseLandscape`. Keys are variable names (must
              match columns if `X` is DataFrame), values are 'boolean',
              'categorical', or 'ordinal'.
            - If str: Specifies a specialized subclass. Must be one of
              "dna", "rna", "protein", or "boolean". The appropriate subclass
              (e.g., `DNALandscape`) will be instantiated, and it will infer
              the column data types automatically.
        maximize : bool, default=True
            Determines the optimization direction. If True, the landscape seeks
            higher fitness values (peaks are optima). If False, it seeks lower
            fitness values (valleys are optima). Affects optima identification
            and graph edge direction.
        epsilon : float or 'auto', default='auto'
            Tolerance value used for floating-point comparisons when determining
            fitness improvements or optima.
        calculate_basins : bool, default=True
            If True, calculates the basins of attraction for each local optimum
            using a hill-climbing algorithm. Populates the `basin_index`
            attribute on the returned landscape object.
        calculate_paths : bool, default=False
            If True, calculates accessible paths (ancestors) for local optima.
            This uses `networkx.ancestors` and can be computationally intensive.
            Populates the `size_basin_accessible` node attribute. Skipped if
            number of configurations > 200,000.
        impute : bool, default=False
            If True, attempts to fill in missing fitness values (`NaN` in `f`)
            using a regression model based on the configurations `X`. Requires
            scikit-learn or a user-provided `impute_model`.
        impute_model : object, optional
            A custom model object with `fit` and `predict` methods used for
            imputation if `impute=True`. If None and `impute=True`, defaults
            to scikit-learn's `RandomForestRegressor` (if available).
        verbose : bool, default=True
            Controls printed output during the construction process. Passed to
            the constructor of the chosen landscape class.
        n_edit : int, default=1
            The edit distance defining the neighborhood for constructing the
            landscape graph. An edge connects configurations `u` and `v` if
            distance(u, v) <= `n_edit`. For sequence and boolean landscapes, this
            is typically fixed to 1 internally, representing single point
            mutations or bit flips.

        Returns
        -------
        BaseLandscape or subclass instance
            A populated instance of the appropriate landscape class
            (e.g., `DNALandscape`, `BooleanLandscape`, `BaseLandscape`).

        Raises
        ------
        ValueError
            If `data_types` is not a valid dictionary or recognized string identifier.
            If input data `X`, `f` are invalid (e.g., mismatched lengths, empty).
            If `X` format is incompatible with the specified `data_types` string.
        TypeError
            If `data_types` is not a string or dictionary.
            If `X` or `f` cannot be processed into the required internal formats.
        ImportError
            If `impute=True` and scikit-learn is not installed, unless an
            `impute_model` is provided.

        Attributes Available on the Returned Instance
        --------------------------------------------
        (Attributes are populated by the internal `from_data` call)
        graph : networkx.DiGraph or None
            The directed graph representing the fitness landscape.
        configs : pandas.Series or None
            Mapping from node indices to configuration representations.
        config_dict : dict or None
            Description of the configuration variable encoding.
        data_types : dict or None
            Validated dictionary of variable data types.
        n_configs : int or None
            Total number of configurations (nodes).
        n_vars : int or None
            Number of variables defining a configuration (dimensionality).
        n_edges : int or None
            Total number of directed edges in the graph.
        n_lo : int or None
            Number of local optima.
        lo_index : list[int] or None
            Sorted list of local optima node indices.
        go_index : int or None
            Node index of the global optimum.
        go : dict or None
            Attributes dictionary of the global optimum node.
        basin_index : dict[int, int] or None
            Mapping from node index to its basin's local optimum index (if calculated).
        lon : networkx.DiGraph or None
            Local Optima Network graph (if `get_lon` is called).
        has_lon : bool
            Flag indicating if the LON has been calculated.
        maximize : bool
            The optimization direction (True for maximization).
        sequence_length : int or None
            (SequenceLandscape subclasses) Length of sequences.
        bit_length : int or None
            (BooleanLandscape) Length of bitstrings.
        verbose : bool
            The verbosity level.

        Methods Available on the Returned Instance
        ------------------------------------------
        get_data(lo_only=False) : pandas.DataFrame
            Returns landscape data as a DataFrame. May include 'sequence' or
            'bitstring' columns for relevant subclasses.
        describe() : None
            Prints a summary of the landscape.
        get_lon(...) : networkx.DiGraph
            Computes and returns the Local Optima Network.
        apply(X) : numpy.ndarray
            Returns the leaf node indices for input samples `X`.
        decision_path(X) : tuple[scipy.sparse.csr_matrix, numpy.ndarray]
            Returns the decision path indicators for input samples `X`.
        __len__() : int
            Returns the number of configurations (`n_configs`).
        __getitem__(index) : dict
            Returns the attributes of the node with the given `index`.
        __iter__() : iterator
            Iterates over the node indices.
        __contains__(item) : bool
            Checks if a node index `item` exists.
        (Plus other methods inherited from BaseLandscape)
        """
        landscape_class: type[BaseLandscape]

        # Arguments for the chosen class's __init__
        init_kwargs = {"verbose": verbose}
        # Arguments for the chosen class's from_data method
        # (Common args minus verbose, which goes to init)
        from_data_kwargs = {
            "maximize": maximize,
            "epsilon": epsilon,
            "calculate_basins": calculate_basins,
            "calculate_paths": calculate_paths,
            "impute": impute,
            "impute_model": impute_model,
            "n_edit": n_edit,
        }

        if isinstance(data_types, str):
            dt_str = data_types.lower()
            if dt_str == "dna":
                landscape_class = DNALandscape
            elif dt_str == "rna":
                landscape_class = RNALandscape
            elif dt_str == "protein":
                landscape_class = ProteinLandscape
            elif dt_str == "boolean":
                landscape_class = BooleanLandscape
            else:
                raise ValueError(
                    f"Unknown data_types string: '{data_types}'. "
                    "Expected 'dna', 'rna', 'protein', or 'boolean'."
                )

            # Instantiate the specific subclass (e.g., DNALandscape)
            instance = landscape_class(**init_kwargs)
            # Call its specialized from_data method
            # Note: Sequence/BooleanLandscape.from_data does not take data_types arg
            instance.from_data(X=X, f=f, **from_data_kwargs)
            return instance

        elif isinstance(data_types, dict):
            # Use the BaseLandscape for generic dictionary-defined types
            landscape_class = BaseLandscape
            # Instantiate BaseLandscape
            instance = landscape_class(**init_kwargs)
            # Call its standard from_data method, passing the data_types dict
            from_data_kwargs["data_types"] = data_types
            instance.from_data(X=X, f=f, **from_data_kwargs)
            return instance

        else:
            # Handle invalid data_types type
            raise TypeError(
                f"Unsupported type for data_types: {type(data_types)}. "
                "Expected Dict[str, str] or one of 'dna', 'rna', 'protein', 'boolean'."
            )

    @classmethod
    def from_graph(
        cls,
        filepath: str,
        *,
        verbose: bool = True,
        calculate_basins: bool = False,
        calculate_paths: bool = False,
        analyze_distance: bool = False,
    ) -> BaseLandscape:
        """Construct a landscape from a saved graph file.

        This factory method loads a previously saved landscape graph from a file and
        returns an instance of the appropriate landscape class based on the metadata
        stored in the graph.

        Parameters
        ----------
        filepath : str
            Path to the saved graph file (.graphml).
        verbose : bool, default=True
            Controls verbosity of output during loading and analysis.
        calculate_basins : bool, default=False
            Whether to (re)calculate basins of attraction after loading.
        calculate_paths : bool, default=False
            Whether to (re)calculate accessible paths after loading.
        analyze_distance : bool, default=False
            Whether to calculate distance metrics to global optimum.

        Returns
        -------
        BaseLandscape or subclass instance
            A populated instance of the appropriate landscape class
            (e.g., `DNALandscape`, `BooleanLandscape`, `BaseLandscape`).

        Raises
        ------
        ValueError
            If the file cannot be read or doesn't contain valid graph data.
        FileNotFoundError
            If the specified file doesn't exist.

        Examples
        --------
        >>> from graphfla.landscape import Landscape
        >>> # Load a previously saved landscape
        >>> landscape = Landscape.from_graph("my_landscape.graphml")
        >>> landscape.describe()
        """
        try:
            import igraph as ig

            graph = ig.Graph.Read_GraphML(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to load graph from {filepath}: {e}")

        # Get the landscape class from the graph metadata, defaulting to BaseLandscape
        if "landscape_class" in graph.attributes():
            landscape_class_name = graph["landscape_class"]
        else:
            landscape_class_name = "BaseLandscape"

        # Map string class names to actual classes
        class_mapping = {
            "DNALandscape": DNALandscape,
            "RNALandscape": RNALandscape,
            "ProteinLandscape": ProteinLandscape,
            "BooleanLandscape": BooleanLandscape,
            "BaseLandscape": BaseLandscape,
        }

        # Get the appropriate class
        target_class = class_mapping.get(landscape_class_name, BaseLandscape)
        if verbose:
            print(f"Loading {landscape_class_name} instance from graph file...")

        # Call the class's from_graph method
        return target_class.from_graph(
            filepath=filepath,
            verbose=verbose,
            calculate_basins=calculate_basins,
            calculate_paths=calculate_paths,
            analyze_distance=analyze_distance,
        )
