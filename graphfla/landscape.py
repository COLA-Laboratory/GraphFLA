import pandas as pd
import numpy as np
import igraph as ig
import warnings
import time

from typing import Tuple, Dict, List, Union, Optional, Any
from collections import defaultdict
from tqdm import tqdm

from .lon import get_lon
from .algorithms import hill_climb
from .utils import add_network_metrics
from .distances import mixed_distance, hamming_distance
from .filters import LandscapeFilter

from ._processor import (
    DataPreprocessor,
    BooleanPreprocessor,
    DefaultPreprocessor,
    SequencePreprocessor,
)

from ._neighbors import (
    NeighborGenerator,
    BooleanNeighborGenerator,
    DefaultNeighborGenerator,
    SequenceNeighborGenerator,
)

from functools import wraps


def timeit(method):
    """
    A decorator to measure and log the execution time of a method.
    """

    @wraps(method)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Method {method.__name__} executed in {elapsed_time:.4f} seconds.")
        return result

    return timed


ALLOWED_DATA_TYPES = {"boolean", "categorical", "ordinal"}
DNA_ALPHABET = ["A", "C", "G", "T"]
RNA_ALPHABET = ["A", "C", "G", "U"]
PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


class Landscape:
    """Class implementing the fitness landscape object.

    This class provides a foundational structure for fitness landscapes,
    conceptualized as a mapping from a genotype (configuration) space to
    a fitness value. It typically represents the landscape as a directed graph
    where nodes are genotypes and edges connect mutational neighbors, pointing
    towards fitter variants.

    The landscape can be either constructed from raw data (using `build_from_data()`) or
    leveraging existing graph (using `build_from_graph()`).

    After construction:
    - The graph object representing the landscape can be accessed via `Landscape.graph`.
    - Tabular information about the configurations can be obtained with `get_data()`.
    - Basic landscape properties is available via `describe()`.
    - Other methods in the `graphfla.analysis` and `graphfla.plotting` modules can be
      used for advanced analysis and visualization.

    Parameters
    ----------
    type : str, default='default'
        The type of landscape to create. This determines the preprocessing and
        neighbor generation strategies used. Options include 'boolean', 'dna',
        'rna', 'protein', or 'default' for general landscapes.
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values (peaks are optima). If False, it seeks lower
        fitness values (valleys are optima).

    Attributes
    ----------
    graph : ig.Graph or None
        The directed ig.Graph representing the fitness landscape. Nodes represent
        configurations (genotypes) and edges connect neighboring configurations,
        typically pointing from lower to higher fitness if `maximize` is True.
        Each node usually has a 'fitness' attribute. Populated after calling
        `build_from_data` or `build_from_graph`. Fitness difference between neighboring nodes
        is stored in the 'delta_fit' attribute.
    configs : pandas.Series or None
        A pandas Series mapping node indices (int) to their corresponding
        configuration representation (often a tuple). This represents the
        genotypes in the landscape. Populated after calling `build_from_data`.
    config_dict : dict or None
        A dictionary describing the encoding scheme for configuration variables.
        Keys are typically integer indices of variables, and values are
        dictionaries specifying properties like 'type' (e.g., 'boolean',
        'categorical') and 'max' (maximum encoded value). Populated after
        calling `build_from_data`.
    data_types : dict or None
        A dictionary specifying the data type for each variable in the
        configuration space (e.g., {'var_0': 'boolean', 'var_1': 'categorical'}).
        Validated and stored during `build_from_data`. Required for certain distance
        calculations.
    n_configs : int or None
        The total number of configurations (nodes) in the landscape graph.
        Populated after calling `build_from_data` or `build_from_graph`.
    n_vars : int or None
        The number of variables (dimensions) defining a configuration in the
        genotype space. Populated after calling `build_from_data` or inferred
        by `build_from_graph`.
    n_edges : int or None
        The total number of directed edges (connections) in the landscape graph.
        Populated after calling `build_from_data` or `build_from_graph`.
    n_lo : int or None
        The number of local optima (peaks or valleys depending on `maximize`)
        in the landscape. Local optima are nodes with no outgoing edges to
        fitter neighbors (Papkou 2023). Populated after graph analysis.
    lo_index : list[int] or None
        A sorted list of the node indices corresponding to local optima.
        Populated after graph analysis.
    go_index : int or None
        The node index of the global optimum (the configuration with the
        highest or lowest fitness). Populated after graph analysis.
    go : dict or None
        A dictionary containing the attributes (including 'fitness') of the
        global optimum node. Populated after graph analysis.
    lon : ig.graph or None
        The Local Optima Network (LON) graph, if calculated via `get_lon`.
        Nodes in the LON are local optima from the main landscape graph, and
        edges represent accessibility between their basins (Ochoa 2021).
    has_lon : bool
        Flag indicating whether the LON has been calculated and stored in the
        `lon` attribute.
    maximize : bool
        Indicates whether the objective is to maximize (True) or minimize
        (False) the fitness values. Set during `build_from_data` or `build_from_graph`.
    epsilon : float or str
        Tolerance for floating point comparisons, potentially used in
        determining neighbors or optima. Set during `build_from_data` or `build_from_graph`.
    verbose : bool
        The verbosity level set during initialization or construction.
    _is_built : bool
        Internal flag indicating if the landscape has been populated via
        `build_from_data` or `build_from_graph`.


    References
    ----------
    .. [Wright 1932] Wright, S. The roles of mutation, inbreeding,
       crossbreeding and selection in evolution. Proceedings Sixth
       International Congress Genetics 1, 356-366 (1932).

    .. [Papkou 2023] Papkou, A. et al. A rugged yet easily navigable
       fitness landscape. Science 382, eadh3860 (2023).

    .. [Li 2016] Li, C. et al. The fitness landscape of a tRNA gene.
       Science 352, 837-840 (2016).

    .. [Puchta 2016] Puchta, O. et al. Network of epistatic interactions
       within a yeast snoRNA. Science 352, 840-844 (2016).

    .. [Poelwijk 2007] Poelwijk, F. J. et al. Empirical fitness landscapes
       reveal accessible evolutionary paths. Nature 445, 383-386 (2007).

    .. [Carneiro 2010] Carneiro, M. & Hartl, D. L. Adaptive landscapes and
       protein evolution. PNAS 107, 1747-1751 (2010).

    .. [Ochoa 2021] Ochoa, G. et al. Local optima networks: A survey.
       Journal of Heuristics 27, 79-134 (2021).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    # Example Data (replace with actual data)
    >>> X_data = pd.DataFrame({'var_0': [0, 0, 1, 1], 'var_1': [0, 1, 0, 1]})
    >>> f_data = pd.Series([1.0, 2.0, 3.0, 2.5])
    >>> data_types_dict = {'var_0': 'boolean', 'var_1': 'boolean'}

    >>> # Use the Landscape factory for automatic type selection (recommended)
    >>> # from landscape_lib import Landscape # Assuming library structure
    >>> # landscape = Landscape(type="boolean").build_from_data(X_data, f_data)

    ...
    Landscape Summary
    --- Landscape Summary ---
    Class: BaseLandscape
    Built: True
    Variables (n_vars): 2
    Configurations (n_configs): 4
    Connections (n_edges): 4
    Local Optima (n_lo): 1
    Global Optimum Index: 2
    Maximize Fitness: True
    Basins Calculated: True
    Paths Calculated: False
    LON Calculated: False
    ---

    >>> print(f"Number of configurations: {landscape.n_configs}")
    Number of configurations: 4
    >>> print(f"Global optimum fitness: {landscape.go['fitness']}")
    Global optimum fitness: 3.0
    """

    # Class-level registries for strategies
    _preprocessors = {
        "boolean": BooleanPreprocessor(),
        "dna": SequencePreprocessor(DNA_ALPHABET),
        "rna": SequencePreprocessor(RNA_ALPHABET),
        "protein": SequencePreprocessor(PROTEIN_ALPHABET),
        "default": DefaultPreprocessor(),
    }

    _neighbor_generators = {
        "boolean": BooleanNeighborGenerator(),
        "dna": SequenceNeighborGenerator(len(DNA_ALPHABET)),
        "rna": SequenceNeighborGenerator(len(RNA_ALPHABET)),
        "protein": SequenceNeighborGenerator(len(PROTEIN_ALPHABET)),
        "default": DefaultNeighborGenerator(),
    }

    def __init__(self, type: str = "default", maximize: bool = True):
        # Core attributes
        self.graph = None
        self.configs = None
        self.config_dict = None
        self.data_types = None
        self.n_configs = None
        self.n_vars = None
        self.n_edges = None
        self.n_lo = None
        self.lo_index = None
        self.go_index = None
        self.go = None
        self.lon = None
        self.has_lon = False
        self.type = type

        # Landscape construction parameters
        self.maximize = maximize
        self.epsilon = "auto"

        # Build status flags
        self._is_built = False
        self._path_calculated = False
        self._basin_calculated = False
        self._distance_calculated = False
        self._neighbor_fit_calculated = False

        # Strategy objects (will be set in build_from_data)
        self._preprocessor = None
        self._neighbor_generator = None

    @property
    def shape(self):
        """Return the shape (n_configs, n_edges) of the landscape graph."""
        self._check_built()
        if self.graph is None:
            return (0, 0)
        return (self.graph.vcount(), self.graph.ecount())

    def __getitem__(self, index):
        """Return the node attributes dictionary for the given index."""
        self._check_built()
        if self.graph is None:
            raise RuntimeError("Graph is None.")

        try:
            # If index is an integer within bounds, use positional indexing.
            if isinstance(index, int) and 0 <= index < self.graph.vcount():
                vertex = self.graph.vs[index]
            else:
                # Otherwise, assume the vertex 'name' attribute is used.
                vertex = self.graph.vs.find(name=index)
            # Return the vertex attributes as a dictionary.
            return vertex.attributes()
        except (IndexError, ValueError):
            raise KeyError(f"Index {index} not found among landscape configurations.")

    def __str__(self):
        """Return a string summary of the landscape."""
        if not self._is_built:
            return f"{self.__class__.__name__} object (uninitialized)"

        n_configs = self.graph.vcount() if self.graph is not None else "?"
        n_edges = self.graph.ecount() if self.graph is not None else "?"
        n_vars_str = str(self.n_vars) if self.n_vars is not None else "?"
        n_lo_str = str(self.n_lo) if self.n_lo is not None else "?"

        return (
            f"{self.__class__.__name__} with {n_vars_str} variables, "
            f"{n_configs} configurations, {n_edges} connections, "
            f"and {n_lo_str} local optima."
        )

    def __repr__(self):
        """Return a concise string representation of the landscape."""
        return self.__str__()

    def __len__(self):
        """Return the number of configurations (nodes) in the landscape."""
        self._check_built()
        return self.graph.vcount() if self.graph is not None else 0

    def __iter__(self):
        """Iterate over the configuration indices (nodes) in the landscape."""
        self._check_built()
        if self.graph is None:
            raise RuntimeError("Graph is None.")
        return (v.index for v in self.graph.vs)

    def __contains__(self, item):
        """Check if a configuration index (node) exists in the landscape."""
        self._check_built()
        if self.graph is None:
            raise RuntimeError("Graph is None.")
        try:
            if isinstance(item, int):
                # Check bounds if item is an index.
                return 0 <= item < self.graph.vcount()
            else:
                # Try to find a vertex with a matching 'name' attribute.
                self.graph.vs.find(name=item)
                return True
        except ValueError:
            return False

    def __eq__(self, other):
        """Compare two BaseLandscape instances for equality.

        Equality is based on graph structure isomorphism and the optimization direction (`maximize`).
        """
        if not isinstance(other, Landscape):
            return NotImplemented
        if not self._is_built or not other._is_built:
            # Consider unbuilt instances equal only if both are unbuilt.
            return self._is_built == other._is_built
        if self.graph is None or other.graph is None:
            # Consider None graphs equal only if both are None.
            return self.graph is None and other.graph is None

        return self.graph.isomorphic(other.graph) and self.maximize == other.maximize

    def __bool__(self):
        """Return True if the landscape is built and has configurations."""
        return self._is_built and self.graph is not None and self.graph.vcount() > 0

    @classmethod
    def register_preprocessor(
        cls, data_type: str, preprocessor: DataPreprocessor
    ) -> None:
        """Register a custom preprocessor for a data type."""
        cls._preprocessors[data_type] = preprocessor

    @classmethod
    def register_neighbor_generator(
        cls, data_type: str, generator: NeighborGenerator
    ) -> None:
        """Register a custom neighbor generator for a data type."""
        cls._neighbor_generators[data_type] = generator

    @timeit
    def build_from_data(
        self,
        X: Any,
        f: Union[pd.Series, list, np.ndarray],
        data_types: Optional[Dict[str, str]] = None,
        epsilon: Union[float, str] = "auto",
        calculate_basins: bool = True,
        calculate_paths: bool = True,
        calculate_distance: bool = True,
        calculate_neighbor_fit: bool = True,
        impute: bool = False,
        impute_model: Optional[Any] = None,
        filter: Optional["LandscapeFilter"] = None,
        n_edit: int = 1,
        verbose: Optional[bool] = True,
    ) -> None:
        """Construct the landscape graph and properties from configuration data.

        This method takes genotype-phenotype data (configurations `X` and their
        corresponding fitness values `f`) and builds the underlying graph
        structure of the fitness landscape. Nodes represent configurations, and
        edges connect neighbors based on the specified edit distance (`n_edit`).
        It then calculates various basic landscape properties like local optima,
        basins of attraction, and the global optimum (if specified).

        This method populates the core attributes of the `Landscape` instance.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The configuration data, where each row represents a genotype or
            configuration, and columns represent variables or sites.
        f : pandas.Series, list, or numpy.ndarray
            The fitness values corresponding to each configuration in `X`. Must
            have the same length as `X`.
        data_types : dict[str, str]
            A dictionary specifying the type of each variable (column in `X` if
            DataFrame, or inferred column index if ndarray). Keys must match
            column names/indices, and values must be one of 'boolean',
            'categorical', or 'ordinal'. This information is crucial for
            determining neighborhood relationships and calculating distances.
        epsilon : float or 'auto', default='auto'
            Noise tolerance value used for floating-point comparisons when determining
            fitness improvements or optima. If 'auto', a default might be used.
            Typically, a high tolerance would result in a smoother landscape, whereas
            a low tolerance would yield a more rugged landscape.
        calculate_basins : bool, default=True
            If True, calculates the basins of attraction for each local optimum
            using a greedy hill-climbing algorithm. This identifies which configurations
            lead to which peak. Populates the `size_basin_greedy` and `radius_basin_greedy`
            attributes for each local optimum.
        calculate_paths : bool, default=False
            If True, calculates accessible paths (ancestors) for local optima.
            This can be computationally intensive for large landscapes. Populates the
            `size_basin_accessible` attribute for each local optimum.
        impute : bool, default=False
            If True, attempts to fill in missing fitness values (`NaN` in `f`)
            using a regression model based on the configurations `X`. Requires
            scikit-learn or a user-provided `impute_model`.
        impute_model : object, optional
            A custom model object with `fit` and `predict` methods used for
            imputation if `impute=True`. If None and `impute=True`, a
            `RandomForestRegressor` from scikit-learn is used by default (if
            available).
        filter: LandscapeFilter, optional
            A filter object to apply to the data before building the landscape.
        n_edit : int, default=1
            The edit distance defining the neighborhood. An edge exists between
            two configurations if their distance (typically Hamming or mixed)
            is less than or equal to `n_edit`. Default is 1, connecting only
            adjacent mutational neighbors.
        verbose : bool, optional
            If provided, overrides the instance's verbosity setting for this
            method call. Controls printed output during the build process.
        Returns
        -------
        self : BaseLandscape
            The instance itself, now populated with the graph and landscape
            properties derived from the input data.

        Raises
        ------
        RuntimeError
            If the `build_from_data` or `build_from_graph` method has already been called
            on this instance. Create a new instance to rebuild.
        ValueError
            If input data `X`, `f`, or `data_types` are invalid (e.g., mismatched
            lengths, empty data, invalid types in `data_types`), or if the
            construction process fails (e.g., imputation fails, resulting
            graph is empty).
        TypeError
            If input types are incorrect (e.g., `data_types` is not a dict).
        ImportError
            If `impute=True` and scikit-learn is not installed, unless an
            `impute_model` is provided.

        Notes
        -----
        The construction process involves several steps:
        1. Select appropriate preprocessing and neighbor generation strategies.
        2. Apply any provided filters to the data.
        3. Preprocess the data based on the landscape type.
        4. Handle missing values and duplicates.
        5. Prepare the data for graph construction (encoding, config_dict).
        6. Construct the core landscape graph.
        7. Analyze basic features (local optima, basins, paths, distances).
        """
        if self._is_built:
            raise RuntimeError(
                "This Landscape instance has already been built. Create a new instance to rebuild."
            )

        # Set parameters
        self.epsilon = epsilon
        self.verbose = verbose

        if verbose:
            print(f"Building Landscape from data...")

        # STEP 1: Select appropriate strategies

        # Select data preprocessing strategies based on landscape type
        self._preprocessor = self._preprocessors.get(self.type)
        if self._preprocessor is None:
            raise ValueError(
                f"No data preprocessor available for landscape type: {self.type}"
            )

        # Select neighborhood generation strategies based on landscape type
        self._neighbor_generator = self._neighbor_generators.get(self.type)
        if self._neighbor_generator is None:
            raise ValueError(
                f"No neighbor generator available for landscape type: {self.type}"
            )

        # STEP 2: Apply filter if provided (before preprocessing)
        if filter is not None:
            if verbose:
                print("Applying data filter...")
            X, f = filter.filter_data(X, f)
            if len(X) == 0:
                raise ValueError("All data removed by filter. Cannot build landscape.")
            if verbose:
                print(f" - {len(X)} configurations remain after filtering.")

        # STEP 3: Preprocess data based on landscape type
        if self.type == "default":
            # Default preprocessor requires data_types
            if not isinstance(data_types, dict):
                raise ValueError(
                    "Data_types must be a dictionary, e.g., {'var_0': 'boolean', 'var_1': 'categorical'}"
                    f"got {type(data_types)}."
                )

            X_processed, f_processed, self.data_types, self.n_vars = (
                self._preprocessor.preprocess(X, f, data_types, verbose=verbose)
            )
        else:
            # Other preprocessors infer data_types
            X_processed, f_processed, self.data_types, self.n_vars = (
                self._preprocessor.preprocess(X, f, verbose=verbose)
            )

        # STEP 4: Data wrangling
        if verbose:
            print(" - Handling missing values and duplicates...")

        # Handling missing values, and if needed, imputing them
        X_clean, f_clean = self._handle_missing_values(
            X_processed, f_processed, self.data_types, impute, impute_model
        )

        if len(X_clean) == 0:
            raise ValueError("All data removed during missing value handling.")

        # Detecting and removing duplicates
        X_final, f_final = self._handle_duplicates(X_clean, f_clean)

        if len(X_final) == 0:
            raise ValueError("All data removed after handling duplicates.")

        # STEP 5: Prepare data (encoding and config_dict)
        processed_data = self._prepare_data(X_final, f_final, self.data_types)
        self.n_configs = len(X_final)

        # STEP 6: Construct landscape graph
        if verbose:
            print("Constructing landscape graph...")

        edges, delta_fits = self._construct_neighborhoods(processed_data, n_edit=n_edit)
        self.graph = self._construct_landscape(processed_data, edges, delta_fits)

        self.n_configs = self.graph.vcount()

        if self.n_configs == 0:
            raise RuntimeError(
                "Landscape graph construction resulted in an empty graph."
            )

        # STEP 7: Analyze graph properties
        self._analyze_landscape(
            calculate_distance=calculate_distance,
            calculate_basins=calculate_basins,
            calculate_paths=calculate_paths,
            calculate_neighbor_fit=calculate_neighbor_fit,
        )

        # Mark as built and return
        self._is_built = True
        if verbose:
            print(f"Landscape built successfully.\n")
            self.describe()

    @classmethod
    def build_from_graph(
        cls,
        filepath: str,
        *,
        verbose: bool = True,
    ) -> "Landscape":
        """Construct a landscape from a saved graph file.

        This class method creates a new landscape instance by loading a previously
        saved graph, avoiding the need to reconstruct the landscape from original
        configuration data. This is significantly faster than building from scratch.

        Parameters
        ----------
        filepath : str
            Path to the saved graph file (.graphml).
        verbose : bool, default=True
            Controls verbosity of output during loading and analysis.
        calculate_basins : bool, default=False
            Whether to (re)calculate basins of attraction after loading.
            If the original graph already has basin information, this can be False.
        calculate_paths : bool, default=False
            Whether to (re)calculate accessible paths after loading.
        calculate_distance : bool, default=False
            Whether to calculate distance metrics to global optimum.

        Returns
        -------
        BaseLandscape
            A new instance populated with the graph and inferred properties.

        Raises
        ------
        ValueError
            If the file cannot be read or doesn't contain valid graph data.
        FileNotFoundError
            If the specified file doesn't exist.

        Notes
        -----
        This method will:
        1. Load the saved graph structure and attributes
        2. Infer essential landscape properties from the graph
        3. (Optionally) Recalculate landscape analysis metrics

        Some specialized attributes from subclasses (like sequence_length in
        SequenceLandscape) will be inferred where possible.
        """
        # Instantiate a new landscape
        instance = cls(verbose=verbose)

        if verbose:
            print(f"Loading landscape from {filepath}...")

        try:
            # Load the graph using GraphML format
            graph = ig.Graph.Read_GraphML(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to load graph from {filepath}: {e}")

        # Assign the graph to the instance
        instance.graph = graph

        # Extract landscape attributes from the loaded graph
        if "maximize" in graph.attributes():
            instance.maximize = graph["maximize"]
        else:
            # Default to maximize=True if not specified
            instance.maximize = True
            if verbose:
                print(
                    "Warning: 'maximize' attribute not found in graph. Defaulting to True."
                )

        if "epsilon" in graph.attributes():
            try:
                # Convert string representation back to original type
                epsilon_str = graph["epsilon"]
                if epsilon_str == "auto":
                    instance.epsilon = "auto"
                else:
                    instance.epsilon = float(epsilon_str)
            except:
                instance.epsilon = "auto"
                if verbose:
                    print(
                        "Warning: Could not parse 'epsilon' attribute. Defaulting to 'auto'."
                    )
        else:
            instance.epsilon = "auto"

        # Extract configs data if available
        if "configs_data" in graph.attributes():
            try:
                config_dict_str = graph["configs_data"]
                # Convert the string representation back to a dictionary
                # (careful with literal_eval for security)
                import ast

                config_dict = ast.literal_eval(config_dict_str)

                # Rebuild the configs Series
                configs_data = {}
                for idx_str, config_str in config_dict.items():
                    try:
                        idx = int(idx_str)
                        # Convert string representation back to tuple
                        config = ast.literal_eval(config_str)
                        configs_data[idx] = config
                    except:
                        if verbose:
                            print(
                                f"Warning: Could not parse config entry: {idx_str} -> {config_str}"
                            )

                instance.configs = pd.Series(configs_data)
            except Exception as e:
                if verbose:
                    print(
                        f"Warning: Could not reconstruct configs from saved data: {e}"
                    )
                instance.configs = None
        else:
            instance.configs = None
            if verbose:
                print(
                    "Warning: No configs data found in graph. Some analyses may be limited."
                )

        # Extract config_dict if available
        if "config_dict_data" in graph.attributes():
            try:
                import ast

                instance.config_dict = ast.literal_eval(graph["config_dict_data"])
            except:
                instance.config_dict = None
                if verbose:
                    print("Warning: Could not parse config_dict from graph attributes.")
        else:
            instance.config_dict = None

        # Extract data_types if available
        if "data_types_data" in graph.attributes():
            try:
                import ast

                instance.data_types = ast.literal_eval(graph["data_types_data"])
            except:
                instance.data_types = None
                if verbose:
                    print("Warning: Could not parse data_types from graph attributes.")
        else:
            instance.data_types = None

        # Set up basic properties
        instance._infer_properties_from_graph()

        # Determine if this is a specialized landscape subclass
        landscape_class = (
            graph["landscape_class"]
            if "landscape_class" in graph.attributes()
            else "BaseLandscape"
        )

        # Handle specific subclass attributes
        if landscape_class == "SequenceLandscape" or landscape_class in [
            "DNALandscape",
            "RNALandscape",
            "ProteinLandscape",
        ]:
            # For sequence landscapes, infer sequence_length from n_vars
            if instance.n_vars is not None:
                instance.sequence_length = instance.n_vars
        elif landscape_class == "BooleanLandscape":
            # For boolean landscapes, infer bit_length from n_vars
            if instance.n_vars is not None:
                instance.bit_length = instance.n_vars

        # Mark as built
        instance._is_built = True

        if verbose:
            print(
                f"Landscape successfully loaded. Graph has {instance.n_configs} nodes and {instance.n_edges} edges."
            )
            instance.describe()

        return instance

    def to_graph(self, filepath: str, include_configs: bool = True) -> None:
        """Save the landscape graph and essential attributes to a file.

        This method serializes the landscape's graph structure and relevant
        attributes to a GraphML file, which can later be loaded using `build_from_graph`.
        This allows efficient storage and sharing of landscapes without requiring
        re-construction from scratch.

        Parameters
        ----------
        filepath : str
            The path where the graph file will be saved. If the file doesn't end
            with '.graphml', this extension will be added automatically.
        include_configs : bool, default=True
            Whether to include the configurations mapping (self.configs) in the saved
            graph. Set to False to reduce file size if this information isn't needed
            or can be reconstructed later.

        Raises
        ------
        RuntimeError
            If the landscape has not been built.
        ValueError
            If the graph cannot be saved to the specified path.

        Notes
        -----
        The GraphML format preserves the graph structure and all vertex/edge attributes.
        In addition to the graph itself, essential landscape attributes like `maximize`
        and `epsilon` are stored as graph attributes.
        """
        self._check_built()

        if self.graph is None:
            raise ValueError("Cannot save an empty graph.")

        # Ensure the filepath has the correct extension
        if not filepath.endswith(".graphml"):
            filepath = f"{filepath}.graphml"

        # Create a copy of the graph to modify
        graph_copy = self.graph.copy()

        # Save essential landscape attributes as graph attributes
        graph_copy["maximize"] = self.maximize
        graph_copy["epsilon"] = str(self.epsilon)  # Convert to string for compatibility
        graph_copy["landscape_class"] = self.__class__.__name__

        # Include configs information if requested
        if include_configs and self.configs is not None:
            # Convert configs Series to a format that can be stored as a graph attribute
            config_dict = {}
            for idx, config in self.configs.items():
                config_dict[str(idx)] = str(
                    config
                )  # Convert to strings for compatibility

            graph_copy["configs_data"] = str(config_dict)

        # Save config_dict if available
        if self.config_dict is not None:
            graph_copy["config_dict_data"] = str(self.config_dict)

        # Save data_types if available
        if self.data_types is not None:
            graph_copy["data_types_data"] = str(self.data_types)

        try:
            # Save the graph using GraphML format
            graph_copy.write_graphml(filepath)
            if self.verbose:
                print(f"Landscape graph saved to {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to save graph to {filepath}: {e}")

    def get_data(self, lo_only: bool = False) -> pd.DataFrame:
        """Extracts landscape data as a pandas DataFrame.

        Returns a DataFrame where rows correspond to configurations (nodes)
        and columns correspond to their attributes (e.g., fitness, degree,
        basin information, original features).

        Parameters
        ----------
        lo_only : bool, default=False
            If True, returns data only for the configurations identified as
            local optima. If the Local Optima Network (LON) has been computed
            (see `get_lon`), data from the LON graph is returned. Otherwise,
            it returns data from the main graph filtered for local optima nodes.
            If False, returns data for all configurations in the main graph.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the attributes of the landscape nodes.
            Index corresponds to the node indices.

        Raises
        ------
        RuntimeError
            If the landscape has not been built (via `build_from_data` or
            `build_from_graph`) before calling this method.
            If `lo_only=True` and the LON graph (`self.lon`) is unexpectedly
            None despite `self.has_lon` being True.
        """
        self._check_built()
        if self.graph is None:
            # This check is primarily for internal consistency.
            raise RuntimeError("Graph is None despite landscape being built.")

        if lo_only:
            if self.lo_index is None or not self.lo_index:
                warnings.warn(
                    "Local optima not found. Cannot filter for LO.",
                    RuntimeWarning,
                )
                return pd.DataFrame()  # Return empty DataFrame

            # First get base local optima data
            lo_subgraph = self.graph.subgraph(self.lo_index)

            # Create base DataFrame with LO attributes
            vertex_attrs = lo_subgraph.vs.attributes()
            data_dicts = []

            for i in range(lo_subgraph.vcount()):
                # Extract all attributes for this vertex
                vertex_dict = {attr: lo_subgraph.vs[i][attr] for attr in vertex_attrs}
                data_dicts.append(vertex_dict)

            # Create DataFrame with LO indices
            data_lo = pd.DataFrame(data_dicts, index=self.lo_index)

            # Columns typically irrelevant for LO-only view from main graph
            cols_to_drop = ["is_lo", "out_degree", "in_degree", "basin_index"]
            data_lo = data_lo.drop(columns=cols_to_drop, errors="ignore")

            # If LON is available, append LON-specific attributes with prefixes
            if self.has_lon:
                if self.lon is None:
                    raise RuntimeError("LON graph is None despite has_lon=True.")

                # Extract LON-specific attributes
                lon_attrs = self.lon.vs.attributes()
                lon_specific_attrs = [
                    "escape_difficulty",
                    "improve_rate",
                    "accessibility",
                    "in_degree",
                    "out_degree",
                ]

                # Create a DataFrame for LON attributes
                lon_data_dicts = []

                for i in range(self.lon.vcount()):
                    lon_vertex_dict = {}

                    # Get the original node index (stored as "name" attribute)
                    original_idx = self.lon.vs[i]["name"]

                    # Extract LON-specific attributes
                    for attr in lon_attrs:
                        if attr in lon_specific_attrs:
                            # Add prefix to LON-specific attributes
                            lon_vertex_dict[f"lon_{attr}"] = self.lon.vs[i][attr]

                    lon_data_dicts.append((original_idx, lon_vertex_dict))

                # Create LON attributes DataFrame with original node indices
                lon_df = pd.DataFrame(
                    [d for _, d in lon_data_dicts],
                    index=[idx for idx, _ in lon_data_dicts],
                )

                # Merge LON attributes with base LO data
                data_lo = data_lo.join(lon_df, how="left")

            # Sort by index for consistency
            data_lo.sort_index(inplace=True)

            data_lo.rename(
                columns={
                    old: new
                    for old, new in zip(
                        data_lo.columns[: self.n_vars], self.data_types.keys()
                    )
                },
                inplace=True,
            )

            return data_lo

        else:
            # Return data for all nodes in the main graph
            vertex_attrs = self.graph.vs.attributes()
            data_dicts = []

            for i in range(self.graph.vcount()):
                # Extract all attributes for this vertex
                vertex_dict = {attr: self.graph.vs[i][attr] for attr in vertex_attrs}
                data_dicts.append(vertex_dict)

            # Create DataFrame with vertex indices
            data = pd.DataFrame(data_dicts, index=range(self.graph.vcount()))

            # Sort by index
            data.sort_index(inplace=True)

            # Drop intermediate calculation columns if they exist
            # These were used to compute basin properties but are less relevant for final output
            if self._path_calculated:
                cols_to_drop = [
                    "size_basin_greedy",
                    "radius_basin_greedy",
                    "size_basin_accessible",
                ]
            else:
                cols_to_drop = ["size_basin_greedy", "radius_basin_greedy"]

            data.rename(
                columns={
                    old: new
                    for old, new in zip(
                        data.columns[: self.n_vars], self.data_types.keys()
                    )
                },
                inplace=True,
            )

            return data.drop(columns=cols_to_drop, errors="ignore")

    def get_lon(self, *args, **kwargs) -> ig.Graph:
        """Constructs and returns the Local Optima Network (LON).

        The LON is a coarse-grained representation of the fitness landscape
        where nodes are the local optima of the original landscape, and edges
        represent the possibility of transitions between their basins of
        attraction, typically weighted by the fitness difference or distance
        between the optima. This method requires the landscape graph to be
        built and local optima to be identified.

        Parameters
        ----------
        *args :
            Positional arguments passed to the underlying `lon.get_lon` function.
        **kwargs :
            Keyword arguments passed to the underlying `lon.get_lon` function.
            Common arguments might include distance metrics or transition probability
            thresholds. Refer to the `lon.get_lon` documentation for details.

        Returns
        -------
        ig.Graph
            The constructed Local Optima Network graph. The graph is also stored
            in the `self.lon` attribute, and `self.has_lon` is set to True.

        Raises
        ------
        RuntimeError
            If the landscape has not been built, or if essential attributes
            (`graph`, `configs`, `lo_index`, `config_dict`) required for LON
            construction are missing.
        """
        self._check_built()
        if self.graph is None:
            raise RuntimeError("Graph is None.")

        # Check for required attributes set by build_from_data or provided to build_from_graph
        if self.configs is None or self.lo_index is None or self.config_dict is None:
            raise RuntimeError(
                "Cannot compute LON: Required attributes missing "
                "(configs, lo_index, config_dict). Ensure landscape was built "
                "from data or these were provided to build_from_graph."
            )

        if not self.lo_index:
            # Handle case with no local optima
            warnings.warn("No local optima found, LON will be empty.", RuntimeWarning)
            self.lon = ig.Graph()  # Create an empty graph
            self.has_lon = True
            return self.lon

        if self.verbose:
            print("Constructing Local Optima Network (LON)...")

        # Call the external get_lon function
        # Assumes lon.get_lon exists and accepts these arguments
        self.lon = get_lon(
            graph=self.graph,
            configs=self.configs,
            lo_index=self.lo_index,
            config_dict=self.config_dict,
            maximize=self.maximize,
            verbose=self.verbose,  # Pass verbosity down
            **kwargs,  # Pass any additional user arguments
        )
        self.has_lon = True  # Set flag

        if self.verbose:
            print(
                f"LON constructed with {self.lon.vcount()} nodes "
                f"and {self.lon.ecount()} edges."
            )
        return self.lon

    def _check_built(self) -> None:
        """Raise an error if the landscape hasn't been built yet."""
        if not self._is_built:
            raise RuntimeError(
                "Landscape has not been built yet. Call build_from_data() or "
                "build_from_graph() first."
            )

    def _generate_neighbors(
        self, config: Tuple, config_dict: Dict, n_edit: int
    ) -> List[Tuple]:
        """Generate neighbors using the selected strategy."""
        return self._neighbor_generator.generate(config, config_dict, n_edit)

    @timeit
    def _construct_neighborhoods(self, data, n_edit):
        """Identifies connections (edges) between neighboring configurations."""
        if self.configs is None or self.config_dict is None:
            raise RuntimeError(
                "Cannot construct neighborhoods: configs/config_dict missing."
            )
        if self.n_configs is None:
            raise RuntimeError("n_configs not set before _construct_neighborhoods")

        # Efficient lookups for index and fitness based on encoded config tuple
        config_to_index = dict(zip(self.configs, data.index))
        config_to_fitness = dict(zip(self.configs, data["fitness"]))

        edges, delta_fits = [], []
        # Use tqdm for progress bar if verbose
        configs_iter = (
            tqdm(
                self.configs, total=self.n_configs, desc="# Constructing neighborhoods"
            )
            if self.verbose
            else self.configs
        )

        # Get the appropriate neighbor generation function (allows override by subclass)
        neighbor_generator = getattr(
            self, "_generate_neighbors", Landscape._generate_neighbors
        )

        for config_tuple in configs_iter:
            current_fit = config_to_fitness[config_tuple]
            current_id = config_to_index[config_tuple]

            # Generate potential neighbors based on n_edit
            neighbors = neighbor_generator(config_tuple, self.config_dict, n_edit)

            for neighbor_tuple in neighbors:
                # Check if the generated neighbor exists in our dataset
                neighbor_idx = config_to_index.get(neighbor_tuple)
                if neighbor_idx is not None:
                    neighbor_fit = config_to_fitness[neighbor_tuple]
                    # Calculate fitness difference (used as edge weight)
                    # Note: delta_fit sign convention depends on maximization goal
                    # delta_fit > 0 means current is fitter than neighbor
                    delta_fit = current_fit - neighbor_fit

                    # Determine if the neighbor represents an improvement
                    # Handles epsilon implicitly for now, strict inequality used.
                    is_improvement = (self.maximize and delta_fit < 0) or (
                        not self.maximize and delta_fit > 0
                    )

                    # Add edge if neighbor is fitter (points towards higher fitness)
                    if is_improvement:
                        # Store edge as (source, target, weight)
                        # Weight is absolute difference |current_fit - neighbor_fit|
                        edges.append((current_id, neighbor_idx))
                        delta_fits.append(abs(delta_fit))

        if self.verbose:
            print(f" - Identified {len(edges)} potential improving connections.")
        return edges, delta_fits

    @timeit
    def _construct_landscape(self, data, edges, delta_fits):
        """Builds the ig.Graph object from nodes and edges."""
        if self.verbose:
            print(" - Constructing graph object...")

        graph = ig.Graph(directed=True)
        graph.add_vertices(len(data.index))  # data.index must be 0..N-1

        graph.add_edges(edges)
        if delta_fits:
            graph.es["delta_fit"] = delta_fits

        if self.verbose:
            print(" - Adding node attributes (fitness, etc.)...")
        # Add original features and fitness as node attributes
        for column in data.columns:
            # Ensure column name is suitable for igraph attribute name
            attr_name = str(column)
            graph.vs[attr_name] = data[column].values

        self.n_edges = graph.ecount()  # Update edge count based on final graph

        # Check if node count changed (e.g., due to disconnected nodes not in edge_list)
        # This check is mainly relevant when building from data.
        original_node_count = len(data)
        final_node_count = graph.vcount()
        if final_node_count != original_node_count:
            # This might happen if the neighborhood definition (n_edit) results
            # in some configurations having no connections within the dataset.
            warnings.warn(
                f"Node count mismatch: {original_node_count} initial configurations "
                f"-> {final_node_count} nodes in the final graph. "
                "This may indicate disconnected components or isolated configurations "
                "within the provided dataset and neighborhood definition.",
                UserWarning,
            )
            # Prune self.configs to match the actual graph nodes if necessary
            # to avoid issues in later analyses (like distance calculations)
            if self.configs is not None:
                nodes_in_graph = set(range(graph.vcount()))
                self.configs = self.configs[
                    self.configs.index.map(lambda idx: idx in nodes_in_graph)
                ]

        return graph

    def _get_default_distance_metric(self):
        """Get the appropriate distance metric based on landscape type."""
        if self.type in ["boolean", "dna", "rna", "protein"]:
            return hamming_distance
        else:
            return mixed_distance

    def _add_network_metrics(self, graph, weight):
        """
        Calculates and adds node/edge metrics to the graph. Currently this includes
        in-degree, out-degree, and PageRank. The weight parameter allows for
        specifying which edge attribute to use for weighted calculations.
        """
        if self.verbose:
            print(" - Calculating network metrics (degrees)...")
        # Assumes utils.add_network_metrics exists and modifies the graph in place
        # or returns the modified graph.
        return add_network_metrics(graph, weight=weight)

    @timeit
    def determine_local_optima(self):
        """Identifies local optima nodes in the landscape graph using igraph."""

        if self.verbose:
            print(" - Determining local optima...")

        # Get out-degrees of all nodes
        out_degrees = self.graph.outdegree()

        # Determine local optima: nodes with out-degree 0
        is_lo = [deg == 0 for deg in out_degrees]

        # Set the 'is_lo' attribute for each vertex
        self.graph.vs["is_lo"] = is_lo

        # Count local optima
        self.n_lo = sum(is_lo)

        # Get indices of local optima and sort them
        self.lo_index = sorted([i for i, val in enumerate(is_lo) if val])

        if self.verbose:
            print(f"   - Found {self.n_lo} local optima.")

    @timeit
    def determine_basin_of_attraction(self):
        """
        Calculates the basin of attraction for each node using a greedy
        hill climbing. The method iterates through each node in the graph
        to perform a hill climbing search, where in each step, it moves
        to the neighbor with the highest fitness (or lowest cost, depending
        on the optimization direction). The process continues until a local
        optimum is reached, and the node is said to be in the basin of that
        local optimum and adds up the number of nodes (i.e., size) in that basin.
        """
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        if self.n_configs is None:
            raise RuntimeError("n_configs is None.")  # Internal check

        if self.verbose:
            print(" - Calculating basins of attraction via hill climbing...")

        n_vertices = self.graph.vcount()

        # Pre-allocate arrays for all results to avoid repeated dict lookups
        basin_indices = np.full(n_vertices, -1, dtype=np.int32)
        step_counts = np.zeros(n_vertices, dtype=np.int32)

        # Process vertices in batches
        batch_size = 10000  # Adjust based on available memory and typical graph size

        # Cache local optima information to avoid redundant hill climbing
        is_lo = (
            np.array(self.graph.vs["is_lo"])
            if "is_lo" in self.graph.vs.attributes()
            else None
        )

        # Prepare iterator with progress bar if verbose
        batch_ranges = list(range(0, n_vertices, batch_size))
        nodes_iter = (
            tqdm(batch_ranges, desc="   - Hill climbing batches")
            if self.verbose
            else batch_ranges
        )

        for batch_start in nodes_iter:
            batch_end = min(batch_start + batch_size, n_vertices)
            batch_indices = range(batch_start, batch_end)

            # Process each node in the batch
            for i in batch_indices:
                # Skip computation if we already know this is a local optimum
                if is_lo is not None and is_lo[i]:
                    # Local optima are in their own basin
                    basin_indices[i] = i
                    step_counts[i] = 0
                    continue

                try:
                    # hill_climb function returns the index of the LO reached and steps taken
                    lo, steps = hill_climb(self.graph, i, "delta_fit")
                    basin_indices[i] = lo
                    step_counts[i] = steps
                except Exception as e:
                    # Handle cases where hill climbing might fail (e.g., complex cycles)
                    if self.verbose:
                        warnings.warn(
                            f"Hill climb failed for node {i}: {e}. Assigning node to its own basin.",
                            RuntimeWarning,
                        )
                    basin_indices[i] = i  # Assign node to its own basin as fallback
                    step_counts[i] = 0

        # Calculate basin sizes and diameters efficiently using numpy
        unique_basins, basin_counts = np.unique(basin_indices, return_counts=True)
        basin_size_map = dict(zip(unique_basins, basin_counts))

        # Create size_basin array mapping each node to its basin size
        size_basin_values = np.zeros(n_vertices, dtype=np.int32)
        for i in range(n_vertices):
            basin_idx = basin_indices[i]
            size_basin_values[i] = basin_size_map.get(basin_idx, 0)

        # Calculate basin radii (max step count per basin)
        max_steps_per_basin = {}
        for i in range(n_vertices):
            basin_idx = basin_indices[i]
            max_steps_per_basin[basin_idx] = max(
                max_steps_per_basin.get(basin_idx, 0), step_counts[i]
            )

        radius_basin_values = np.zeros(n_vertices, dtype=np.int32)
        for i in range(n_vertices):
            basin_idx = basin_indices[i]
            radius_basin_values[i] = max_steps_per_basin.get(basin_idx, 0)

        # Assign results to graph attributes
        self.graph.vs["basin_index"] = basin_indices.tolist()
        self.graph.vs["size_basin_greedy"] = size_basin_values.tolist()
        self.graph.vs["radius_basin_greedy"] = radius_basin_values.tolist()

        # Mark as calculated
        self._basin_calculated = True

        if self.verbose:
            print(f"   - Basins calculated for {len(unique_basins)} local optima.")

    @timeit
    def determine_accessible_paths(self):
        """Determines the size of basins based on accessible paths (ancestors).

        This method calculates the basin size differently from hill climbing.
        It identifies all nodes from which a local optimum can be reached via
        an arbitrary fitness-increasing paths. This uses igraph's subcomponent
        functionality to find all ancestors of each local optima. The result is
        stored as the 'size_basin_accessible' node attribute.

        Note
        ----
        This method is computationally intensive and might be slow for large
        landscapes.
        """
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check

        if self.lo_index is None or not self.lo_index or self.n_lo is None:
            warnings.warn(
                "No local optima found or determined. Cannot calculate accessible paths.",
                RuntimeWarning,
            )
            self._path_calculated = False  # Mark as not calculated
            return

        if self.verbose:
            print(" - Determining accessible paths (ancestors)...")

        dict_size = defaultdict(int)
        # Prepare iterator with progress bar if verbose
        los_iter = (
            tqdm(self.lo_index, total=self.n_lo, desc="   - Finding ancestors")
            if self.verbose
            else self.lo_index
        )

        try:
            # Calculate ancestors for each local optimum
            for lo in los_iter:
                # Find all nodes from which 'lo' is reachable using igraph's subcomponent
                # mode="in" means we're looking for vertices that have paths *to* lo
                ancestors_set = self.graph.subcomponent(lo, mode="in")
                # Basin size includes the ancestors plus the local optimum itself
                # (though the optimum is already included in ancestors_set by subcomponent)
                dict_size[lo] = len(ancestors_set)

            # Add the result as a vertex attribute
            # Create a list where each position corresponds to a vertex index
            size_basin_accessible_values = [0] * self.graph.vcount()
            for vertex_id, basin_size in dict_size.items():
                size_basin_accessible_values[vertex_id] = basin_size

            # Assign the values to the graph as a vertex attribute
            self.graph.vs["size_basin_accessible"] = size_basin_accessible_values

            self._path_calculated = True
            if self.verbose:
                print(
                    f"   - Accessible paths calculated for {len(dict_size)} local optima."
                )
        except Exception as e:
            # Handle potential errors during the calculation
            self._path_calculated = False
            warnings.warn(
                f"Error during accessible path calculation: {e}. "
                "'size_basin_accessible' attribute may be incomplete.",
                RuntimeWarning,
            )

    @timeit
    def determine_neighbor_fitness(self) -> "Landscape":
        """Calculates the mean fitness of neighbors for each node and the difference
        in mean neighbor fitness between connected nodes.

        This method adds two new attributes to the landscape graph:
        1. 'mean_neighbor_fit': A vertex attribute representing the mean fitness of all
        neighboring nodes.
        2. 'delta_mean_neighbor_fit': An edge attribute representing the difference in
        mean neighbor fitness between the connected nodes, calculated from the
        higher fitness node to the lower fitness node.

        This can be useful for identifying evolvability-enhancing (EE) mutations as
        introduced in Wagner (2023).

        References
        ----------
        .. Wagner, A. The role of evolvability in the evolution of
        complex traits. Nat Rev Genet 24, 1-16 (2023).
        https://doi.org/10.1038/s41576-023-00559-0

        Returns
        -------
        self : BaseLandscape
            The landscape instance with the new attributes added.

        Raises
        ------
        RuntimeError
            If the landscape has not been built yet.
        """
        if self.graph is None:
            raise RuntimeError("Graph is None despite landscape being built.")

        if self.verbose:
            print("Calculating neighbor fitness metrics...")

        # Get total number of vertices for progress tracking
        n_vertices = self.graph.vcount()

        # Pre-fetch all fitness values into a numpy array for faster lookup
        fitness_array = np.array(self.graph.vs["fitness"])

        # Step 1: Calculate mean neighbor fitness for each node
        # Preallocate array for mean neighbor fitness values
        mean_neighbor_fit = np.full(n_vertices, np.nan)

        # Process nodes in batches to balance speed and memory usage
        batch_size = 1000  # Adjust based on typical graph size and available memory

        for i in range(0, n_vertices, batch_size):
            # Get the batch of vertices to process
            batch_end = min(i + batch_size, n_vertices)
            batch_indices = list(range(i, batch_end))

            # Process each vertex in the batch
            for vertex_idx in batch_indices:
                # Get neighbors using the built-in neighbors method (memory efficient)
                neighbors = self.graph.neighbors(vertex_idx, mode="all")

                # If the node has neighbors, calculate mean fitness
                if neighbors:
                    # Use vectorized operation on the pre-fetched fitness array
                    mean_neighbor_fit[vertex_idx] = np.mean(fitness_array[neighbors])

        # Add the mean neighbor fitness as a vertex attribute
        self.graph.vs["mean_neighbor_fit"] = mean_neighbor_fit.tolist()

        if self.verbose:
            print(f" - Added 'mean_neighbor_fit' attribute for {n_vertices} nodes")

        # Step 2: Calculate delta mean neighbor fitness for each edge
        # Get total number of edges for progress tracking
        n_edges = self.graph.ecount()

        # Create a more memory-efficient approach for edge processing
        # Process edges in batches to keep memory usage low
        batch_size_edges = 10000  # Adjust based on typical graph size
        delta_mean_neighbor_fit = np.zeros(n_edges)

        for i in range(0, n_edges, batch_size_edges):
            batch_end = min(i + batch_size_edges, n_edges)
            batch_edges = list(range(i, batch_end))

            for edge_idx in batch_edges:
                edge = self.graph.es[edge_idx]
                src, tgt = edge.source, edge.target

                # Calculate difference in mean neighbor fitness
                delta = mean_neighbor_fit[tgt] - mean_neighbor_fit[src]
                delta_mean_neighbor_fit[edge_idx] = delta

        # Add the delta mean neighbor fitness as an edge attribute
        self.graph.es["delta_mean_neighbor_fit"] = delta_mean_neighbor_fit.tolist()

        if self.verbose:
            print(f" - Added 'delta_mean_neighbor_fit' attribute for {n_edges} edges")

        self._neighbor_fit_calculated = True
        return self

    @timeit
    def determine_global_optimum(self):
        """Identifies the global optimum node in the landscape graph using igraph."""
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        if self.verbose:
            print(" - Determining global optimum...")

        if "fitness" not in self.graph.vs.attributes():
            warnings.warn(
                "Cannot determine global optimum: 'fitness' attribute missing from graph nodes.",
                RuntimeWarning,
            )
            self.go_index = None
            self.go = None
            return

        fitness_values = self.graph.vs["fitness"]

        if self.maximize:
            self.go_index = max(
                range(len(fitness_values)), key=lambda i: fitness_values[i]
            )
        else:
            self.go_index = min(
                range(len(fitness_values)), key=lambda i: fitness_values[i]
            )

        try:
            self.go = self.graph.vs[self.go_index].attributes()
            if self.verbose:
                print(
                    f"   - Global optimum found at index {self.go_index} with fitness {self.go['fitness']:.4f}."
                )
        except IndexError:
            warnings.warn(
                f"Global optimum index {self.go_index} not found in graph nodes. Resetting GO.",
                RuntimeWarning,
            )
            self.go_index = None
            self.go = None

    @timeit
    def determine_dist_to_go(self, distance=None):
        """Calculates the distance from each node to the global optimum."""

        # Check prerequisites
        if (
            self.graph is None
            or self.configs is None
            or self.go_index is None
            or self.data_types is None
        ):
            if self.verbose:
                print("Skipping distance calculation - missing prerequisites")
            return

        # Use default distance function if none provided
        if distance is None:
            distance = self._get_default_distance_metric()

        if self.verbose:
            print(
                f" - Calculating distances to global optimum using {distance.__name__}..."
            )

        try:
            configs = np.vstack(self.configs.values)
            go_config = configs[self.go_index]
            distances = distance(configs, go_config, self.data_types)

            # Add calculated distances as a node attribute
            self.graph.vs["dist_go"] = distances
            self._distance_calculated = True

            if self.verbose:
                print(
                    "   - Distances to GO calculated and added as node attribute 'dist_go'."
                )

        except Exception as e:
            warnings.warn(f"Error calculating distances to GO: {e}", RuntimeWarning)
            self._distance_calculated = False

    def describe(self) -> None:
        """Prints a summary description of the landscape properties.

        Provides a quick overview of the landscape's size, complexity,
        and analysis status.
        """
        print("--- Landscape Summary ---")
        print(f"Class: {self.__class__.__name__}")
        print(f"Built: {self._is_built}")
        if self._is_built:
            print(
                f"Variables (n_vars): {self.n_vars if self.n_vars is not None else 'Unknown'}"
            )
            print(
                f"Configurations (n_configs): {self.n_configs if self.n_configs is not None else 'Unknown'}"
            )
            print(
                f"Connections (n_edges): {self.n_edges if self.n_edges is not None else 'Unknown'}"
            )
            print(
                f"Local Optima (n_lo): {self.n_lo if self.n_lo is not None else 'Not Calculated'}"
            )
            go_idx_str = (
                str(self.go_index)
                if self.go_index is not None
                else "Not Calculated/Found"
            )
            print(f"Global Optimum Index: {go_idx_str}")
            print(f"Maximize Fitness: {self.maximize}")
            # Add status of optional calculations
            print(f"Basins Calculated (Hill Climb): {self._basin_calculated}")
            print(f"Accessible Paths Calculated: {self._path_calculated}")
            print(f"LON Calculated: {self.has_lon}")
        else:
            print(" (Landscape not built yet)")
        print("---")

    def _standardize_input_formats(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        f: Union[pd.Series, list, np.ndarray],
        data_types: Dict[str, str],
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
        """Standardizes inputs to DataFrame/Series and aligns indices."""
        if self.verbose:
            print(" - Standardizing input formats...")

        if isinstance(X, np.ndarray):
            try:
                columns = [f"var_{i}" for i in range(X.shape[1])]
                X_standard = pd.DataFrame(X, columns=columns).copy()
            except Exception as e:
                raise TypeError(
                    f"Could not convert input X (ndarray) to DataFrame: {e}"
                )
        elif isinstance(X, pd.DataFrame):
            X_standard = X.copy()
        else:
            raise TypeError(
                f"Input X must be a pandas DataFrame or numpy ndarray, got {type(X)}."
            )

        if isinstance(f, (list, np.ndarray)):
            try:
                f_standard = pd.Series(f, name="fitness").copy()
            except Exception as e:
                raise TypeError(
                    f"Could not convert input f (list/ndarray) to Series: {e}"
                )
        elif isinstance(f, pd.Series):
            f_standard = f.copy()
            f_standard.name = "fitness"
        else:
            raise TypeError(
                f"Input f must be a pandas Series, list, or numpy ndarray, got {type(f)}."
            )

        if len(X_standard) != len(f_standard):
            raise ValueError(
                f"Inconsistent lengths after standardization: X has {len(X_standard)} rows, "
                f"f has {len(f_standard)} elements."
            )

        X_standard.reset_index(drop=True, inplace=True)
        f_standard.reset_index(drop=True, inplace=True)
        f_standard.index = X_standard.index

        dt_standard = data_types.copy() if data_types else {}

        return X_standard, f_standard, dt_standard

    @timeit
    def _handle_missing_values(
        self,
        X_in: pd.DataFrame,
        f_in: pd.Series,
        data_types: Dict[str, str],
        impute: bool,
        impute_model: Optional[Any],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handles missing values (NaN) in X and f, with optional imputation."""
        if self.verbose:
            print(" - Handling missing values...")

        X_out, f_out = X_in.copy(), f_in.copy()
        initial_count = len(X_out)

        if X_out.isnull().values.any():
            nan_rows_X = X_out.isnull().any(axis=1)
            n_removed_X = nan_rows_X.sum()
            if self.verbose:
                print(
                    f"   - Found {n_removed_X} rows with NaN in X (features). Removing them."
                )
            X_out = X_out[~nan_rows_X]
            f_out = f_out[~nan_rows_X]
            if len(X_out) == 0:
                warnings.warn("All rows removed due to NaNs in X.", RuntimeWarning)
                return X_out, f_out

        nan_mask_f = f_out.isnull()
        if nan_mask_f.any():
            n_missing_f = nan_mask_f.sum()
            if not impute:
                if self.verbose:
                    print(
                        f"   - Found {n_missing_f} rows with NaN in f (fitness). Removing them (impute=False)."
                    )
                X_out = X_out[~nan_mask_f]
                f_out = f_out[~nan_mask_f]
            else:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.compose import ColumnTransformer
                    from sklearn.preprocessing import OneHotEncoder
                    from sklearn.pipeline import Pipeline
                except:
                    ImportError(
                        "Fitness value imputation requires scikit-learn. "
                        "Please install it (`pip install scikit-learn`)."
                    )

                if self.verbose:
                    print(
                        f"   - Found {n_missing_f} missing fitness values. Attempting imputation (impute=True)."
                    )

                X_known_f = X_out[~nan_mask_f]
                f_known = f_out[~nan_mask_f]
                X_unknown_f = X_out[nan_mask_f]

                if len(X_known_f) == 0:
                    raise ValueError(
                        "Cannot impute fitness: No data points with known "
                        "fitness values available to train the imputation model."
                    )
                if len(X_unknown_f) == 0:
                    warnings.warn(
                        "Imputation requested, but no missing fitness values "
                        "found after handling NaNs in X.",
                        RuntimeWarning,
                    )
                    return X_out, f_out

                categorical_features = [
                    col
                    for col, dtype in data_types.items()
                    if dtype in ["categorical", "ordinal"]
                ]

                preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            categorical_features,
                        )
                    ],
                    remainder="passthrough",
                )

                if impute_model is None:
                    model = RandomForestRegressor(random_state=42, n_jobs=-1)
                    if self.verbose:
                        print(
                            "     - Using default RandomForestRegressor for imputation."
                        )
                else:
                    if not (
                        hasattr(impute_model, "fit")
                        and hasattr(impute_model, "predict")
                    ):
                        raise TypeError(
                            "Provided impute_model must have 'fit' and 'predict' methods."
                        )
                    model = impute_model
                    if self.verbose:
                        print(
                            f"     - Using user-provided imputation model: {type(model).__name__}"
                        )

                imputation_pipeline = Pipeline(
                    steps=[("preprocess", preprocessor), ("regressor", model)]
                )

                try:
                    if self.verbose:
                        print(
                            f"     - Training imputation model on {len(X_known_f)} data points..."
                        )
                    imputation_pipeline.fit(X_known_f, f_known)
                except Exception as e:
                    raise ValueError(f"Failed to train the imputation model: {e}")

                try:
                    if self.verbose:
                        print(
                            f"     - Predicting {len(X_unknown_f)} missing fitness values..."
                        )
                    f_predicted = imputation_pipeline.predict(X_unknown_f)
                except Exception as e:
                    raise ValueError(
                        f"Failed to predict using the imputation model: {e}"
                    )

                f_out.loc[nan_mask_f] = f_predicted
                if self.verbose:
                    print("     - Imputation complete.")

        final_count = len(X_out)
        if self.verbose and initial_count != final_count:
            print(
                f"   - Missing value handling complete. Kept {final_count}/{initial_count} configurations."
            )

        f_out = f_out.loc[X_out.index]

        return X_out, f_out

    @timeit
    def _handle_duplicates(
        self, X_in: pd.DataFrame, f_in: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Removes duplicate configurations in X, keeping the first occurrence."""
        if self.verbose:
            print(" - Handling duplicate configurations...")

        initial_count = len(X_in)
        mask_duplicates = X_in.duplicated(keep="first")
        num_removed = mask_duplicates.sum()

        if num_removed > 0:
            if self.verbose:
                print(
                    f"   - Found {num_removed} duplicate configurations in X. Keeping first occurrence."
                )
            X_out = X_in[~mask_duplicates].copy()
            f_out = f_in[~mask_duplicates].copy()
        else:
            if self.verbose:
                print("   - No duplicate configurations found.")
            X_out, f_out = X_in, f_in

        final_count = len(X_out)
        if self.verbose and initial_count != final_count:
            print(
                f"   - Duplicate handling complete. Kept {final_count}/{initial_count} configurations."
            )

        return X_out, f_out

    @timeit
    def _prepare_data(self, X, f, data_types):
        """Encodes data and sets `configs` and `config_dict` attributes."""
        if self.verbose:
            print("Preparing data for landscape construction (encoding variables)...")
        X_encoded = X.copy()
        # Encode based on data type
        for col, dtype in data_types.items():
            if dtype == "boolean":
                # Convert boolean (True/False or 1/0) to integer 0 or 1
                X_encoded[col] = X_encoded[col].astype(bool).astype(int)
            elif dtype == "categorical":
                # Factorize categorical data into integer codes (0, 1, 2, ...)
                X_encoded[col] = pd.factorize(X_encoded[col])[0]
            elif dtype == "ordinal":
                # Convert ordinal data into integer codes based on inherent order
                X_encoded[col] = pd.Categorical(X_encoded[col], ordered=True).codes
            else:
                # This case should ideally not be reached if validation is done
                raise ValueError(
                    f"Unsupported data type '{dtype}' encountered during encoding."
                )

        # Store configurations as tuples in a Series, indexed like X
        self.configs = pd.Series(X_encoded.apply(tuple, axis=1), index=X_encoded.index)

        # Check for duplicates in encoded configurations (should be handled earlier, but as safeguard)
        if self.configs.duplicated().any():
            warnings.warn(
                "Duplicate encoded configurations found after validation. "
                "This might indicate issues in the duplicate handling step.",
                RuntimeWarning,
            )

        # Generate the config_dict describing the encoding
        self.config_dict = self._generate_config_dict(data_types, X_encoded)

        # Return original data plus fitness for setting node attributes later
        data_for_attributes = pd.concat([X, f], axis=1)
        data_for_attributes.reset_index(
            drop=True, inplace=True
        )  # Ensure index alignment is maintained

        return data_for_attributes

    def _generate_config_dict(self, data_types, data_encoded):
        """Generates config_dict based on encoded data."""
        config_dict = {}
        max_encoded = data_encoded.max()
        for i, (col, dtype) in enumerate(data_types.items()):
            # Max value is 1 for boolean, otherwise use the max encoded value found
            max_val = 1 if dtype == "boolean" else max_encoded[col]
            config_dict[i] = {"type": dtype, "max": int(max_val)}
        return config_dict

    @timeit
    def _analyze_landscape(
        self,
        calculate_distance: bool,
        calculate_basins: bool,
        calculate_paths: bool,
        calculate_neighbor_fit: bool,
    ) -> None:
        """Internal helper to run analysis steps on the landscape graph.

        This method orchestrates the calculation of key landscape properties
        after the graph has been constructed or provided.

        Parameters
        ----------
        calculate_distance : bool
            Whether to calculate distance-based metrics, specifically the
            distance of each node to the global optimum. Requires `configs`
            and `data_types` attributes to be set.
        calculate_basins : bool
            Whether to calculate basins of attraction.
        calculate_paths : bool
            Whether to calculate accessible paths (ancestors).
        calculate_neighbor_fit : bool
            Whether to calculate mean neighbor fitness.
        """
        if self.graph is None:
            # This check is primarily for internal consistency.
            raise RuntimeError("Graph is None, cannot analyze.")

        if self.graph.vcount() == 0:
            warnings.warn("Cannot analyze an empty graph.", RuntimeWarning)
            # Reset analysis attributes for consistency
            self.n_lo = 0
            self.lo_index = []
            self.go_index = None
            self.go = None
            self.lon = None
            self.has_lon = False
            self._basin_calculated = calculate_basins  # Mark as attempted/done
            self._path_calculated = calculate_paths  # Mark as attempted/done
            return

        if self.verbose:
            print("Calculating landscape properties...")

        # Add basic network metrics if they don't already exist
        if "out_degree" not in self.graph.vs.attributes():
            if self.verbose:
                print(" - Adding network metrics (degrees)...")
            # Assumes 'delta_fit' exists if built from data, might need adjustment
            # if graph provided externally lacks this weight.
            weight_key = (
                "delta_fit" if "delta_fit" in self.graph.es.attributes() else None
            )
            self.graph = self._add_network_metrics(self.graph, weight=weight_key)

        # Determine Optima
        self.determine_local_optima()
        self.determine_global_optimum()  # Needs LO info if maximize=False? Check logic. GO depends only on fitness attr.

        # Determine Basins (requires optima info)
        if calculate_basins:
            self.determine_basin_of_attraction()
        else:
            self._basin_calculated = False
            if self.verbose:
                print(" - Skipping basin calculation.")

        # Determine Accessible Paths (requires optima info)
        if calculate_paths:
            # Added size check to prevent excessive computation time
            if self.n_configs is None or self.n_configs < 200000:
                self.determine_accessible_paths()
            elif self.verbose:
                warnings.warn(
                    f"Landscape size ({self.n_configs} nodes) exceeds threshold "
                    "(200,000). Skipping accessible paths calculation.",
                    RuntimeWarning,
                )
                self._path_calculated = False
        else:
            self._path_calculated = False
            if self.verbose:
                print(" - Skipping accessible paths calculation.")

        # Distance Calculation (requires configs, data_types, and GO)
        if calculate_distance:
            if (
                self.configs is not None
                and self.go_index is not None
                and self.data_types is not None
            ):
                # Allow subclasses to specify a different default distance metric
                distance_func = getattr(
                    self, "_get_default_distance_metric", lambda: mixed_distance
                )()
                self.determine_dist_to_go(distance=distance_func)
            elif self.verbose:
                print(
                    "   - Skipping distance to Global Optimum calculation "
                    "(requires configuration data, data_types, and successful "
                    "GO identification)."
                )
        if calculate_neighbor_fit:
            self.determine_neighbor_fitness()

        elif self.verbose:
            print(" - Skipping distance to Global Optimum calculation (not requested).")

    def _infer_properties_from_graph(self):
        """Set basic landscape properties based on the assigned self.graph."""
        if self.graph is None:
            # This check is primarily for internal consistency, should not be
            # reachable if called correctly by build_from_graph.
            raise RuntimeError(
                "_infer_properties_from_graph called before graph assignment."
            )

        self.n_configs = self.graph.vcount()
        self.n_edges = self.graph.ecount()

        # Attempt to infer the number of variables (dimensionality)
        if self.data_types:
            self.n_vars = len(self.data_types)
        elif self.configs is not None and len(self.configs) > 0:
            try:
                # Assumes configs series contains tuples/lists of variables
                self.n_vars = len(self.configs.iloc[0])
            except Exception:
                self.n_vars = None  # Failed inference
        else:
            # Fallback: try to guess from node attributes (less reliable)
            try:
                if self.graph.vcount() > 0:
                    vertex_attrs = self.graph.vs.attributes()
                    # Heuristic: look for attributes like 'var_0', 'pos_1', etc.
                    potential_var_keys = [
                        k
                        for k in vertex_attrs
                        if isinstance(k, str)
                        and (k.startswith(("var_", "pos_", "bit_")))
                    ]
                    if potential_var_keys:
                        self.n_vars = len(potential_var_keys)
                    else:
                        self.n_vars = None  # No obvious variable attributes
                else:
                    self.n_vars = None  # No vertices to examine
            except Exception:
                self.n_vars = None  # Failed inference

        if self.n_vars is None and self.verbose:
            warnings.warn(
                "Could not reliably determine 'n_vars' (number of variables) "
                "from the provided graph and parameters. Distance calculations "
                "or analyses requiring dimensionality might fail.",
                UserWarning,
            )


class BooleanLandscape(Landscape):
    """A specialized landscape class for boolean configuration spaces.

    This class represents fitness landscapes where each configuration is a bit string
    (sequence of 0s and 1s), which is common in many optimization problems like
    NK landscapes, MAXSAT, and binary encoding of combinatorial problems.

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> X_data = pd.DataFrame({'var_0': [0, 0, 1, 1], 'var_1': [0, 1, 0, 1]})
    >>> f_data = pd.Series([1.0, 2.0, 3.0, 2.5])
    >>> landscape = BooleanLandscape().build_from_data(X_data, f_data)
    """

    def __init__(self, maximize: bool = True):
        """Initialize a boolean landscape.

        Parameters
        ----------
        maximize : bool, default=True
            Determines the optimization direction.
        """
        super().__init__(type="boolean", maximize=maximize)


class SequenceLandscape(Landscape):
    """A specialized landscape class for sequence-based configuration spaces.

    This class represents fitness landscapes where each configuration is a sequence
    of characters from a defined alphabet. Examples include DNA/RNA sequences,
    protein sequences, or any other discrete sequence space.

    Parameters
    ----------
    alphabet : list[str]
        The set of valid characters that can appear in the sequences.
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.
    """

    def __init__(
        self,
        alphabet: List[str],
        maximize: bool = True,
    ):
        # Register our processor with the custom alphabet
        from ._processor import SequencePreprocessor
        from ._neighbors import SequenceNeighborGenerator

        preprocessor = SequencePreprocessor(alphabet)
        neighbor_generator = SequenceNeighborGenerator(len(alphabet))

        # Use custom type to register these custom processors
        type_key = f"sequence_{id(alphabet)}"
        Landscape.register_preprocessor(type_key, preprocessor)
        Landscape.register_neighbor_generator(type_key, neighbor_generator)

        super().__init__(type=type_key, maximize=maximize)
        self.alphabet = alphabet


class DNALandscape(SequenceLandscape):
    """A specialized landscape class for DNA sequence configuration spaces.

    This class represents fitness landscapes where each configuration is a DNA sequence
    using the standard DNA alphabet (A, C, G, T).

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.
    """

    def __init__(self, maximize: bool = True):
        super().__init__(alphabet=DNA_ALPHABET, maximize=maximize)


class RNALandscape(SequenceLandscape):
    """A specialized landscape class for RNA sequence configuration spaces.

    This class represents fitness landscapes where each configuration is an RNA sequence
    using the standard RNA alphabet (A, C, G, U).

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.
    """

    def __init__(self, maximize: bool = True):
        super().__init__(alphabet=RNA_ALPHABET, maximize=maximize)


class ProteinLandscape(SequenceLandscape):
    """A specialized landscape class for protein sequence configuration spaces.

    This class represents fitness landscapes where each configuration is a protein sequence
    using the standard protein alphabet (20 amino acids).

    Parameters
    ----------
    maximize : bool, default=True
        Determines the optimization direction. If True, the landscape seeks
        higher fitness values. If False, it seeks lower values.
    """

    def __init__(self, maximize: bool = True):
        super().__init__(
            alphabet=PROTEIN_ALPHABET,
            maximize=maximize,
        )
