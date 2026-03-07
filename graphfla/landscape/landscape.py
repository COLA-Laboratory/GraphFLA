import pandas as pd
import numpy as np
import igraph as ig
import warnings
import time

from typing import Tuple, Dict, List, Union, Optional, Any
from collections import defaultdict
from tqdm import tqdm

from ..lon import get_lon
from ..algorithms import hill_climb
from ..utils import add_network_metrics, apply_pre_construction_filter, apply_post_construction_filter
from ..distances import mixed_distance, hamming_distance

from .._processor import (
    DataPreprocessor,
    BooleanPreprocessor,
    DefaultPreprocessor,
    SequencePreprocessor,
)

from .._neighbors import (
    NeighborGenerator,
    BooleanNeighborGenerator,
    DefaultNeighborGenerator,
    SequenceNeighborGenerator,
    select_neighborhood_strategy,
    construct_neighborhoods_active,
    construct_neighborhoods_pairwise,
    construct_neighborhoods_broadcast,
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
    epsilon : float
        Neutrality threshold. Neighbors with ``|fitness_a - fitness_b| <= epsilon``
        are classified as neutral rather than improving/worsening. When
        ``epsilon > 0``, a plateau layer is constructed and downstream analyses
        become plateau-aware. Defaults to 0.
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
        self.epsilon = 0
        self.verbose = False

        # Plateau / neutrality layer attributes
        self._has_plateaus = False
        self._node_to_plateau = None   # np.ndarray int32: node_idx → plateau_id
        self._plateaus = None           # dict: plateau_id → list[int] of member nodes
        self._neutral_neighbors = None  # dict: node_idx → list[int] of neutral neighbors
        self.n_plateau_lo = None
        self.plateau_lo_index = None    # list of plateau_ids that are LOs

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
            if isinstance(index, int) and 0 <= index < self.graph.vcount():
                vertex = self.graph.vs[index]
            else:
                vertex = self.graph.vs.find(name=index)
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
                return 0 <= item < self.graph.vcount()
            else:
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
            return self._is_built == other._is_built
        if self.graph is None or other.graph is None:
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
        epsilon: float = 0,
        calculate_basins: bool = False,
        calculate_paths: bool = False,
        calculate_distance: bool = False,
        calculate_neighbor_fit: bool = False,
        tau: Optional[float] = None,
        filter_mode: str = "any",
        impute: bool = False,
        impute_model: Optional[Any] = None,
        n_edit: int = 1,
        neighborhood_strategy: str = "auto",
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
        X : pandas.DataFrame or numpy.ndarray or a list of strings 
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
        epsilon : float, default=0
            Neutrality threshold. Neighboring configurations whose absolute
            fitness difference is ``<= epsilon`` are treated as neutral
            (equal-fitness) rather than strictly improving/worsening. When
            ``epsilon > 0``, a plateau layer is constructed: neutral neighbors
            are grouped into plateaus via connected components, and downstream
            analyses (local optima, basins, accessible paths) become
            plateau-aware. A higher epsilon produces a smoother landscape with
            fewer, larger local optima; ``epsilon=0`` preserves strict-inequality
            behavior (the default).
        calculate_basins : bool, default=True
            If True, calculates the basins of attraction for each local optimum
            using a greedy hill-climbing algorithm. This identifies which configurations
            lead to which peak. Populates the `size_basin_greedy` and `radius_basin_greedy`
            attributes for each local optimum.
        calculate_paths : bool, default=False
            If True, calculates accessible paths (ancestors) for local optima.
            This can be computationally intensive for large landscapes. Populates the
            `size_basin_accessible` attribute for each local optimum.
        tau : float, optional
            Functional threshold. Configurations whose fitness is above
            (when ``maximize=True``) or below (when ``maximize=False``) this
            value are considered "functional". Used together with
            ``filter_mode`` to focus the landscape on biologically or
            practically relevant regions.
        filter_mode : str, default='any'
            How to apply the functional threshold ``tau``. Options:

            - ``'any'``: Pre-construction filter. Remove non-functional
              configurations before building the graph. If maximize: keep
              fitness >= tau; if minimize: keep fitness <= tau.
            - ``'both'``: Post-construction filter. Keep all configurations,
              but remove edges where both endpoints are non-functional
              (both < tau when maximize, both > tau when minimize). Preserves
              transitions between functional and non-functional regions.
        impute : bool, default=False
            If True, attempts to fill in missing fitness values (`NaN` in `f`)
            using a regression model based on the configurations `X`. Requires
            scikit-learn or a user-provided `impute_model`.
        impute_model : object, optional
            A custom model object with `fit` and `predict` methods used for
            imputation if `impute=True`. If None and `impute=True`, a
            `RandomForestRegressor` from scikit-learn is used by default (if
            available).
        n_edit : int, default=1
            The edit distance defining the neighborhood. An edge exists between
            two configurations if their distance (typically Hamming or mixed)
            is less than or equal to `n_edit`. Default is 1, connecting only
            adjacent mutational neighbors.
        neighborhood_strategy : str, default='auto'
            Strategy for identifying neighboring configurations. Options:

            - ``'auto'``: Automatically selects the fastest strategy based on
              dataset size, sequence length, and alphabet size. Uses
              ``'pairwise'`` when the full distance matrix fits in memory
              (~4 GiB), falls back to ``'broadcast'`` when vectorized
              per-config distances are cheaper than candidate generation, and
              defaults to ``'active'`` otherwise.
            - ``'active'``: For each configuration, enumerates all possible
              single-edit (or ``n_edit``-edit) mutant neighbors and checks
              whether they exist in the dataset via hash lookup. Efficient for
              dense datasets where most proposed neighbors are present.
            - ``'pairwise'``: Computes the full pairwise Hamming distance
              matrix using ``scipy.spatial.distance.pdist``. Very fast for
              small-to-moderate datasets (up to ~25 000 configurations) thanks
              to highly optimized C code. Memory usage scales as
              ``O(n_configs^2)``.
            - ``'broadcast'``: For each configuration, computes its Hamming
              distance to all other configurations using vectorized NumPy
              operations. Suitable for large datasets with long sequences and
              sparse sampling, where the ``'active'`` strategy would waste
              time generating candidates that don't exist and ``'pairwise'``
              would exceed memory limits.
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
        2. Preprocess the data based on the landscape type.
        3. Handle missing values and duplicates.
        4. Prepare the data for graph construction (encoding, config_dict).
        5. Construct the core landscape graph.
        6. Analyze basic features (local optima, basins, paths, distances).
        """
        if self._is_built:
            raise RuntimeError(
                "This Landscape instance has already been built. Create a new instance to rebuild."
            )

        self.epsilon = float(epsilon)
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

        # STEP 2: Apply pre-construction function-based fitness filter
        X, f = apply_pre_construction_filter(
            X, f, self.maximize, tau, filter_mode, verbose
        )

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

        edges, delta_fits, neutral_pairs = self._construct_neighborhoods(
            processed_data, n_edit=n_edit, strategy=neighborhood_strategy
        )
        self.graph = self._construct_landscape(processed_data, edges, delta_fits)

        # STEP 7: Post-construction pruning
        self.graph, self.n_configs, self.n_edges, kept_indices = (
            apply_post_construction_filter(
                self.graph, self.maximize, tau, filter_mode, verbose
            )
        )

        # STEP 7b: Remap configs/neutral-pairs if the filter removed vertices
        if kept_indices is not None:
            old_to_new = {old: new for new, old in enumerate(kept_indices)}

            if self.configs is not None:
                self.configs = pd.Series(
                    [self.configs.iloc[old] for old in kept_indices],
                    index=range(len(kept_indices)),
                )

            if neutral_pairs:
                neutral_pairs = [
                    (old_to_new[u], old_to_new[v])
                    for u, v in neutral_pairs
                    if u in old_to_new and v in old_to_new
                ]

        self._identify_plateaus(neutral_pairs)

        # STEP 8: Analyze graph properties
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

        return None

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

        Returns
        -------
        Landscape
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
        3. Recalculate local optima and global optimum from the graph structure

        Previously computed attributes (basins, accessible paths, distances,
        neighbor fitness) are preserved from the saved graph if present.

        Some specialized attributes from subclasses (like sequence_length in
        SequenceLandscape) will be inferred where possible.
        """
        instance = cls()
        instance.verbose = verbose

        if verbose:
            print(f"Loading landscape from {filepath}...")

        try:
            graph = ig.Graph.Read_GraphML(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to load graph from {filepath}: {e}")

        instance.graph = graph

        if "maximize" in graph.attributes():
            instance.maximize = bool(graph["maximize"])
        else:
            instance.maximize = True
            if verbose:
                print(
                    "Warning: 'maximize' attribute not found in graph. Defaulting to True."
                )

        if "epsilon" in graph.attributes():
            try:
                epsilon_str = graph["epsilon"]
                if epsilon_str == "auto":
                    instance.epsilon = 0.0
                else:
                    instance.epsilon = float(epsilon_str)
            except Exception:
                instance.epsilon = 0.0
                if verbose:
                    print(
                        "Warning: Could not parse 'epsilon' attribute. Defaulting to 0."
                    )
        else:
            instance.epsilon = 0.0

        if "landscape_type" in graph.attributes():
            instance.type = graph["landscape_type"]

        # Extract configs data if available
        import ast

        if "configs_data" in graph.attributes():
            try:
                config_dict_str = graph["configs_data"]
                config_dict = ast.literal_eval(config_dict_str)

                configs_data = {}
                for idx_str, config_str in config_dict.items():
                    try:
                        idx = int(idx_str)
                        config = ast.literal_eval(config_str)
                        configs_data[idx] = config
                    except Exception:
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

        if "config_dict_data" in graph.attributes():
            try:
                instance.config_dict = ast.literal_eval(graph["config_dict_data"])
            except Exception:
                instance.config_dict = None
                if verbose:
                    print("Warning: Could not parse config_dict from graph attributes.")
        else:
            instance.config_dict = None

        if "data_types_data" in graph.attributes():
            try:
                instance.data_types = ast.literal_eval(graph["data_types_data"])
            except Exception:
                instance.data_types = None
                if verbose:
                    print("Warning: Could not parse data_types from graph attributes.")
        else:
            instance.data_types = None

        # Infer basic properties (n_configs, n_edges, n_vars)
        instance._infer_properties_from_graph()

        # Determine if this is a specialized landscape subclass
        landscape_class = (
            graph["landscape_class"]
            if "landscape_class" in graph.attributes()
            else "BaseLandscape"
        )

        if landscape_class == "SequenceLandscape" or landscape_class in [
            "DNALandscape",
            "RNALandscape",
            "ProteinLandscape",
        ]:
            if instance.n_vars is not None:
                instance.sequence_length = instance.n_vars
        elif landscape_class == "BooleanLandscape":
            if instance.n_vars is not None:
                instance.bit_length = instance.n_vars

        # Reconstruct plateau data structures if saved in the graph
        if "plateau_id" in graph.vs.attributes():
            instance._reconstruct_plateaus()

        # Determine local optima and global optimum from graph structure
        if instance.graph.vcount() > 0:
            instance.determine_local_optima()
            instance.determine_global_optimum()

        # Infer calculation status flags from saved graph attributes
        if "basin_index" in graph.vs.attributes():
            instance._basin_calculated = True
        if "size_basin_accessible" in graph.vs.attributes():
            instance._path_calculated = True
        if "dist_go" in graph.vs.attributes():
            instance._distance_calculated = True
        if "mean_neighbor_fit" in graph.vs.attributes():
            instance._neighbor_fit_calculated = True

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
        graph_copy["landscape_type"] = self.type

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

            if self.data_types is not None and self.n_vars is not None:
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

            if self.data_types is not None and self.n_vars is not None:
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

    def _resolve_epsilon(self) -> float:
        """Return the numeric epsilon value."""
        return float(self.epsilon)

    def _generate_neighbors(
        self, config: Tuple, config_dict: Dict, n_edit: int
    ) -> List[Tuple]:
        """Generate neighbors using the selected strategy."""
        return self._neighbor_generator.generate(config, config_dict, n_edit)

    @timeit
    def _construct_neighborhoods(self, data, n_edit, strategy="auto"):
        """Identifies connections (edges) between neighboring configurations.

        This is the main dispatcher that delegates to one of three concrete
        strategies depending on the ``strategy`` argument.

        Parameters
        ----------
        data : pandas.DataFrame
            Configuration data with a ``'fitness'`` column and integer index
            ``0 .. n_configs-1``.
        n_edit : int
            Maximum edit distance for two configurations to be neighbors.
        strategy : str, default='auto'
            One of ``'auto'``, ``'active'``, ``'pairwise'``, or
            ``'broadcast'``.

        Returns
        -------
        edges : list[tuple[int, int]]
            Directed edges from lower-fitness to higher-fitness neighbors.
        delta_fits : list[float]
            Absolute fitness differences for each edge.
        neutral_pairs : list[tuple[int, int]]
            Undirected pairs of neighbors within epsilon fitness of each other.
            Each pair (u, v) is stored once (u < v) to avoid duplicates.
        """
        if self.configs is None or self.config_dict is None:
            raise RuntimeError(
                "Cannot construct neighborhoods: configs/config_dict missing."
            )
        if self.n_configs is None:
            raise RuntimeError("n_configs not set before _construct_neighborhoods")

        valid_strategies = {"auto", "active", "pairwise", "broadcast"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown neighborhood strategy '{strategy}'. "
                f"Choose from {sorted(valid_strategies)}."
            )

        if strategy == "auto":
            strategy = self._select_neighborhood_strategy(n_edit)
            if self.verbose:
                print(f" - Auto-selected '{strategy}' neighborhood strategy.")

        if strategy == "pairwise":
            return self._construct_neighborhoods_pairwise(data, n_edit)
        elif strategy == "broadcast":
            return self._construct_neighborhoods_broadcast(data, n_edit)
        else:
            return self._construct_neighborhoods_active(data, n_edit)

    def _select_neighborhood_strategy(self, n_edit: int) -> str:
        """Choose the fastest neighborhood strategy for the current dataset."""
        return select_neighborhood_strategy(
            self.n_configs, self.n_vars, self.config_dict, n_edit
        )

    def _construct_neighborhoods_active(self, data, n_edit):
        """Enumerate candidate mutant neighbors and check dataset membership."""
        epsilon = self._resolve_epsilon()
        neighbor_generator = getattr(
            self, "_generate_neighbors", Landscape._generate_neighbors
        )
        return construct_neighborhoods_active(
            self.configs,
            self.config_dict,
            data,
            n_edit,
            epsilon,
            self.maximize,
            self.verbose,
            neighbor_generator,
        )

    def _construct_neighborhoods_pairwise(self, data, n_edit):
        """Compute the full pairwise Hamming distance matrix with pdist."""
        epsilon = self._resolve_epsilon()
        return construct_neighborhoods_pairwise(
            data, n_edit, self.configs, epsilon, self.verbose, self.maximize
        )

    def _construct_neighborhoods_broadcast(self, data, n_edit):
        """For each config, compute Hamming distance to all others via NumPy."""
        epsilon = self._resolve_epsilon()
        return construct_neighborhoods_broadcast(
            data, n_edit, self.configs, epsilon, self.maximize, self.verbose
        )

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

    @timeit
    def _identify_plateaus(self, neutral_pairs):
        """Build neutral network and identify connected components (plateaus).

        Each connected component of nodes linked by neutral (within-epsilon
        fitness) edges forms a plateau. Only multi-member plateaus are tracked;
        singleton nodes remain unaffected.

        Parameters
        ----------
        neutral_pairs : list[tuple[int, int]]
            Undirected neutral neighbor pairs from ``_construct_neighborhoods``.
        """
        if not neutral_pairs:
            self._has_plateaus = False
            return

        if self.graph is None or self.n_configs is None:
            self._has_plateaus = False
            return

        # Build a lightweight undirected graph just for component detection
        neutral_graph = ig.Graph(
            n=self.n_configs, edges=neutral_pairs, directed=False
        )
        components = neutral_graph.connected_components()

        # For singletons, plateau_id == node index (identity mapping).
        # Multi-member plateaus get IDs starting at n_configs to avoid collisions.
        node_to_plateau = np.arange(self.n_configs, dtype=np.int32)
        plateaus = {}
        next_plateau_id = self.n_configs

        for members in components:
            if len(members) > 1:
                member_list = sorted(members)
                plateaus[next_plateau_id] = member_list
                for node in member_list:
                    node_to_plateau[node] = next_plateau_id
                next_plateau_id += 1

        if not plateaus:
            self._has_plateaus = False
            return

        self._node_to_plateau = node_to_plateau
        self._plateaus = plateaus
        self._has_plateaus = True

        # Build per-node neutral neighbor adjacency
        neutral_neighbors = defaultdict(list)
        for u, v in neutral_pairs:
            neutral_neighbors[u].append(v)
            neutral_neighbors[v].append(u)
        self._neutral_neighbors = dict(neutral_neighbors)

        # Annotate graph vertices
        self.graph.vs["plateau_id"] = node_to_plateau.tolist()
        plateau_sizes = np.ones(self.n_configs, dtype=np.int32)
        for pid, members in plateaus.items():
            size = len(members)
            for node in members:
                plateau_sizes[node] = size
        self.graph.vs["plateau_size"] = plateau_sizes.tolist()

        if self.verbose:
            total_neutral_nodes = sum(len(m) for m in plateaus.values())
            print(
                f" - Identified {len(plateaus)} neutral plateaus "
                f"covering {total_neutral_nodes} nodes."
            )

    def _reconstruct_plateaus(self):
        """Rebuild plateau data structures from saved vertex attributes.

        Called by ``build_from_graph`` when a graph with ``plateau_id``
        attributes is loaded.  When ``self.configs`` is available, neutral
        neighbor adjacency (``_neutral_neighbors``) is also reconstructed so
        that ``determine_neighbor_fitness`` remains plateau-aware.
        """
        if self.graph is None or "plateau_id" not in self.graph.vs.attributes():
            self._has_plateaus = False
            return

        n = self.graph.vcount()
        node_to_plateau = np.array(self.graph.vs["plateau_id"], dtype=np.int32)

        plateaus = defaultdict(list)
        for node_idx in range(n):
            pid = int(node_to_plateau[node_idx])
            if pid != node_idx:
                plateaus[pid].append(node_idx)

        if not plateaus:
            self._has_plateaus = False
            return

        self._node_to_plateau = node_to_plateau
        self._plateaus = dict(plateaus)
        self._has_plateaus = True

        # Rebuild _neutral_neighbors from configs (single-edit adjacency
        # within each plateau).  Falls back to None if configs are unavailable.
        if self.configs is not None and len(self.configs) == n:
            neutral_neighbors = defaultdict(list)
            for pid, members in self._plateaus.items():
                member_configs = [np.array(self.configs.iloc[m]) for m in members]
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        if np.sum(member_configs[i] != member_configs[j]) == 1:
                            neutral_neighbors[members[i]].append(members[j])
                            neutral_neighbors[members[j]].append(members[i])
            self._neutral_neighbors = dict(neutral_neighbors) if neutral_neighbors else None
        else:
            self._neutral_neighbors = None

    def _find_plateau_exit(self, plateau_id, visited_plateaus):
        """Find a node in a plateau that has an improving edge to an unvisited plateau.

        Returns
        -------
        tuple[int, int] or tuple[None, None]
            (exit_node_in_plateau, successor_outside_plateau), or (None, None)
            if the plateau has no external exit.
        """
        for node in self._plateaus[plateau_id]:
            for successor in self.graph.successors(node):
                target_pid = int(self._node_to_plateau[successor])
                if target_pid != plateau_id and target_pid not in visited_plateaus:
                    return node, successor
        return None, None

    def _plateau_aware_climb(self, start_node, initial_lo, initial_steps):
        """Extend a hill climb across neutral plateaus.

        After a standard hill_climb reaches a node with out-degree 0, this
        method checks whether that node's plateau has an exit to a different,
        higher-fitness region. If so, the climb continues from that exit.

        Parameters
        ----------
        start_node : int
            The original starting node (unused except conceptually).
        initial_lo : int
            The local optimum reached by the initial hill climb.
        initial_steps : int
            Steps taken in the initial hill climb.

        Returns
        -------
        tuple[int, int]
            (final_lo, total_steps)
        """
        current_lo = initial_lo
        total_steps = initial_steps
        visited_plateaus = set()

        while True:
            pid = int(self._node_to_plateau[current_lo])
            if pid not in self._plateaus:
                break  # Singleton, not part of a multi-member plateau
            if pid in visited_plateaus:
                break
            visited_plateaus.add(pid)

            exit_node, exit_target = self._find_plateau_exit(pid, visited_plateaus)
            if exit_node is None:
                break  # Plateau is a true local optimum

            total_steps += 1
            next_lo, next_steps = hill_climb(self.graph, exit_target, "delta_fit")
            total_steps += next_steps
            current_lo = next_lo

        return current_lo, total_steps

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
        """Identifies local optima nodes in the landscape graph.

        A node is a per-node local optimum if its out-degree is 0 (no strictly
        improving neighbor). When plateaus are present (``epsilon > 0``), an
        additional plateau-level classification is computed: a plateau is a
        local optimum if **no** member has an improving edge to a node outside
        the plateau.

        Per-node results are stored in ``n_lo`` / ``lo_index`` / ``is_lo``
        for backward compatibility. Plateau-level results are stored in
        ``n_plateau_lo`` / ``plateau_lo_index`` / ``is_plateau_lo``.
        """

        if self.verbose:
            print(" - Determining local optima...")

        # --- Per-node LO detection (unchanged) ---
        out_degrees = self.graph.outdegree()
        is_lo = [deg == 0 for deg in out_degrees]
        self.graph.vs["is_lo"] = is_lo
        self.n_lo = sum(is_lo)
        self.lo_index = sorted([i for i, val in enumerate(is_lo) if val])

        if self.verbose:
            print(f"   - Found {self.n_lo} per-node local optima.")

        # --- Plateau-level LO detection ---
        if self._has_plateaus:
            n_vertices = self.graph.vcount()
            plateau_is_lo = {}

            for pid, members in self._plateaus.items():
                has_external_exit = False
                for node in members:
                    for successor in self.graph.successors(node):
                        if int(self._node_to_plateau[successor]) != pid:
                            has_external_exit = True
                            break
                    if has_external_exit:
                        break
                plateau_is_lo[pid] = not has_external_exit

            # For nodes not in any multi-member plateau, fall back to per-node is_lo
            is_plateau_lo = [False] * n_vertices
            for i in range(n_vertices):
                pid = int(self._node_to_plateau[i])
                if pid in plateau_is_lo:
                    is_plateau_lo[i] = plateau_is_lo[pid]
                else:
                    is_plateau_lo[i] = is_lo[i]

            self.graph.vs["is_plateau_lo"] = is_plateau_lo
            self.n_plateau_lo = (
                sum(v for v in plateau_is_lo.values() if v)
                + sum(1 for i in range(n_vertices)
                      if int(self._node_to_plateau[i]) not in plateau_is_lo and is_lo[i])
            )
            self.plateau_lo_index = [
                pid for pid, v in plateau_is_lo.items() if v
            ] + [
                i for i in range(n_vertices)
                if int(self._node_to_plateau[i]) not in plateau_is_lo and is_lo[i]
            ]

            if self.verbose:
                print(f"   - Found {self.n_plateau_lo} plateau-level local optima.")

    @timeit
    def determine_basin_of_attraction(self):
        """Calculates the basin of attraction for each node via hill climbing.

        For each node, a greedy hill climb follows improving edges until a
        per-node local optimum is reached. When plateaus are present
        (``_has_plateaus``), the climb is extended across neutral plateaus
        using ``_plateau_aware_climb``, and basin assignments are canonicalized
        so that all members of a plateau-LO share the same representative
        (minimum-index member).
        """
        if self.graph is None:
            raise RuntimeError("Graph is None.")
        if self.n_configs is None:
            raise RuntimeError("n_configs is None.")

        if self.verbose:
            print(" - Calculating basins of attraction via hill climbing...")

        n_vertices = self.graph.vcount()

        basin_indices = np.full(n_vertices, -1, dtype=np.int32)
        step_counts = np.zeros(n_vertices, dtype=np.int32)

        batch_size = 10000

        is_lo = (
            np.array(self.graph.vs["is_lo"])
            if "is_lo" in self.graph.vs.attributes()
            else None
        )

        batch_ranges = list(range(0, n_vertices, batch_size))
        nodes_iter = (
            tqdm(batch_ranges, desc="   - Hill climbing batches")
            if self.verbose
            else batch_ranges
        )

        for batch_start in nodes_iter:
            batch_end = min(batch_start + batch_size, n_vertices)

            for i in range(batch_start, batch_end):
                if is_lo is not None and is_lo[i]:
                    basin_indices[i] = i
                    step_counts[i] = 0
                    continue

                try:
                    lo, steps = hill_climb(self.graph, i, "delta_fit")
                    if self._has_plateaus:
                        lo, steps = self._plateau_aware_climb(i, lo, steps)
                    basin_indices[i] = lo
                    step_counts[i] = steps
                except Exception as e:
                    if self.verbose:
                        warnings.warn(
                            f"Hill climb failed for node {i}: {e}. "
                            "Assigning node to its own basin.",
                            RuntimeWarning,
                        )
                    basin_indices[i] = i
                    step_counts[i] = 0

        # Canonicalize: map plateau-LO members to a single representative
        if self._has_plateaus:
            plateau_representative = {}
            for pid, members in self._plateaus.items():
                plateau_representative[pid] = min(members)

            for i in range(n_vertices):
                lo_pid = int(self._node_to_plateau[basin_indices[i]])
                if lo_pid in plateau_representative:
                    basin_indices[i] = plateau_representative[lo_pid]

        # Calculate basin sizes efficiently using numpy
        unique_basins, basin_counts = np.unique(basin_indices, return_counts=True)
        basin_size_map = dict(zip(unique_basins, basin_counts))

        size_basin_values = np.zeros(n_vertices, dtype=np.int32)
        for i in range(n_vertices):
            size_basin_values[i] = basin_size_map.get(basin_indices[i], 0)

        # Calculate basin radii (max step count per basin)
        max_steps_per_basin = {}
        for i in range(n_vertices):
            basin_idx = basin_indices[i]
            max_steps_per_basin[basin_idx] = max(
                max_steps_per_basin.get(basin_idx, 0), step_counts[i]
            )

        radius_basin_values = np.zeros(n_vertices, dtype=np.int32)
        for i in range(n_vertices):
            radius_basin_values[i] = max_steps_per_basin.get(basin_indices[i], 0)

        self.graph.vs["basin_index"] = basin_indices.tolist()
        self.graph.vs["size_basin_greedy"] = size_basin_values.tolist()
        self.graph.vs["radius_basin_greedy"] = radius_basin_values.tolist()

        self._basin_calculated = True

        if self.verbose:
            print(f"   - Basins calculated for {len(unique_basins)} local optima.")

    @timeit
    def determine_accessible_paths(self):
        """Determines the size of basins based on accessible paths (ancestors).

        Identifies all nodes from which a local optimum can be reached via any
        fitness-increasing path. When plateaus are active, ancestors are computed
        for each plateau-level LO by taking the union of ancestors across all
        plateau members, assigned to the canonical representative (min index).

        Note
        ----
        This method is computationally intensive and might be slow for large
        landscapes.
        """
        if self.graph is None:
            raise RuntimeError("Graph is None.")

        if self.lo_index is None or not self.lo_index or self.n_lo is None:
            warnings.warn(
                "No local optima found or determined. Cannot calculate accessible paths.",
                RuntimeWarning,
            )
            self._path_calculated = False
            return

        if self.verbose:
            print(" - Determining accessible paths (ancestors)...")

        dict_size = defaultdict(int)

        try:
            if self._has_plateaus and self.plateau_lo_index:
                # Plateau-aware: union ancestors of all members per plateau-LO
                plateau_lo_items = []
                for item in self.plateau_lo_index:
                    if item in self._plateaus:
                        plateau_lo_items.append(("plateau", item))
                    else:
                        plateau_lo_items.append(("node", item))

                items_iter = (
                    tqdm(plateau_lo_items, desc="   - Finding ancestors")
                    if self.verbose
                    else plateau_lo_items
                )

                for kind, item_id in items_iter:
                    if kind == "plateau":
                        members = self._plateaus[item_id]
                        ancestors = set()
                        for member in members:
                            ancestors.update(self.graph.subcomponent(member, mode="in"))
                        representative = min(members)
                        dict_size[representative] = len(ancestors)
                    else:
                        ancestors_set = self.graph.subcomponent(item_id, mode="in")
                        dict_size[item_id] = len(ancestors_set)
            else:
                los_iter = (
                    tqdm(self.lo_index, total=self.n_lo, desc="   - Finding ancestors")
                    if self.verbose
                    else self.lo_index
                )
                for lo in los_iter:
                    ancestors_set = self.graph.subcomponent(lo, mode="in")
                    dict_size[lo] = len(ancestors_set)

            size_basin_accessible_values = [0] * self.graph.vcount()
            for vertex_id, basin_size in dict_size.items():
                size_basin_accessible_values[vertex_id] = basin_size

            self.graph.vs["size_basin_accessible"] = size_basin_accessible_values

            self._path_calculated = True
            if self.verbose:
                print(
                    f"   - Accessible paths calculated for {len(dict_size)} local optima."
                )
        except Exception as e:
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
        mean neighbor fitness along the improving (lower-fitness → higher-fitness)
        direction, i.e. mean_neighbor_fit(target) - mean_neighbor_fit(source).

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
                # Graph neighbors cover improving + worsening edges
                neighbors = self.graph.neighbors(vertex_idx, mode="all")
                # Include neutral neighbors when plateau layer is active
                if self._neutral_neighbors and vertex_idx in self._neutral_neighbors:
                    neighbors = list(
                        set(neighbors) | set(self._neutral_neighbors[vertex_idx])
                    )

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
            n_vertices = self.graph.vcount()
            if len(self.configs) != n_vertices:
                warnings.warn(
                    f"configs length ({len(self.configs)}) does not match graph "
                    f"vertex count ({n_vertices}). Skipping distance calculation.",
                    RuntimeWarning,
                )
                self._distance_calculated = False
                return

            configs = np.vstack(
                [self.configs.iloc[i] for i in range(n_vertices)]
            )
            go_config = configs[self.go_index]
            distances = distance(configs, go_config, self.data_types)

            self.graph.vs["dist_go"] = distances.tolist()
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
            if self._has_plateaus and self.n_plateau_lo is not None:
                print(f"Plateau-Level Local Optima (n_plateau_lo): {self.n_plateau_lo}")
                print(f"Neutral Plateaus: {len(self._plateaus)}")
            go_idx_str = (
                str(self.go_index)
                if self.go_index is not None
                else "Not Calculated/Found"
            )
            print(f"Global Optimum Index: {go_idx_str}")
            print(f"Maximize Fitness: {self.maximize}")
            print(f"Epsilon: {self.epsilon}")
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
                except ImportError:
                    raise ImportError(
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

        invariant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if invariant_cols:
            X = X.drop(columns=invariant_cols)
            data_types = {k: v for k, v in data_types.items() if k not in invariant_cols}
            self.data_types = data_types
            self.n_vars = len(data_types)
            if self.verbose:
                print(
                    f" - Removed {len(invariant_cols)} invariant variable(s); "
                    f"{self.n_vars} variable(s) remaining."
                )

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
        elif self.verbose:
            print(" - Skipping distance to Global Optimum calculation (not requested).")

        if calculate_neighbor_fit:
            self.determine_neighbor_fitness()
        elif self.verbose:
            print(" - Skipping neighbor fitness calculation (not requested).")

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
