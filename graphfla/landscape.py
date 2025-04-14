import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Any, Dict, Tuple, Union, Optional
from collections import defaultdict
from tqdm import tqdm
import warnings

from .lon import get_lon
from .algorithms import hill_climb
from .utils import add_network_metrics
from .distances import mixed_distance, hamming_distance

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Constants ---
ALLOWED_DATA_TYPES = {"boolean", "categorical", "ordinal"}
DNA_ALPHABET = ["A", "C", "G", "T"]
RNA_ALPHABET = ["A", "C", "G", "U"]
PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


# --- Base Landscape Class ---
class BaseLandscape:
    """Base class for representing and analyzing fitness landscapes.

    This class provides a foundational structure for fitness landscapes,
    conceptualized as a mapping from a genotype (configuration) space to
    a fitness value. It typically represents the landscape as a directed graph
    where nodes are genotypes and edges connect mutational neighbors, pointing
    towards fitter variants. This concept, originating from Wright's work
    (Wright 1932), is central to understanding evolutionary dynamics,
    adaptation, epistasis, and constraints in various biological systems
    like proteins, RNA, or populations adapting to environments (Papkou 2023,
    Li 2016, Puchta 2016, Poelwijk 2007, Carneiro 2010).

    Instances are typically created using factory methods like
    `Landscape.from_data` which selects an appropriate subclass based on the
    data type, or by direct subclass instantiation (e.g., `DNALandscape`).
    Direct use of `BaseLandscape` is possible for generic data types
    ('boolean', 'categorical', 'ordinal').

    The landscape graph and its properties (optima, basins, etc.) are populated
    by calling either the `from_data` or `from_graph` method after initialization.

    Parameters
    ----------
    verbose : bool, default=True
        Controls the verbosity of the output during landscape construction
        and analysis.

    Attributes
    ----------
    graph : networkx.DiGraph or None
        The directed graph representing the fitness landscape. Nodes represent
        configurations (genotypes) and edges connect neighboring configurations,
        typically pointing from lower to higher fitness if `maximize` is True.
        Each node usually has a 'fitness' attribute. Populated after calling
        `from_data` or `from_graph`.
    configs : pandas.Series or None
        A pandas Series mapping node indices (int) to their corresponding
        configuration representation (often a tuple). This represents the
        genotypes in the landscape. Populated after calling `from_data`.
    config_dict : dict or None
        A dictionary describing the encoding scheme for configuration variables.
        Keys are typically integer indices of variables, and values are
        dictionaries specifying properties like 'type' (e.g., 'boolean',
        'categorical') and 'max' (maximum encoded value). Populated after
        calling `from_data`.
    data_types : dict or None
        A dictionary specifying the data type for each variable in the
        configuration space (e.g., {'var_0': 'boolean', 'var_1': 'categorical'}).
        Validated and stored during `from_data`. Required for certain distance
        calculations.
    n_configs : int or None
        The total number of configurations (nodes) in the landscape graph.
        Populated after calling `from_data` or `from_graph`.
    n_vars : int or None
        The number of variables (dimensions) defining a configuration in the
        genotype space. Populated after calling `from_data` or inferred
        (potentially less reliably) by `from_graph`.
    n_edges : int or None
        The total number of directed edges (connections) in the landscape graph.
        Populated after calling `from_data` or `from_graph`.
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
    basin_index : dict[int, int] or None
        A dictionary mapping each node index to the index of the local optimum
        reached by an adaptive walk (e.g., hill climbing) starting from that
        node. Defines the basins of attraction (Papkou 2023). Populated if
        `calculate_basins` is True during construction or analysis.
    lon : networkx.DiGraph or None
        The Local Optima Network (LON) graph, if calculated via `get_lon`.
        Nodes in the LON are local optima from the main landscape graph, and
        edges represent accessibility between their basins (Ochoa 2021).
    has_lon : bool
        Flag indicating whether the LON has been calculated and stored in the
        `lon` attribute.
    maximize : bool
        Indicates whether the objective is to maximize (True) or minimize
        (False) the fitness values. Set during `from_data` or `from_graph`.
    epsilon : float or str
        Tolerance for floating point comparisons, potentially used in
        determining neighbors or optima. Set during `from_data` or `from_graph`.
    verbose : bool
        The verbosity level set during initialization or construction.
    _is_built : bool
        Internal flag indicating if the landscape has been populated via
        `from_data` or `from_graph`.

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
    >>> # landscape = Landscape.from_data(X_data, f_data, data_types='boolean')

    >>> # Or, instantiate BaseLandscape directly for generic types
    >>> landscape = BaseLandscape(verbose=False)
    >>> landscape.from_data(X_data, f_data, data_types=data_types_dict,
    ...                     maximize=True, calculate_basins=True)
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

    def __init__(self, verbose: bool = True):
        """Initializes an empty BaseLandscape object.

        Parameters
        ----------
        verbose : bool, default=True
            Controls the verbosity of the output during landscape construction
            and analysis methods like `from_data` or `from_graph`.
        """
        self.graph: Optional[nx.DiGraph] = None
        self.configs: Optional[pd.Series] = None
        self.config_dict: Optional[Dict[int, Dict[str, Any]]] = None
        self.data_types: Optional[Dict[str, str]] = None
        self.n_configs: Optional[int] = None
        self.n_vars: Optional[int] = None
        self.n_edges: Optional[int] = None
        self.n_lo: Optional[int] = None
        self.lo_index: Optional[List[int]] = None
        self.go_index: Optional[int] = None
        self.go: Optional[Dict[str, Any]] = None
        self.basin_index: Optional[Dict[int, int]] = None
        self.lon: Optional[nx.DiGraph] = None
        self.has_lon: bool = False

        self.maximize: bool = True
        self.epsilon: Union[float, str] = "auto"
        self.verbose: bool = verbose

        self._is_built: bool = False
        self._path_calculated = False
        self._lo_determined = False
        self._basin_calculated = False

        if self.verbose:
            print(
                "Empty BaseLandscape object initialized. "
                "Call from_data() or from_graph() to populate."
            )

    def _check_built(self) -> None:
        """Raise an error if the landscape hasn't been built yet."""
        if not self._is_built:
            raise RuntimeError(
                "Landscape has not been built yet. Call from_data() or "
                "from_graph() first."
            )

    def from_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        f: Union[pd.Series, list, np.ndarray],
        data_types: Dict[str, str],
        *,
        maximize: bool = True,
        epsilon: Union[float, str] = "auto",
        calculate_basins: bool = True,
        calculate_paths: bool = False,
        impute: bool = False,
        impute_model: Optional[Any] = None,
        verbose: Optional[bool] = None,
        n_edit: int = 1,
    ) -> "BaseLandscape":
        """Construct the landscape graph and properties from configuration data.

        This method takes genotype-phenotype data (configurations `X` and their
        corresponding fitness values `f`) and builds the underlying graph
        structure of the fitness landscape. Nodes represent configurations, and
        edges connect neighbors based on the specified edit distance (`n_edit`).
        It then calculates various landscape properties like local optima,
        basins of attraction, and the global optimum.

        This method populates the core attributes of the `BaseLandscape`
        instance.

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
        maximize : bool, default=True
            Determines the optimization direction. If True, the landscape seeks
            higher fitness values (peaks are optima). If False, it seeks lower
            fitness values (valleys are optima).
        epsilon : float or 'auto', default='auto'
            Tolerance value used for floating-point comparisons when determining
            fitness improvements or optima. If 'auto', a default might be used.
        calculate_basins : bool, default=True
            If True, calculates the basins of attraction for each local optimum
            using a hill-climbing algorithm. This identifies which configurations
            lead to which peak. Populates the `basin_index` attribute.
        calculate_paths : bool, default=False
            If True, calculates accessible paths (ancestors) for local optima.
            This can be computationally intensive for large landscapes and is
            skipped if `n_configs` exceeds 200,000. Populates the
            `size_basin_first` node attribute.
        impute : bool, default=False
            If True, attempts to fill in missing fitness values (`NaN` in `f`)
            using a regression model based on the configurations `X`. Requires
            scikit-learn or a user-provided `impute_model`.
        impute_model : object, optional
            A custom model object with `fit` and `predict` methods used for
            imputation if `impute=True`. If None and `impute=True`, a
            `RandomForestRegressor` from scikit-learn is used by default (if
            available).
        verbose : bool, optional
            If provided, overrides the instance's verbosity setting for this
            method call. Controls printed output during the build process.
        n_edit : int, default=1
            The edit distance defining the neighborhood. An edge exists between
            two configurations if their distance (typically Hamming or mixed)
            is less than or equal to `n_edit`. Default is 1, connecting only
            adjacent mutational neighbors.

        Returns
        -------
        self : BaseLandscape
            The instance itself, now populated with the graph and landscape
            properties derived from the input data.

        Raises
        ------
        RuntimeError
            If the `from_data` or `from_graph` method has already been called
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
        1.  Input standardization and validation (`_validate_data`).
        2.  Data preparation, including encoding (`_prepare_data`).
        3.  Neighborhood identification based on `n_edit` (`_construct_neighborhoods`).
        4.  Graph construction (`_construct_landscape`).
        5.  Graph analysis to determine properties like optima and basins
            (`_analyze_graph`).

        Epistasis (non-additive interactions between mutations) is implicitly
        captured by the structure and fitness values within the landscape graph
        but is not explicitly quantified by this base class. The ruggedness
        (number of peaks) and pathways reflect these interactions
        (Weinreich 2006, Phillips 2008, Bank 2022).
        """
        if self._is_built:
            raise RuntimeError(
                "This BaseLandscape instance has already been built. Create a "
                "new instance to rebuild."
            )

        if verbose is not None:
            self.verbose = verbose

        if self.verbose:
            print(f"Building {self.__class__.__name__} from data...")

        self.maximize = maximize
        self.epsilon = epsilon
        self._should_calculate_basins = calculate_basins
        self._should_calculate_paths = calculate_paths

        if X is None or f is None or data_types is None:
            raise ValueError("Arguments 'X', 'f', and 'data_types' are required.")
        if len(X) != len(f):
            raise ValueError(
                f"Inconsistent lengths: X has {len(X)} rows, f has {len(f)} elements."
            )
        if len(X) == 0:
            raise ValueError("Input data 'X' and 'f' cannot be empty.")

        X_validated, f_validated, dt_validated = self._validate_data(
            X, f, data_types, impute, impute_model
        )
        self.data_types = dt_validated

        if len(X_validated) == 0:
            raise ValueError("All data removed during validation. Cannot build.")

        processed_data = self._prepare_data(X_validated, f_validated, self.data_types)

        self.n_configs = len(X_validated)
        self.n_vars = len(self.data_types)

        if self.verbose:
            print("Constructing landscape graph...")
        edge_list = self._construct_neighborhoods(processed_data, n_edit=n_edit)
        graph = self._construct_landscape(processed_data, edge_list)

        self.n_configs = graph.number_of_nodes()
        self.graph = graph

        if self.n_configs == 0:
            raise RuntimeError(
                "Landscape graph construction resulted in an empty graph."
            )

        self._analyze_graph(
            analyze_distance=True,
            calculate_basins=calculate_basins,
            calculate_paths=calculate_paths,
        )

        self._is_built = True
        if self.verbose:
            print(f"{self.__class__.__name__} built successfully.\n")
            self.describe()

        return self

    def to_graph(self) -> nx.DiGraph:
        """Export the landscape as a NetworkX DiGraph with serialized attributes.

        This method creates a copy of the internal graph and enriches it with
        all the metadata needed to reconstruct the landscape using from_graph().
        The result can be used for persistence (saving/loading) or transferring
        the landscape between processes.

        Returns
        -------
        networkx.DiGraph
            A directed graph containing all landscape information as graph attributes.

        Raises
        ------
        RuntimeError
            If the landscape hasn't been built yet.

        See Also
        --------
        from_graph : Reconstruct a landscape from a graph created by to_graph.

        Examples
        --------
        >>> landscape = BaseLandscape()
        >>> # ... build landscape with from_data ...
        >>> graph = landscape.to_graph()
        >>> # Save graph using networkx methods if needed
        >>> # nx.write_gpickle(graph, "landscape.gpickle")
        >>> # Later, reconstruct the landscape
        >>> new_landscape = BaseLandscape().from_graph(graph)
        """
        self._check_built()

        if self.graph is None:
            raise RuntimeError("Cannot export: internal graph is None.")

        # Create a copy of the internal graph to avoid modifying the original
        exported_graph = self.graph.copy()

        # Store landscape-level attributes as graph attributes
        landscape_attrs = {
            "landscape_class": self.__class__.__name__,
            "configs": self.configs.to_json() if self.configs is not None else None,
            "config_dict": self.config_dict,
            "data_types": self.data_types,
            "n_configs": self.n_configs,
            "n_vars": self.n_vars,
            "n_edges": self.n_edges,
            "n_lo": self.n_lo,
            "lo_index": self.lo_index,
            "go_index": self.go_index,
            "basin_index": self.basin_index,
            "maximize": self.maximize,
            "epsilon": self.epsilon,
            "_lo_determined": self._lo_determined,
            "_basin_calculated": self._basin_calculated,
            "_path_calculated": self._path_calculated,
        }

        # Add the landscape attributes to the graph
        nx.set_graph_attributes(exported_graph, landscape_attrs)

        return exported_graph

    def from_graph(
        self,
        graph: nx.DiGraph,
        *,
        analyze_distance: bool = True,
        calculate_basins: bool = False,
        calculate_paths: bool = False,
        verbose: Optional[bool] = None,
    ) -> "BaseLandscape":
        """Reconstruct a landscape from a NetworkX DiGraph with embedded attributes.

        This method populates the landscape object using a graph created by to_graph().
        It restores all the landscape attributes and structure from the graph.

        Parameters
        ----------
        graph : networkx.DiGraph
            A directed graph containing landscape information as graph attributes,
            typically created by the to_graph() method.
        analyze_distance : bool, default=True
            Whether to calculate distances to the global optimum. Requires configs
            and data_types to be present in the graph.
        calculate_basins : bool, default=False
            Whether to recalculate basins of attraction. If False, basins will be
            loaded from the graph if available.
        calculate_paths : bool, default=False
            Whether to recalculate accessible paths. If False, path information will
            be loaded from the graph if available.
        verbose : bool, optional
            If provided, overrides the instance's verbosity setting for this method call.

        Returns
        -------
        self : BaseLandscape
            The populated landscape instance.

        Raises
        ------
        RuntimeError
            If the landscape has already been built.
            If the provided graph is missing required attributes.
        ValueError
            If the provided graph is of an incorrect type or empty.

        See Also
        --------
        to_graph : Export a landscape as a graph for later reconstruction.

        Notes
        -----
        This method is the inverse of to_graph() and should restore the landscape
        to a state equivalent to what would be achieved through from_data().
        Depending on the parameters, some analyses like basin calculation might
        be either loaded from the graph or recalculated.
        """
        if self._is_built:
            raise RuntimeError(
                "This BaseLandscape instance has already been built. Create a "
                "new instance to rebuild."
            )

        if verbose is not None:
            self.verbose = verbose

        if self.verbose:
            print(f"Building {self.__class__.__name__} from graph...")

        # Validate input
        if not isinstance(graph, nx.DiGraph):
            raise ValueError(f"Expected a NetworkX DiGraph, got {type(graph)}.")

        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot build from an empty graph.")

        # Extract graph attributes
        graph_attrs = graph.graph

        # Set basic attributes from graph
        self.graph = graph.copy()  # Use a copy to avoid modifying the input

        # Extract and set landscape attributes from graph attributes
        required_attributes = ["maximize", "data_types"]
        for attr in required_attributes:
            if attr not in graph_attrs:
                raise RuntimeError(f"Required attribute '{attr}' missing from graph.")

        # Set optimization direction and epsilon
        self.maximize = graph_attrs.get("maximize", True)
        self.epsilon = graph_attrs.get("epsilon", "auto")

        # Set configs if available
        configs_json = graph_attrs.get("configs")
        if configs_json is not None:
            self.configs = pd.read_json(configs_json, typ="series")
        else:
            self.configs = None

        # Set other attributes
        self.config_dict = graph_attrs.get("config_dict")
        self.data_types = graph_attrs.get("data_types")
        self.n_configs = graph.number_of_nodes()  # Use actual count from graph
        self.n_vars = graph_attrs.get("n_vars")
        self.n_edges = graph.number_of_edges()  # Use actual count from graph

        # Handle optional analysis results
        self._lo_determined = graph_attrs.get("_lo_determined", False)
        if self._lo_determined:
            self.n_lo = graph_attrs.get("n_lo")
            self.lo_index = graph_attrs.get("lo_index")
            self.go_index = graph_attrs.get("go_index")
            if self.go_index is not None:
                self.go = graph.nodes.get(self.go_index, {})
        else:
            self.n_lo = None
            self.lo_index = None
            self.go_index = None
            self.go = None

        self._basin_calculated = graph_attrs.get("_basin_calculated", False)
        if self._basin_calculated and not calculate_basins:
            self.basin_index = graph_attrs.get("basin_index", {})
        else:
            self.basin_index = None

        self._path_calculated = graph_attrs.get("_path_calculated", False)

        # Run necessary analyses based on parameters
        # If we need to recompute any properties or they weren't in the graph
        if (
            (not self._lo_determined)
            or calculate_basins
            or calculate_paths
            or analyze_distance
        ):
            self._infer_properties_from_graph()
            self._analyze_graph(
                analyze_distance=analyze_distance,
                calculate_basins=calculate_basins or not self._basin_calculated,
                calculate_paths=calculate_paths or not self._path_calculated,
            )

        self._is_built = True
        self.has_lon = False  # LON is not transferred via to_graph/from_graph
        self.lon = None

        if self.verbose:
            print(f"{self.__class__.__name__} built successfully from graph.\n")
            self.describe()

        return self

    def _infer_properties_from_graph(self):
        """Set basic landscape properties based on the assigned self.graph."""
        if self.graph is None:
            # This check is primarily for internal consistency, should not be
            # reachable if called correctly by from_graph.
            raise RuntimeError(
                "_infer_properties_from_graph called before graph assignment."
            )

        self.n_configs = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()

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
                sample_node_attrs = self.graph.nodes[next(iter(self.graph.nodes()))]
                # Heuristic: look for attributes like 'var_0', 'pos_1', etc.
                potential_var_keys = [
                    k
                    for k in sample_node_attrs
                    if isinstance(k, str) and (k.startswith(("var_", "pos_", "bit_")))
                ]
                if potential_var_keys:
                    self.n_vars = len(potential_var_keys)
                else:
                    self.n_vars = None  # No obvious variable attributes
            except Exception:
                self.n_vars = None  # Failed inference

        if self.n_vars is None and self.verbose:
            warnings.warn(
                "Could not reliably determine 'n_vars' (number of variables) "
                "from the provided graph and parameters. Distance calculations "
                "or analyses requiring dimensionality might fail.",
                UserWarning,
            )

    def _analyze_graph(
        self, analyze_distance: bool, calculate_basins: bool, calculate_paths: bool
    ) -> None:
        """Internal helper to run analysis steps on the landscape graph.

        This method orchestrates the calculation of key landscape properties
        after the graph has been constructed or provided.

        Parameters
        ----------
        analyze_distance : bool
            Whether to calculate distance-based metrics, specifically the
            distance of each node to the global optimum. Requires `configs`
            and `data_types` attributes to be set.
        calculate_basins : bool
            Whether to calculate basins of attraction.
        calculate_paths : bool
            Whether to calculate accessible paths (ancestors).
        """
        if self.graph is None:
            # This check is primarily for internal consistency.
            raise RuntimeError("Graph is None, cannot analyze.")

        if self.graph.number_of_nodes() == 0:
            warnings.warn("Cannot analyze an empty graph.", RuntimeWarning)
            # Reset analysis attributes for consistency
            self.n_lo = 0
            self.lo_index = []
            self.go_index = None
            self.go = None
            self.basin_index = {}
            self.lon = None
            self.has_lon = False
            self._lo_determined = True
            self._basin_calculated = calculate_basins  # Mark as attempted/done
            self._path_calculated = calculate_paths  # Mark as attempted/done
            return

        if self.verbose:
            print("Calculating landscape properties...")

        # Add basic network metrics if they don't already exist
        if not nx.get_node_attributes(self.graph, "out_degree"):
            if self.verbose:
                print(" - Adding network metrics (degrees)...")
            # Assumes 'delta_fit' exists if built from data, might need adjustment
            # if graph provided externally lacks this weight.
            weight_key = (
                "delta_fit"
                if "delta_fit" in next(iter(self.graph.edges(data=True)))[-1]
                else None
            )
            self.graph = self._add_network_metrics(self.graph, weight=weight_key)

        # Determine Optima
        self._determine_local_optima()
        self._determine_global_optimum()  # Needs LO info if maximize=False? Check logic. GO depends only on fitness attr.

        # Determine Basins (requires optima info)
        if calculate_basins:
            self._determine_basin_of_attraction()
        else:
            self._basin_calculated = False
            self.basin_index = None
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
        if analyze_distance:
            if (
                self.configs is not None
                and self.go_index is not None
                and self.data_types is not None
            ):
                # Allow subclasses to specify a different default distance metric
                distance_func = getattr(
                    self, "_get_default_distance_metric", lambda: mixed_distance
                )()
                self._determine_dist_to_go(distance=distance_func)
            elif self.verbose:
                print(
                    "   - Skipping distance to Global Optimum calculation "
                    "(requires configuration data, data_types, and successful "
                    "GO identification)."
                )
        elif self.verbose:
            print(" - Skipping distance to Global Optimum calculation (not requested).")

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
                if not SKLEARN_AVAILABLE:
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

    def _validate_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        f: Union[pd.Series, list, np.ndarray],
        data_types: Dict[str, str],
        impute: bool,
        impute_model: Optional[Any],
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
        """Orchestrates the data validation and cleaning pipeline."""
        if self.verbose:
            print("Validating input data...")
        try:
            initial_count = len(X)
        except TypeError:
            initial_count = -1

        X_std, f_std, dt_std = self._standardize_input_formats(X, f, data_types)

        temp_dt_for_imputation = dt_std.copy()
        x_cols_std = set(X_std.columns)
        dt_keys_std = set(temp_dt_for_imputation.keys())
        if x_cols_std != dt_keys_std:
            if not all(c.startswith("var_") for c in x_cols_std) or len(
                x_cols_std
            ) != len(dt_keys_std):
                warnings.warn(
                    f"Column names in standardized X ({x_cols_std}) do not perfectly match "
                    f"initial data_types keys ({dt_keys_std}). Imputation might behave unexpectedly "
                    "if relying on specific column names. Final validation occurs later.",
                    UserWarning,
                )
            if len(x_cols_std) == len(dt_keys_std):
                temp_dt_for_imputation = {
                    x_col: list(dt_std.values())[i]
                    for i, x_col in enumerate(X_std.columns)
                }
                if self.verbose:
                    print(
                        "   - Aligning data_types keys to standardized X columns for imputation."
                    )
            else:
                raise ValueError(
                    "Cannot proceed with imputation: Mismatch in number of columns "
                    "between standardized X and provided data_types."
                )

        X_clean, f_clean = self._handle_missing_values(
            X_std, f_std, temp_dt_for_imputation, impute, impute_model
        )

        if len(X_clean) == 0:
            warnings.warn(
                "All data removed during missing value handling.", RuntimeWarning
            )
            return X_clean, f_clean, {}

        X_unique, f_unique = self._handle_duplicates(X_clean, f_clean)

        if len(X_unique) == 0:
            warnings.warn("All data removed after handling duplicates.", RuntimeWarning)
            return X_unique, f_unique, {}

        final_dt = self._validate_data_types_dict(X_unique, dt_std)

        final_count = len(X_unique)
        if self.verbose:
            msg = "Data validation complete."
            if initial_count >= 0:
                msg += f" Kept {final_count}/{initial_count} configurations."
            else:
                msg += f" Kept {final_count} configurations."
            print(msg)

        return X_unique, f_unique, final_dt

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
        data_for_attributes.index = X.index  # Ensure index alignment is maintained

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

    def _generate_neighbors(
        self, config: Tuple, config_dict: Dict, n_edit: int
    ) -> List[Tuple]:
        """Generates neighbors for a given configuration (default implementation).

        This method defines the neighborhood structure based on edit distance.
        For `n_edit=1`, it generates all configurations reachable by changing
        a single variable according to its type (flipping boolean, changing
        category/ordinal value). Subclasses (e.g., for sequences) often
        override this method.

        Parameters
        ----------
        config : tuple
            The encoded configuration (genotype) for which to find neighbors.
        config_dict : dict
            Dictionary describing the encoding for each variable index.
        n_edit : int
            The edit distance defining the neighborhood.

        Returns
        -------
        list[tuple]
            A list of neighboring encoded configurations.
        """
        neighbors = []
        num_vars = len(config)

        if n_edit == 1:
            for i in range(num_vars):
                info = config_dict[i]
                current_val = config[i]
                dtype = info["type"]

                if dtype == "boolean":
                    # Flip the bit (0 to 1, 1 to 0)
                    new_vals = [1 - current_val]
                elif dtype in ["categorical", "ordinal"]:
                    # Iterate through all possible values for this variable
                    max_val = info["max"]
                    new_vals = [v for v in range(max_val + 1) if v != current_val]
                else:
                    # Should not happen with validated data types
                    warnings.warn(
                        f"Unsupported dtype '{dtype}' in _generate_neighbors, skipping var {i}",
                        RuntimeWarning,
                    )
                    continue

                # Create neighbor tuples
                for new_val in new_vals:
                    neighbor_list = list(config)
                    neighbor_list[i] = new_val
                    neighbors.append(tuple(neighbor_list))
        else:
            # Implementation for n_edit > 1 is more complex and not provided
            # in this base class. Subclasses might implement it.
            warnings.warn(
                f"Neighbor generation for n_edit={n_edit} is not implemented "
                "in BaseLandscape. Only n_edit=1 is supported here.",
                UserWarning,
            )
            # Returning empty list as per the warning.
        return neighbors

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

        edge_list = []
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
            self, "_generate_neighbors", BaseLandscape._generate_neighbors
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
                    # TODO: Explicit epsilon handling might be needed here based on self.epsilon
                    is_improvement = (self.maximize and delta_fit < 0) or (
                        not self.maximize and delta_fit > 0
                    )

                    # Add edge if neighbor is fitter (points towards higher fitness)
                    if is_improvement:
                        # Store edge as (source, target, weight)
                        # Weight is absolute difference |current_fit - neighbor_fit|
                        edge_list.append((current_id, neighbor_idx, abs(delta_fit)))

        if self.verbose:
            print(f" - Identified {len(edge_list)} potential improving connections.")
        return edge_list

    def _construct_landscape(self, data, edge_list):
        """Builds the NetworkX graph object from nodes and edges."""
        if self.verbose:
            print(" - Constructing graph object...")
        graph = nx.DiGraph()

        # Add all nodes from the validated data first to ensure all configurations
        # are represented, even if disconnected.
        graph.add_nodes_from(data.index)

        # Add edges representing fitness-increasing transitions
        # 'delta_fit' is used as the weight attribute
        graph.add_weighted_edges_from(edge_list, weight="delta_fit")

        if self.verbose:
            print(" - Adding node attributes (fitness, etc.)...")
        # Add original features and fitness as node attributes
        for column in data.columns:
            # Ensure column name is suitable for networkx attribute name (e.g., string)
            attr_name = str(column)
            nx.set_node_attributes(graph, data[column].to_dict(), attr_name)

        self.n_edges = graph.number_of_edges()  # Update edge count based on final graph

        # Check if node count changed (e.g., due to disconnected nodes not in edge_list)
        # This check is mainly relevant when building from data.
        original_node_count = len(data)
        final_node_count = graph.number_of_nodes()
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
                nodes_in_graph = set(graph.nodes())
                self.configs = self.configs[
                    self.configs.index.map(lambda idx: idx in nodes_in_graph)
                ]

        return graph

    def _add_network_metrics(self, graph, weight):
        """Calculates and adds node/edge metrics to the graph."""
        if self.verbose:
            print(" - Calculating network metrics (degrees)...")
        # Assumes utils.add_network_metrics exists and modifies the graph in place
        # or returns the modified graph.
        return add_network_metrics(graph, weight=weight)

    def _determine_local_optima(self):
        """Identifies local optima nodes in the landscape graph."""
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        if self.verbose:
            print(" - Determining local optima...")

        # Local optima are nodes with out-degree 0 (no outgoing edges to fitter neighbors)
        out_degrees = dict(self.graph.out_degree())
        # Create a dictionary mapping node index to boolean (True if LO)
        is_lo = {node: out_degrees.get(node, 0) == 0 for node in self.graph.nodes}

        # Add 'is_lo' attribute to graph nodes
        nx.set_node_attributes(self.graph, is_lo, "is_lo")

        # Calculate and store LO count and indices
        self.n_lo = sum(is_lo.values())
        # Convert dict to Series for efficient filtering
        is_lo_series = pd.Series(is_lo)
        # Get indices where is_lo is True and sort them
        self.lo_index = sorted(list(is_lo_series[is_lo_series].index))
        self._lo_determined = True
        if self.verbose:
            print(f"   - Found {self.n_lo} local optima.")

    def _determine_basin_of_attraction(self):
        """Calculates the basin of attraction for each node using hill climbing."""
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        if self.n_configs is None:
            raise RuntimeError("n_configs is None.")  # Internal check

        if self.verbose:
            print(" - Calculating basins of attraction via hill climbing...")

        basin_index = defaultdict(int)  # Maps node index -> its LO index
        dict_size = defaultdict(int)  # Stores size of each basin (number of nodes)
        dict_diameter = defaultdict(list)  # Stores path lengths within each basin

        # Prepare iterator with progress bar if verbose
        nodes_iter = (
            tqdm(
                list(self.graph.nodes), total=self.n_configs, desc="   - Hill climbing"
            )
            if self.verbose
            else list(self.graph.nodes)
        )

        # Perform hill climbing for each node
        for i in nodes_iter:
            try:
                # hill_climb function should return the index of the LO reached
                # and the number of steps taken. Assumes 'delta_fit' edge weight.
                lo, steps = hill_climb(self.graph, i, "delta_fit")
                basin_index[i] = lo
                dict_size[lo] += 1
                dict_diameter[lo].append(steps)
            except Exception as e:
                # Handle cases where hill climbing might fail (e.g., complex cycles)
                warnings.warn(
                    f"Hill climb failed for node {i}: {e}. Assigning node to its own basin.",
                    RuntimeWarning,
                )
                basin_index[i] = i  # Assign node to its own basin as fallback
                dict_size[i] += 1
                dict_diameter[i].append(0)

        # Add basin information as node attributes
        nx.set_node_attributes(self.graph, basin_index, "basin_index")
        # 'size_basin_best' -> size of the basin node belongs to
        nx.set_node_attributes(self.graph, dict_size, "size_basin")
        # 'max_radius_basin' -> longest path to LO within the basin
        nx.set_node_attributes(
            self.graph,
            {k: max(v) if v else 0 for k, v in dict_diameter.items()},
            "radius_basin",
        )

        # Store the basin index map
        self.basin_index = dict(basin_index)
        self._basin_calculated = True
        if self.verbose:
            print(f"   - Basins calculated for {len(dict_size)} local optima.")

    def determine_accessible_paths(self):
        """Determines the size of basins based on accessible paths (ancestors).

        This method calculates the basin size differently from hill climbing.
        It identifies all nodes from which a local optimum can be reached via
        any path (not just strictly increasing fitness paths considered by
        hill climbing). This uses `nx.ancestors` and can be computationally
        expensive. The result is stored as the 'size_basin_accessible' node attribute.

        Note
        ----
        This method is computationally intensive and might be slow for large
        landscapes. It is automatically skipped by `from_data` or `from_graph`
        if the landscape size exceeds a threshold (currently 200,000 nodes).
        """
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        if not self._lo_determined:
            # Ensure local optima are identified first
            self._determine_local_optima()

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
                # nx.ancestors finds all nodes from which 'lo' is reachable.
                ancestors_set = nx.ancestors(self.graph, lo)
                # Basin size includes the ancestors plus the local optimum itself.
                dict_size[lo] = len(ancestors_set) + 1

            # Add the result as a node attribute
            nx.set_node_attributes(self.graph, dict(dict_size), "size_basin_accessible")
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

    def _determine_global_optimum(self):
        """Identifies the global optimum node in the landscape graph."""
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        if self.verbose:
            print(" - Determining global optimum...")

        fitness_attr = nx.get_node_attributes(self.graph, "fitness")
        if not fitness_attr:
            warnings.warn(
                "Cannot determine global optimum: 'fitness' attribute missing from graph nodes.",
                RuntimeWarning,
            )
            self.go_index = None
            self.go = None
            return

        fitness_series = pd.Series(fitness_attr)
        if self.maximize:
            self.go_index = fitness_series.idxmax()
        else:
            self.go_index = fitness_series.idxmin()

        try:
            # Store the attributes of the global optimum node
            self.go = self.graph.nodes[self.go_index]
            if self.verbose:
                print(
                    f"   - Global optimum found at index {self.go_index} with fitness {self.go['fitness']:.4f}."
                )
        except KeyError:
            # Handle case where the identified index might not be in the graph
            # (shouldn't happen with current logic, but safeguard)
            warnings.warn(
                f"Global optimum index {self.go_index} not found in graph nodes. Resetting GO.",
                RuntimeWarning,
            )
            self.go_index = None
            self.go = None

    def _get_default_distance_metric(self):
        """Returns the default distance metric function.

        Subclasses can override this to provide landscape-specific distance
        metrics (e.g., Hamming distance for sequences).

        Returns
        -------
        callable
            The distance function to use (defaults to `mixed_distance`).
        """
        return mixed_distance

    def _determine_dist_to_go(self, distance):
        """Calculates the distance from each node to the global optimum."""
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        if self.configs is None or self.go_index is None or self.data_types is None:
            if self.verbose:
                print(
                    "   - Skipping distance to GO (missing configs, go_index, or data_types)."
                )
            return
        # Ensure the global optimum's configuration is available
        if self.go_index not in self.configs.index:
            if self.verbose:
                print(
                    f"   - Skipping distance to GO (GO index {self.go_index} not found in configs map)."
                )
            return

        if self.verbose:
            print(
                f" - Calculating distances to global optimum using {distance.__name__}..."
            )

        try:
            # Prepare configurations for distance calculation
            # Assumes self.configs contains encoded tuples/lists
            configs_list = self.configs.tolist()
            # Convert to numpy array for efficient distance calculation
            configs_array = np.array(configs_list)
            # Get the configuration tuple of the global optimum
            go_config_tuple = self.configs.loc[self.go_index]
            # Convert GO config to array for the distance function
            go_config_array = np.array(go_config_tuple)

            # Calculate distances from all configs to the GO config
            # The distance function signature is assumed to be:
            # distance(all_configs_array, reference_config_array, data_types_dict)
            distances = distance(configs_array, go_config_array, self.data_types)

            if len(distances) != len(self.configs.index):
                # This check ensures the distance function returned expected output
                raise ValueError(
                    "Length mismatch between calculated distances and configs index."
                )

            # Add calculated distances as a node attribute 'dist_go'
            nx.set_node_attributes(
                self.graph, dict(zip(self.configs.index, distances)), "dist_go"
            )
            if self.verbose:
                print(
                    "   - Distances to GO calculated and added as node attribute 'dist_go'."
                )

        except KeyError as e:
            warnings.warn(
                f"KeyError calculating distance to GO: {e}. "
                "Check consistency between go_index and self.configs.",
                RuntimeWarning,
            )
        except Exception as e:
            warnings.warn(
                f"Error calculating distance to GO using {distance.__name__}: {e}",
                RuntimeWarning,
            )

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
            If the landscape has not been built (via `from_data` or
            `from_graph`) before calling this method.
            If `lo_only=True` and the LON graph (`self.lon`) is unexpectedly
            None despite `self.has_lon` being True.
        """
        self._check_built()
        if self.graph is None:
            # This check is primarily for internal consistency.
            raise RuntimeError("Graph is None despite landscape being built.")

        if lo_only:
            if not self._lo_determined:
                # Calculate local optima if not already done
                self._determine_local_optima()
            if self.lo_index is None or not self.lo_index:
                warnings.warn(
                    "Local optima not determined or none found. Cannot filter for LO.",
                    RuntimeWarning,
                )
                return pd.DataFrame()  # Return empty DataFrame

            if not self.has_lon:
                if self.verbose:
                    print(
                        "Extracting data for local optima from the main graph (LON not computed)."
                    )
                # Use the subgraph containing only local optima nodes
                graph_to_use = self.graph.subgraph(self.lo_index)
                # Columns typically irrelevant for LO-only view from main graph
                cols_to_drop = ["is_lo", "out_degree", "in_degree", "basin_index"]
            else:
                if self.lon is None:
                    raise RuntimeError("LON graph is None despite has_lon=True.")
                if self.verbose:
                    print("Extracting data from the Local Optima Network (LON).")
                graph_to_use = self.lon
                # LON graph attributes are specific, usually no standard columns to drop
                cols_to_drop = []

            # Convert node attributes to DataFrame
            data_lo = pd.DataFrame.from_dict(
                dict(graph_to_use.nodes(data=True)), orient="index"
            )
            # Sort by index for consistency
            data_lo.sort_index(inplace=True)
            # Drop potentially irrelevant columns, ignoring errors if columns don't exist
            return data_lo.drop(columns=cols_to_drop, errors="ignore")

        else:
            # Return data for all nodes in the main graph
            if self.verbose:
                print("Extracting data for all configurations from the main graph.")
            data = pd.DataFrame.from_dict(
                dict(self.graph.nodes(data=True)), orient="index"
            )
            # Sort by index
            data.sort_index(inplace=True)
            # Drop intermediate calculation columns if they exist
            # These were used to compute basin properties but are less relevant for final output
            if self._path_calculated:
                cols_to_drop = [
                    "size_basin",
                    "radius_basin",
                    "size_basin_accessible",
                ]
            else:
                cols_to_drop = ["size_basin", "radius_basin"]
            return data.drop(columns=cols_to_drop, errors="ignore")

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

    def get_lon(self, *args, **kwargs) -> nx.DiGraph:
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
        networkx.DiGraph
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
        if not self._lo_determined:
            self._determine_local_optima()  # Calculate if needed

        # Check for required attributes set by from_data or provided to from_graph
        if self.configs is None or self.lo_index is None or self.config_dict is None:
            raise RuntimeError(
                "Cannot compute LON: Required attributes missing "
                "(configs, lo_index, config_dict). Ensure landscape was built "
                "from data or these were provided to from_graph."
            )

        if not self.lo_index:
            # Handle case with no local optima
            warnings.warn("No local optima found, LON will be empty.", RuntimeWarning)
            self.lon = nx.DiGraph()  # Create an empty graph
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
                f"LON constructed with {self.lon.number_of_nodes()} nodes "
                f"and {self.lon.number_of_edges()} edges."
            )
        return self.lon

    @property
    def shape(self):
        """Return the shape (n_configs, n_edges) of the landscape graph."""
        self._check_built()
        n_configs = self.n_configs if self.n_configs is not None else 0
        n_edges = self.n_edges if self.n_edges is not None else 0
        return (n_configs, n_edges)

    def __getitem__(self, index):
        """Return the node attributes dictionary for the given index."""
        self._check_built()
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        try:
            # Access node data using graph.nodes dictionary interface
            return self.graph.nodes[index]
        except KeyError:
            raise KeyError(f"Index {index} not found among landscape configurations.")

    def __str__(self):
        """Return a string summary of the landscape."""
        if not self._is_built:
            return f"{self.__class__.__name__} object (uninitialized)"

        n_vars_str = str(self.n_vars) if self.n_vars is not None else "?"
        n_configs_str = str(self.n_configs) if self.n_configs is not None else "?"
        n_edges_str = str(self.n_edges) if self.n_edges is not None else "?"
        n_lo_str = str(self.n_lo) if self.n_lo is not None else "?"

        return (
            f"{self.__class__.__name__} with {n_vars_str} variables, "
            f"{n_configs_str} configurations, {n_edges_str} connections, "
            f"and {n_lo_str} local optima."
        )

    def __repr__(self):
        """Return a concise string representation of the landscape."""
        return self.__str__()

    def __len__(self):
        """Return the number of configurations (nodes) in the landscape."""
        self._check_built()
        return self.n_configs if self.n_configs is not None else 0

    def __iter__(self):
        """Iterate over the configuration indices (nodes) in the landscape."""
        self._check_built()
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        return iter(self.graph.nodes)

    def __contains__(self, item):
        """Check if a configuration index (node) exists in the landscape."""
        self._check_built()
        if self.graph is None:
            raise RuntimeError("Graph is None.")  # Internal check
        # Check if the item is a valid node in the graph
        return item in self.graph.nodes

    def __eq__(self, other):
        """Compare two BaseLandscape instances for equality.

        Equality is currently based on graph structure isomorphism and the
        optimization direction (`maximize`). A more robust comparison might
        also consider node/edge attributes.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if the landscapes are considered equal, False otherwise.
        """
        if not isinstance(other, BaseLandscape):
            return NotImplemented
        if not self._is_built or not other._is_built:
            # Consider unbuilt instances equal only if both are unbuilt
            return self._is_built == other._is_built
        if self.graph is None or other.graph is None:
            # Consider None graphs equal only if both are None
            return self.graph is None and other.graph is None

        # Basic comparison: graph structure and optimization direction
        # Note: nx.is_isomorphic might be too slow for large graphs,
        # and doesn't compare attributes. Using simple equality for now.
        # Consider nx.utils.graphs_equal(self.graph, other.graph) for attribute check
        return self.graph == other.graph and self.maximize == other.maximize

    def __bool__(self):
        """Return True if the landscape is built and has configurations."""
        return self._is_built and bool(self.n_configs and self.n_configs > 0)


# --- Subclasses (Example: SequenceLandscape) ---
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

    def get_data(self, lo_only: bool = False) -> pd.DataFrame:
        """Extracts landscape data, adding a decoded 'sequence' column.

        Overrides `BaseLandscape.get_data`. After retrieving the standard
        attribute DataFrame from the base class, this method adds a new
        column named 'sequence' containing the human-readable sequence string
        decoded from the internal integer representation using the class
        `ALPHABET`.

        Parameters
        ----------
        lo_only : bool, default=False
            If True, returns data only for local optima. If False, returns
            data for all configurations.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing node attributes, including the decoded
            'sequence' column.
        """
        # Get the base DataFrame
        df = super().get_data(lo_only=lo_only)

        if df.empty:
            return df

        self._check_built()  # Ensure landscape is built
        if self.configs is None or self._seq_map_rev is None:
            # Should not happen if built correctly, but check internal state
            if self.verbose:
                warnings.warn(
                    "Cannot add 'sequence' column: internal state "
                    "(configs or _seq_map_rev) is missing.",
                    RuntimeWarning,
                )
            return df

        try:
            # Efficiently map node indices (DataFrame index) to encoded tuples
            config_map = self.configs.to_dict()
            sequences = []
            valid_indices = []  # Keep track of indices we could decode

            # Iterate through the indices present in the DataFrame
            for idx in df.index:
                if idx in config_map:
                    encoded_tuple = config_map[idx]
                    try:
                        # Decode tuple of integers back to sequence string
                        decoded_seq = "".join(
                            self._seq_map_rev[code] for code in encoded_tuple
                        )
                        sequences.append(decoded_seq)
                        valid_indices.append(idx)
                    except KeyError as e:
                        # Handle cases where an integer code might not be in the map
                        warnings.warn(
                            f"Decoding error for index {idx} (KeyError: {e}). "
                            f"Encoded tuple: {encoded_tuple}. Skipping sequence.",
                            RuntimeWarning,
                        )
                    except Exception as e:
                        # Catch other potential decoding errors
                        warnings.warn(
                            f"Generic decoding error for index {idx}: {e}. Skipping sequence.",
                            RuntimeWarning,
                        )
                else:
                    # Handle cases where DataFrame index might not be in configs map
                    # (e.g., if graph was modified after configs were set)
                    warnings.warn(
                        f"Index {idx} from DataFrame not found in self.configs map. Skipping sequence.",
                        RuntimeWarning,
                    )

            # Add the decoded sequences as a new column, aligning by index
            if valid_indices:
                seq_series = pd.Series(sequences, index=valid_indices, name="sequence")
                # Use join for robust index alignment, correctly handles missing indices
                df = df.join(seq_series)
            else:
                if self.verbose:
                    print("No sequences could be decoded to add to the data frame.")

        except Exception as e:
            # Catch unexpected errors during the process
            warnings.warn(
                f"Error adding decoded sequences to data frame: {e}", RuntimeWarning
            )

        return df


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

    def get_data(self, lo_only: bool = False) -> pd.DataFrame:
        """Extracts landscape data, adding a 'bitstring' column.

        Overrides `BaseLandscape.get_data`. After retrieving the standard
        attribute DataFrame, this method adds a new column named 'bitstring'
        containing the binary string representation (e.g., '010') decoded
        from the internal integer tuple representation.

        Parameters
        ----------
        lo_only : bool, default=False
            If True, returns data only for local optima. If False, returns
            data for all configurations.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing node attributes, including the decoded
            'bitstring' column.
        """
        # Get the base DataFrame
        df = super().get_data(lo_only=lo_only)

        if df.empty:
            return df

        self._check_built()  # Ensure landscape is built
        if self.configs is None:
            # Check internal state
            if self.verbose:
                warnings.warn(
                    "Cannot add 'bitstring' column: self.configs is None.",
                    RuntimeWarning,
                )
            return df

        try:
            # Map node indices to encoded tuples
            config_map = self.configs.to_dict()
            bitstrings = []
            valid_indices = []  # Keep track of indices we could decode

            for idx in df.index:
                if idx in config_map:
                    encoded_tuple = config_map[idx]  # e.g., (0, 1, 0)
                    try:
                        # Convert tuple of ints to string '010'
                        bitstring = "".join(map(str, encoded_tuple))
                        bitstrings.append(bitstring)
                        valid_indices.append(idx)
                    except Exception as e:
                        warnings.warn(
                            f"Error converting index {idx} config tuple to bitstring: {e}. Skipping.",
                            RuntimeWarning,
                        )
                else:
                    warnings.warn(
                        f"Index {idx} from DataFrame not found in self.configs map. Skipping bitstring.",
                        RuntimeWarning,
                    )

            # Add the decoded bitstrings as a new column
            if valid_indices:
                bitstring_series = pd.Series(
                    bitstrings, index=valid_indices, name="bitstring"
                )
                df = df.join(bitstring_series)  # Use join for robust index alignment
            else:
                if self.verbose and not df.empty:
                    print("No bitstrings could be decoded to add to the data frame.")

        except Exception as e:
            warnings.warn(
                f"Error adding decoded bitstrings to data frame: {e}", RuntimeWarning
            )

        return df


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
    {'fitness': 4, 'bit_0': 1, 'bit_1': 1, 'out_degree': 0, 'in_degree': 2, 'is_lo': True, 'basin_index': 3, 'size_basin': 4, 'radius_basin': 1, 'dist_go': 0.0, 'bitstring': '11'}

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


# --- Helper Functions (Keep or adapt as needed) ---
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
