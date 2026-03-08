import pandas as pd
import numpy as np
import igraph as ig
import warnings
import time

from typing import Tuple, Dict, List, Union, Optional, Any

from ..lon import get_lon
from ..algorithms import basin, optima, plateaus
from ..analysis import correlation, navigability
from .._data import (
    DNA_ALPHABET,
    RNA_ALPHABET,
    PROTEIN_ALPHABET,
    PreparedData,
    InputHandler,
    BooleanHandler,
    DefaultHandler,
    SequenceHandler,
    filter_data,
    prepare_data,
    clean_data,
    encode_data,
)
from ..utils import (
    add_network_metrics,
    filter_graph,
    infer_graph_properties,
)
from ..distances import mixed_distance, hamming_distance

from .._neighbors import (
    NeighborGenerator,
    BooleanNeighborGenerator,
    DefaultNeighborGenerator,
    SequenceNeighborGenerator,
    build_edges,
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
        print(f"Method {method.__name__} executed in {elapsed_time:.4f} seconds.")
        return result

    return timed




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
    - Basic landscape properties are available via `describe()`.
    - Other methods in the `graphfla.analysis` and `graphfla.plotting` modules can be
      used for advanced analysis and visualization.

    Parameters
    ----------
    type : str, default='default'
        The type of landscape to create. This determines the input-preparation and
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
        `build_from_data` or `build_from_graph`. Fitness difference between
        neighboring nodes is stored in the edge attribute 'delta_fit'.
    configs : pandas.Series or None
        A pandas Series mapping node indices (int) to their corresponding
        configuration representation (often a tuple). This represents the
        genotypes in the landscape. Populated after calling `build_from_data`
        or when loading a graph via `build_from_graph` if the file contains
        configs data.
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
    Class: Landscape
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
    _input_handlers = {
        "boolean": BooleanHandler(),
        "dna": SequenceHandler(DNA_ALPHABET),
        "rna": SequenceHandler(RNA_ALPHABET),
        "protein": SequenceHandler(PROTEIN_ALPHABET),
        "default": DefaultHandler(),
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
        self._configs_array = None
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

        # Strategy object for neighborhood generation (set in build_from_data)
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
        """Compare two Landscape instances for equality.

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
    def register_input_handler(
        cls, data_type: str, handler: InputHandler
    ) -> None:
        """Register a custom input handler for a data type."""
        cls._input_handlers[data_type] = handler

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
        calculate_basins : bool, default=False
            If True, calculates the basins of attraction for each local optimum
            using a greedy hill-climbing algorithm. This identifies which configurations
            lead to which peak. Populates the `size_basin_greedy` and `radius_basin_greedy`
            attributes for each local optimum.
        calculate_paths : bool, default=False
            If True, calculates accessible paths (ancestors) for local optima.
            This can be computationally intensive for large landscapes. Populates the
            `size_basin_accessible` attribute for each local optimum.
        calculate_distance : bool, default=False
            If True, calculates the distance from each node to the global optimum
            and stores it in the `dist_go` vertex attribute.
        calculate_neighbor_fit : bool, default=False
            If True, calculates mean neighbor fitness per node and delta mean neighbor
            fitness along edges; populates `mean_neighbor_fit` and `delta_mean_neighbor_fit`.
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
        None
            The method modifies the instance in place (populates the graph and
            landscape properties). Nothing is returned.

        Raises
        ------
        RuntimeError
            If the `build_from_data` or `build_from_graph` method has already been called
            on this instance. Create a new instance to rebuild.
        ValueError
            If input data `X`, `f`, or `data_types` are invalid (e.g., mismatched
            lengths, empty data, invalid types in `data_types`), or if the
            construction process fails (e.g., resulting graph is empty).
        TypeError
            If input types are incorrect (e.g., `data_types` is not a dict).

        Notes
        -----
        The construction process involves several steps:
        1. Resolve the type-specific preparation and neighbor-generation strategies.
        2. Apply the optional pre-construction fitness filter.
        3. Prepare the raw input based on the landscape type.
        4. Clean missing values and duplicate configurations.
        5. Encode the data and cache configuration metadata.
        6. Construct the core directed landscape graph.
        7. Apply post-construction pruning and remap cached metadata.
        8. Build neutral plateaus and analyze derived landscape properties.
        """
        self._check_not_built()

        self.epsilon = float(epsilon)
        self.verbose = verbose

        if verbose:
            print("Building Landscape from data...")

        handler = self._resolve_strategies()
        processed_data = self._preprocess_data(
            handler=handler,
            X=X,
            f=f,
            data_types=data_types,
            tau=tau,
            filter_mode=filter_mode,
            verbose=verbose,
        )
        neutral_pairs = self._construct_graph(
            processed_data,
            n_edit=n_edit,
            neighborhood_strategy=neighborhood_strategy,
        )
        neutral_pairs = self._postprocess_graph(
            neutral_pairs=neutral_pairs,
            tau=tau,
            filter_mode=filter_mode,
            verbose=verbose,
        )

        plateaus.build_plateaus(self, neutral_pairs)

        self._analyze(
            calculate_distance=calculate_distance,
            calculate_basins=calculate_basins,
            calculate_paths=calculate_paths,
            calculate_neighbor_fit=calculate_neighbor_fit,
        )

        self._finalize_build()

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
        instance.n_configs, instance.n_edges, instance.n_vars = infer_graph_properties(
            instance.graph,
            data_types=instance.data_types,
            configs=instance.configs,
            verbose=instance.verbose,
        )

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
            plateaus.restore_plateaus(instance)

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
            raise RuntimeError("Graph is None despite landscape being built.")

        if lo_only:
            if self.lo_index is None or not self.lo_index:
                warnings.warn(
                    "Local optima not found. Cannot filter for LO.",
                    RuntimeWarning,
                )
                return pd.DataFrame()

            vs = self.graph.vs[self.lo_index]
            data = pd.DataFrame(
                {attr: vs[attr] for attr in self.graph.vs.attributes()},
                index=self.lo_index,
            )

            data.drop(
                columns=[
                    c
                    for c in ("is_lo", "out_degree", "in_degree", "basin_index")
                    if c in data.columns
                ],
                inplace=True,
            )

            if self.has_lon:
                if self.lon is None:
                    raise RuntimeError("LON graph is None despite has_lon=True.")

                lon_specific = {
                    "escape_difficulty",
                    "improve_rate",
                    "accessibility",
                    "in_degree",
                    "out_degree",
                }
                lon_cols = [
                    a for a in self.lon.vs.attributes() if a in lon_specific
                ]
                if lon_cols:
                    lon_df = pd.DataFrame(
                        {f"lon_{a}": self.lon.vs[a] for a in lon_cols},
                        index=self.lon.vs["name"],
                    )
                    data = data.join(lon_df, how="left")

            data.sort_index(inplace=True)

        else:
            data = pd.DataFrame(
                {attr: self.graph.vs[attr] for attr in self.graph.vs.attributes()},
                index=range(self.graph.vcount()),
            )

            cols_to_drop = ["size_basin_greedy", "radius_basin_greedy"]
            if self._path_calculated:
                cols_to_drop.append("size_basin_accessible")
            data.drop(
                columns=[c for c in cols_to_drop if c in data.columns],
                inplace=True,
            )

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

        return data

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

    def _check_not_built(self) -> None:
        """Raise an error if the landscape has already been built."""
        if self._is_built:
            raise RuntimeError(
                "This Landscape instance has already been built. Create a new instance to rebuild."
            )

    def _resolve_strategies(self) -> InputHandler:
        """Resolve and cache the type-specific strategies for data builds."""
        handler = self._input_handlers.get(self.type)
        if handler is None:
            raise ValueError(
                f"No input handler for landscape type: {self.type}"
            )

        neighbor_generator = self._neighbor_generators.get(self.type)
        if neighbor_generator is None:
            raise ValueError(
                f"No neighbor generator available for landscape type: {self.type}"
            )

        self._neighbor_generator = neighbor_generator
        return handler

    def _preprocess_data(
        self,
        *,
        handler: InputHandler,
        X: Any,
        f: Union[pd.Series, list, np.ndarray],
        data_types: Optional[Dict[str, str]],
        tau: Optional[float],
        filter_mode: str,
        verbose: Optional[bool],
    ) -> pd.DataFrame:
        """Run the preprocessing pipeline and cache encoded build metadata."""
        X_filtered, f_filtered = filter_data(
            X, f, self.maximize, tau, filter_mode, verbose
        )

        X_processed, f_processed, self.data_types, self.n_vars = prepare_data(
            handler, X_filtered, f_filtered, data_types=data_types, verbose=verbose
        )

        X_final, f_final = clean_data(
            X_processed,
            f_processed,
            verbose=verbose,
        )

        prepared = encode_data(X_final, f_final, self.data_types, verbose=verbose)
        return self._cache_metadata(prepared)

    def _cache_metadata(self, prepared: PreparedData) -> pd.DataFrame:
        """Persist encoded build metadata on the landscape instance."""
        self.data_types = prepared.data_types
        self.n_vars = prepared.n_vars
        self._configs_array = prepared.configs_array
        self.configs = prepared.configs
        self.config_dict = prepared.config_dict

        processed_data = prepared.data_for_attributes
        self.n_configs = len(processed_data)
        return processed_data

    def _construct_graph(
        self,
        processed_data: pd.DataFrame,
        *,
        n_edit: int,
        neighborhood_strategy: str,
    ) -> List[Tuple[int, int]]:
        """Construct the graph from preprocessed data and return neutral pairs."""
        if self.verbose:
            print("Constructing landscape graph...")

        edges, delta_fits, neutral_pairs = self._build_edges(
            processed_data, n_edit=n_edit, strategy=neighborhood_strategy
        )
        self.graph = self._build_graph(processed_data, edges, delta_fits)
        return neutral_pairs

    def _postprocess_graph(
        self,
        *,
        neutral_pairs: List[Tuple[int, int]],
        tau: Optional[float],
        filter_mode: str,
        verbose: Optional[bool],
    ) -> List[Tuple[int, int]]:
        """Apply graph pruning and remap cached metadata when vertices are removed."""
        self.graph, self.n_configs, self.n_edges, kept_indices = filter_graph(
            self.graph, self.maximize, tau, filter_mode, verbose
        )
        return self._remap_metadata(kept_indices, neutral_pairs)

    def _remap_metadata(
        self,
        kept_indices: Optional[List[int]],
        neutral_pairs: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Remap configs and neutral pairs after graph filtering changes indices."""
        if kept_indices is None:
            return neutral_pairs

        old_to_new = {old: new for new, old in enumerate(kept_indices)}

        if self.configs is not None:
            self.configs = self.configs.take(kept_indices)
            self.configs.index = range(len(kept_indices))
        if self._configs_array is not None:
            self._configs_array = self._configs_array[kept_indices]

        if not neutral_pairs:
            return neutral_pairs

        return [
            (old_to_new[u], old_to_new[v])
            for u, v in neutral_pairs
            if u in old_to_new and v in old_to_new
        ]

    def _finalize_build(self) -> None:
        """Mark the instance as built and emit the standard completion output."""
        self._is_built = True
        if self.verbose:
            print("Landscape built successfully.\n")
            self.describe()

    def _check_built(self) -> None:
        """Raise an error if the landscape hasn't been built yet."""
        if not self._is_built:
            raise RuntimeError(
                "Landscape has not been built yet. Call build_from_data() or "
                "build_from_graph() first."
            )

    @timeit
    def _build_edges(self, data, n_edit, strategy="auto"):
        """Build improving edges and neutral pairs for the current dataset."""
        if self._neighbor_generator is None:
            raise RuntimeError("Neighbor generator not set before build.")

        result = build_edges(
            configs=self.configs,
            config_dict=self.config_dict,
            data=data,
            n_configs=self.n_configs,
            n_vars=self.n_vars,
            n_edit=n_edit,
            strategy=strategy,
            epsilon=float(self.epsilon),
            maximize=self.maximize,
            verbose=self.verbose,
            neighbor_generator=self._neighbor_generator.generate,
            configs_array=self._configs_array,
        )
        return result.edges, result.delta_fits, result.neutral_pairs

    @timeit
    def _build_graph(self, data, edges, delta_fits):
        """Build the igraph representation from nodes and improving edges."""
        if self.verbose:
            print(" - Constructing graph object...")

        if self.verbose:
            print(" - Adding node attributes (fitness, etc.)...")
        graph = ig.Graph(
            n=len(data),
            edges=edges,
            directed=True,
            vertex_attrs={
                str(column): data[column].to_numpy(copy=False)
                for column in data.columns
            },
            edge_attrs={"delta_fit": delta_fits} if delta_fits else None,
        )

        self.n_edges = graph.ecount()  # Update edge count based on final graph

        return graph

    def _get_default_distance_metric(self):
        """Get the appropriate distance metric based on landscape type."""
        if self.type in ["boolean", "dna", "rna", "protein"]:
            return hamming_distance
        else:
            return mixed_distance

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
        return optima.determine_local_optima(self)

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
        return basin.determine_basin_of_attraction(self)

    @timeit
    def determine_accessible_paths(self):
        """Determines the size of basins based on accessible paths (ancestors).

        Identifies all nodes from which a local optimum can be reached via any
        fitness-increasing path. When plateaus are active, ancestors are computed
        for each plateau-level LO by taking the union of ancestors across all
        plateau members,         assigned to the canonical representative (min index).

        Notes
        -----
        This method is computationally intensive and might be slow for large
        landscapes.
        """
        return navigability.determine_accessible_paths(self)

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
        .. [Wagner 2023] Wagner, A. The role of evolvability in the evolution of
           complex traits. Nat Rev Genet 24, 1-16 (2023).
           https://doi.org/10.1038/s41576-023-00559-0

        Returns
        -------
        Landscape
            The landscape instance (self) with the new attributes added, for
            method chaining.

        Raises
        ------
        RuntimeError
            If the landscape has not been built yet.
        """
        return correlation.determine_neighbor_fitness(self)

    @timeit
    def determine_global_optimum(self):
        """Identifies the global optimum node in the landscape graph using igraph."""
        return navigability.determine_global_optimum(self)

    @timeit
    def determine_dist_to_go(self, distance=None):
        """Calculates the distance from each node to the global optimum."""
        return navigability.determine_dist_to_go(self, distance=distance)

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

    @timeit
    def _analyze(
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
            if self.verbose:
                print(" - Calculating network metrics (degrees)...")
            self.graph = add_network_metrics(self.graph, weight=weight_key)

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

