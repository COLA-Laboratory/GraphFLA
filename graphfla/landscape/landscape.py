from __future__ import annotations

import ast
import inspect
import pandas as pd
import numpy as np
import igraph as ig
import warnings

from typing import Tuple, Dict, List, Union, Optional, Any

from ..lon import get_lon
from . import _basin, _optima, _plateaus, _compute
from .._data import (
    DNA_ALPHABET,
    RNA_ALPHABET,
    PROTEIN_ALPHABET,
    PreparedData,
    InputHandler,
    BooleanHandler,
    DefaultHandler,
    OrdinalHandler,
    SequenceHandler,
    filter_data,
    prepare_data,
    clean_data,
    encode_data,
    configs_series_from_array,
)
from ..utils import (
    filter_graph,
    remove_isolated_nodes,
    infer_graph_properties,
    timeit,
)
from ..distances import mixed_distance, hamming_distance
from ..exceptions import InvalidParameterError, NotBuiltError

from .._neighbors import (
    NeighborGenerator,
    BooleanNeighborGenerator,
    DefaultNeighborGenerator,
    OrdinalNeighborGenerator,
    SequenceNeighborGenerator,
    build_edges,
)
from ._io import _IOMixin


class Landscape(_IOMixin):
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
        The number of distinct local optima (plateau-aware): each neutral
        plateau-LO counts once, plus every single-point LO. This is the
        "number of local optima". Populated after graph analysis.
    n_lo_members : int or None
        The total number of local-optimum *member nodes* (every member of
        every plateau-LO plus single-point LOs); equals ``len(lo_index)``.
        For a landscape with no neutral plateaus this equals ``n_lo``.
        Populated after graph analysis.
    lo_index : list[int] or None
        A sorted list of all node indices that are local optima
        (plateau-aware); its length is ``n_lo_members``. Populated after
        graph analysis.
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
    plateaus : dict or None
        Mapping from 0-based plateau ID to the list of member node indices.
        Only multi-member plateaus are stored; singleton nodes have
        ``plateau_id = -1``.  Populated when ``epsilon > 0`` and neutral
        pairs exist.
    n_plateau : int or None
        The number of neutral plateaus (multi-member connected components of
        neutral neighbors). Populated when ``epsilon > 0`` and neutral pairs
        exist; 0 when no plateaus; None before construction.
    plateau_lo_index : list[int] or None
        Plateau IDs (0-based) whose plateaus are local optima. Does not
        include single-point LOs.
    n_plateau_lo : int or None
        ``len(plateau_lo_index)``.
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
    >>> X_data = pd.DataFrame({'var_0': [0, 0, 1, 1], 'var_1': [0, 1, 0, 1]})
    >>> f_data = pd.Series([1.0, 2.0, 3.0, 2.5])

    >>> landscape = BooleanLandscape().build_from_data(X_data, f_data, verbose=False)
    >>> landscape.describe()["n_lo"]
    1
    >>> repr(landscape)
    'BooleanLandscape(maximize=True)'
    >>> print(f"Number of configurations: {landscape.n_configs}")
    Number of configurations: 4
    >>> print(f"Global optimum fitness: {landscape.go['fitness']}")  
    Global optimum fitness: 3.0
    """

    # Class-level registries for strategies
    _input_handlers = {
        "boolean": BooleanHandler(),
        "ordinal": OrdinalHandler(),
        "dna": SequenceHandler(DNA_ALPHABET),
        "rna": SequenceHandler(RNA_ALPHABET),
        "protein": SequenceHandler(PROTEIN_ALPHABET),
        "default": DefaultHandler(),
    }

    _neighbor_generators = {
        "boolean": BooleanNeighborGenerator(),
        "ordinal": OrdinalNeighborGenerator(),
        "dna": SequenceNeighborGenerator(len(DNA_ALPHABET)),
        "rna": SequenceNeighborGenerator(len(RNA_ALPHABET)),
        "protein": SequenceNeighborGenerator(len(PROTEIN_ALPHABET)),
        "default": DefaultNeighborGenerator(),
    }

    #: Default ``neighborhood_strategy`` for this class when the caller passes
    #: ``None`` to :meth:`build_from_data`. Subclasses override this instead of
    #: re-declaring the whole ``build_from_data`` signature (e.g. ordinal
    #: landscapes set ``"active"`` to enforce ±1 Manhattan neighbours).
    _default_neighborhood_strategy: str = "auto"

    def __init__(
        self,
        kind: str = "default",
        maximize: bool = True,
        input_handler: Optional[InputHandler] = None,
        neighbor_generator: Optional[NeighborGenerator] = None,
        strategy_key: Optional[str] = None,
    ):
        # Core attributes
        self.graph = None
        # ``configs`` tuple Series is built lazily from ``_configs_array`` (the
        # numeric source of truth that construction consumes), so builds that
        # never read it skip the cost.
        self._configs = None
        self._configs_index = None
        self._configs_array = None
        self.config_dict = None
        self.data_types = None
        # Working counts during construction (before self.graph exists); the
        # public n_configs / n_edges properties read the graph once built.
        self._n_configs = None
        self._n_edges = None
        self.n_vars = None
        self.n_lo = None
        self.n_lo_members = None
        self.lo_index = None
        self.go_index = None
        self.go = None
        self.lon = None
        self.has_lon = False
        # ``kind`` is the public semantic identity ('boolean'/'ordinal'/'dna'/
        # 'rna'/'protein'/'default') that downstream code branches on (distance
        # metric, Walsh-Hadamard encoding). ``_strategy_key`` is the internal
        # key into the handler/generator registries; it equals ``kind`` except
        # for sequence subclasses, which share a constant 'sequence' key (the
        # per-instance registry isolates their alphabet-specific handlers).
        self.kind = kind
        self._strategy_key = strategy_key if strategy_key is not None else kind

        # Per-instance copies of the class-level registries so a custom
        # handler/generator never mutates global state or leaks across instances.
        self._input_handlers = dict(Landscape._input_handlers)
        self._neighbor_generators = dict(Landscape._neighbor_generators)
        if input_handler is not None:
            self._input_handlers[self._strategy_key] = input_handler
        if neighbor_generator is not None:
            self._neighbor_generators[self._strategy_key] = neighbor_generator
        if (
            self._strategy_key not in self._input_handlers
            or self._strategy_key not in self._neighbor_generators
        ):
            raise InvalidParameterError(
                f"Unknown landscape kind {kind!r}. Registered kinds: "
                f"{sorted(self._input_handlers)}. Register a custom kind by passing "
                f"input_handler= and neighbor_generator= to the constructor."
            )

        # Landscape construction parameters
        self.maximize = maximize
        self.epsilon = 0.0
        self.verbose = False

        # Plateau / neutrality layer attributes
        self._has_plateaus = False
        self._node_to_plateau = None    # np.ndarray int32: node_idx → plateau_id (-1 = singleton)
        self.plateaus = None            # dict: plateau_id → list[int] of member nodes (0-based IDs)
        self._neutral_neighbors = None  # dict: node_idx → list[int] of neutral neighbors
        self.n_plateau = None           # number of neutral plateaus (multi-member components)
        self.n_plateau_lo = None        # number of plateau-LOs (multi-member only)
        self.plateau_lo_index = None    # list of plateau IDs that are LOs (multi-member only)
        self._peak_index = None         # one representative node per peak (for LON)

        # Build status flags
        self._is_built = False
        self._path_calculated = False
        self._basin_calculated = False
        self._distance_calculated = False
        self._neighbor_fit_calculated = False
        self._pagerank_calculated = False

        # Strategy object for neighborhood generation (set in build_from_data)
        self._neighbor_generator = None

    @property
    def n_configs(self):
        """Number of configurations (nodes). Derived from the graph once built
        (single source of truth); falls back to the working count during
        construction, before ``self.graph`` exists."""
        if self.graph is not None:
            return self.graph.vcount()
        return self._n_configs

    @property
    def n_edges(self):
        """Number of directed edges. Derived from the graph once built (single
        source of truth); falls back to the working count during construction."""
        if self.graph is not None:
            return self.graph.ecount()
        return self._n_edges

    @property
    def shape(self):
        """Return the shape ``(n_configs, n_edges)`` of the landscape graph."""
        self._check_built()
        if self.graph is None:
            return (0, 0)
        return (self.n_configs, self.n_edges)

    @property
    def configs(self) -> Optional[pd.Series]:
        """Per-node configuration tuple ``Series`` (built lazily, then cached).

        The numeric ``_configs_array`` is the source of truth and is what graph
        construction consumes; this tuple ``Series`` is a downstream artifact
        used only by analyses that need per-node configuration tuples.  It is
        materialised from ``_configs_array`` on first access -- builds that never
        read ``configs`` (the common construction-only path) skip the cost
        entirely.  Returns ``None`` only when neither the cached ``Series`` nor a
        numeric array is available (e.g. an unbuilt landscape, or a graph load
        from which configurations could not be reconstructed).

        Note: a ``landscape.configs is None`` check materialises the ``Series``
        if ``_configs_array`` is present; this matches the previous eager
        attribute (which was always present) and is intended -- every caller that
        performs such a check goes on to use the configurations.
        """
        if self._configs is not None:
            return self._configs
        if self._configs_array is not None:
            self._configs = configs_series_from_array(
                self._configs_array, self._configs_index
            )
            return self._configs
        return None

    @configs.setter
    def configs(self, value: Optional[pd.Series]) -> None:
        """Set (or clear) the cached configuration tuple ``Series`` directly."""
        self._configs = value

    # ------------------------------------------------------------------
    # Lazy, cached analysis properties.
    #
    # Computed on first access and cached via the matching ``_*_calculated``
    # guard. Canonical (and only) way to get basins / distances / neighbour
    # fitness / accessible-path sizes.
    # ------------------------------------------------------------------
    @property
    def basins(self) -> pd.Series:
        """Per-node greedy basin size (``size_basin_greedy``), computed lazily."""
        self._check_built()
        if not self._basin_calculated:
            _basin.determine_basin_of_attraction(self)
        return pd.Series(self.graph.vs["size_basin_greedy"], name="size_basin_greedy")

    @property
    def accessible_paths(self) -> pd.Series:
        """Per-node accessible-basin size (``size_basin_accessible``), computed lazily."""
        self._check_built()
        if not self._path_calculated:
            _compute.determine_accessible_paths(self)
        return pd.Series(
            self.graph.vs["size_basin_accessible"], name="size_basin_accessible"
        )

    @property
    def dist_to_go(self) -> pd.Series:
        """Per-node configuration distance to the nearest global optimum
        (``dist_go``), computed lazily."""
        self._check_built()
        if not self._distance_calculated:
            _compute.determine_dist_to_go(
                self, distance=self._get_default_distance_metric()
            )
        return pd.Series(self.graph.vs["dist_go"], name="dist_go")

    @property
    def neighbor_fitness(self) -> pd.Series:
        """Per-node mean neighbour fitness (``mean_neighbor_fit``), computed lazily."""
        self._check_built()
        if not self._neighbor_fit_calculated:
            _compute.determine_neighbor_fitness(self)
        return pd.Series(self.graph.vs["mean_neighbor_fit"], name="mean_neighbor_fit")

    @property
    def pagerank(self) -> pd.Series:
        """Per-node PageRank centrality, computed lazily.

        PageRank is not used by any landscape metric; it is only an optional,
        descriptive node attribute. To keep it off the construction critical
        path (where it was the single dominant cost) it is computed on first
        access here -- producing values identical to the eager computation
        (weighted by ``delta_fit`` when present, ``directed=True``)."""
        self._check_built()
        self._ensure_pagerank()
        return pd.Series(self.graph.vs["pagerank"], name="pagerank")

    def _ensure_pagerank(self) -> None:
        """Materialise the ``pagerank`` node attribute if it is not present.

        Idempotent: respects an externally supplied ``pagerank`` attribute and
        only computes once. Used by the lazy ``pagerank`` property and by any
        consumer (e.g. ``get_data``) that must expose the full attribute set."""
        if self._pagerank_calculated or "pagerank" in self.graph.vs.attributes():
            self._pagerank_calculated = True
            return
        weights = (
            "delta_fit" if "delta_fit" in self.graph.es.attributes() else None
        )
        self.graph.vs["pagerank"] = self.graph.pagerank(
            weights=weights, directed=True
        )
        self._pagerank_calculated = True

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
        """Human-readable one-line summary (sizes when built)."""
        head = f"{self.__class__.__name__}(kind={self.kind!r})"
        if not self._is_built:
            return f"{head} — not built"
        n_vars_str = str(self.n_vars) if self.n_vars is not None else "?"
        n_lo_str = str(self.n_lo) if self.n_lo is not None else "?"
        return (
            f"{head}: {n_vars_str} variables, {self.n_configs} configurations, "
            f"{self.n_edges} edges, {n_lo_str} local optima"
        )

    def __repr__(self):
        """Params-forward representation (scikit-learn style)."""
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

    @classmethod
    def _get_param_names(cls):
        """Constructor parameter names for ``get_params`` (scikit-learn style).

        Introspects this class's ``__init__`` and drops ``self`` plus the
        internal strategy-wiring parameters (``input_handler`` /
        ``neighbor_generator`` / ``strategy_key``), which are implementation
        details rather than reportable configuration.
        """
        sig = inspect.signature(cls.__init__)
        names = []
        for name, p in sig.parameters.items():
            if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if name in ("input_handler", "neighbor_generator", "strategy_key"):
                continue
            names.append(name)
        return sorted(names)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the constructor parameters as a dict (scikit-learn style).

        Enables introspection and ``sklearn.base.clone``-style reconstruction
        (``type(ls)(**ls.get_params())`` yields a fresh, unbuilt landscape).
        """
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params) -> "Landscape":
        """Set constructor parameters by name (scikit-learn style).

        Intended for configuring an unbuilt landscape; unknown names raise
        :class:`InvalidParameterError`.
        """
        valid = set(self._get_param_names())
        for key, value in params.items():
            if key not in valid:
                raise InvalidParameterError(
                    f"Invalid parameter {key!r} for {type(self).__name__}; "
                    f"valid parameters are {sorted(valid)}."
                )
            setattr(self, key, value)
        return self

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

    def __bool__(self):
        """Return True if the landscape is built and has configurations."""
        return self._is_built and self.graph is not None and self.graph.vcount() > 0

    def register_input_handler(
        self, data_type: str, handler: InputHandler
    ) -> None:
        """Register a custom input handler for a data type on this instance."""
        self._input_handlers[data_type] = handler

    def register_neighbor_generator(
        self, data_type: str, generator: NeighborGenerator
    ) -> None:
        """Register a custom neighbor generator for a data type on this instance."""
        self._neighbor_generators[data_type] = generator

    @timeit
    def build_from_data(
        self,
        X: Any,
        f: Union[pd.Series, list, np.ndarray],
        data_types: Optional[Dict[str, str]] = None,
        epsilon: float = 0,
        tau: Optional[float] = None,
        filter_mode: str = "any",
        n_edit: int = 1,
        neighborhood_strategy: Optional[str] = None,
        verbose: Optional[bool] = True,
    ) -> "Landscape":
        """Construct the landscape graph and properties from configuration data.

        This method takes genotype-phenotype data (configurations `X` and their
        corresponding fitness values `f`) and builds the underlying graph
        structure of the fitness landscape. Nodes represent configurations, and
        edges connect neighbors based on the specified edit distance (`n_edit`).
        It then determines the basic landscape properties (local optima and the
        global optimum). More expensive analyses — basins of attraction,
        accessible paths, distance-to-optimum and neighbour fitness — are computed
        lazily on first access via the ``.basins`` / ``.accessible_paths`` /
        ``.dist_to_go`` / ``.neighbor_fitness`` properties.

        This method populates the core attributes of the `Landscape` instance.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray or a list of strings 
            The configuration data, where each row represents a genotype or
            configuration, and columns represent variables or sites.
        f : pandas.Series, list, or numpy.ndarray
            The fitness values corresponding to each configuration in `X`. Must
            have the same length as `X`.
        data_types : dict[str, str], optional
            A dictionary specifying the type of each variable (column in `X` if
            DataFrame, or inferred column index if ndarray). Keys must match
            column names/indices, and values must be one of 'boolean',
            'categorical', or 'ordinal'. This information is crucial for
            determining neighborhood relationships and calculating distances.
            Optional: the typed landscape subclasses (``BooleanLandscape``,
            ``OrdinalLandscape``, ``DNALandscape``/``RNALandscape``/
            ``ProteinLandscape``) auto-detect it. Supply it explicitly for a
            generic ``Landscape(kind="default")`` with heterogeneous (mixed)
            columns.
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
            The edit distance defining the neighborhood. For ``pairwise`` and
            ``broadcast`` strategies, an undirected neighbor pair is kept when
            Hamming distance is positive and ``<= n_edit``. The ``active``
            strategy uses type-specific generators, which only support
            ``n_edit=1`` for boolean, sequence, and default landscapes (use
            ``pairwise`` or ``broadcast`` for multi-edit Hamming graphs).
        neighborhood_strategy : str, optional
            Strategy for identifying neighboring configurations. When ``None``
            (the default), the class default is used
            (:attr:`_default_neighborhood_strategy` — ``'auto'`` for most
            landscapes, ``'active'`` for ordinal). On ordinal (or mixed
            landscapes containing an ordinal variable), ``'auto'`` resolves to
            ``'active'`` and the Hamming-based ``'pairwise'``/``'broadcast'``
            strategies emit a warning, because they ignore the ±1-step ordinal
            adjacency. Options:

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
        Landscape
            The populated instance itself (``self``), to support method
            chaining (e.g. ``BooleanLandscape().build_from_data(X, f)``).

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
        4. Warn on missing values (no automatic row drops); drop duplicate
           configurations; encode step errors if NaNs remain.
        5. Encode the data and cache configuration metadata.
        6. Construct the core directed landscape graph.
        7. Apply post-construction pruning and remap cached metadata.
        8. Build neutral plateaus and analyze derived landscape properties.
        """
        self._check_not_built()

        # Validate the build parameters up front with clear messages, before any
        # state is mutated, instead of failing deep in the pipeline.
        if filter_mode not in ("any", "both"):
            raise InvalidParameterError(
                f"filter_mode must be 'any' or 'both', got {filter_mode!r}."
            )
        if neighborhood_strategy is not None and neighborhood_strategy not in (
            "auto",
            "active",
            "pairwise",
            "broadcast",
        ):
            raise InvalidParameterError(
                "neighborhood_strategy must be one of None, 'auto', 'active', "
                f"'pairwise', 'broadcast', got {neighborhood_strategy!r}."
            )
        if epsilon < 0:
            raise InvalidParameterError(f"epsilon must be >= 0, got {epsilon!r}.")
        if n_edit < 1:
            raise InvalidParameterError(f"n_edit must be >= 1, got {n_edit!r}.")

        self.epsilon = float(epsilon)
        # Coerce to a real bool so the documented bool attribute can't be poisoned
        # by an explicit verbose=None.
        self.verbose = bool(verbose)

        # Fall back to the per-class default (e.g. "active" for ordinal).
        if neighborhood_strategy is None:
            neighborhood_strategy = self._default_neighborhood_strategy

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

        # Ordinal/mixed landscapes need ±1-step (Manhattan-1) neighbours, not
        # Hamming; data_types is only known after preprocessing, so adjust here.
        has_ordinal = (self.kind == "ordinal") or bool(
            self.data_types and "ordinal" in self.data_types.values()
        )
        if has_ordinal:
            if neighborhood_strategy in ("pairwise", "broadcast"):
                warnings.warn(
                    f"neighborhood_strategy={neighborhood_strategy!r} uses Hamming "
                    "distance, which ignores the ±1-step (Manhattan-1) semantics of "
                    "ordinal variables and treats any single-position change as "
                    "adjacent. Use 'active' for a correct ordinal neighbourhood.",
                    UserWarning,
                    stacklevel=3,  # skip the @timeit wrapper to point at the caller
                )
            elif neighborhood_strategy == "auto":
                neighborhood_strategy = "active"

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

        _plateaus.build_plateaus(self, neutral_pairs)

        self._analyze()

        self._finalize_build()

        return self

    def get_data(
        self, lo_only: bool = False, include_pagerank: bool = False
    ) -> pd.DataFrame:
        """Extracts landscape data as a pandas DataFrame.

        Returns a DataFrame where rows correspond to configurations (nodes)
        and columns correspond to their attributes (e.g., fitness, degree,
        basin information, original features). Feature columns appear in the
        original input order.

        Parameters
        ----------
        lo_only : bool, default=False
            If True, returns data only for the configurations identified as
            local optima. If the Local Optima Network (LON) has been computed
            (see `get_lon`), data from the LON graph is returned. Otherwise,
            it returns data from the main graph filtered for local optima nodes.
            If False, returns data for all configurations in the main graph.

        include_pagerank : bool, default=False
            If True, compute (if needed) and include a ``pagerank`` column.
            PageRank is otherwise omitted -- it is an optional, comparatively
            expensive centrality that most callers do not need, so ``get_data``
            no longer triggers it as a hidden side effect.

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

        # PageRank is lazy and comparatively expensive; only materialise it when
        # the caller explicitly asks for it.
        if include_pagerank:
            self._ensure_pagerank()

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

                    # Map every plateau-LO member (not just the min-index
                    # representative in lon.vs["name"]) to its peak so all
                    # members receive the plateau's LON attributes.
                    node_to_peak: dict = {}
                    if self._has_plateaus and self._node_to_plateau is not None:
                        for pid in (self.plateau_lo_index or []):
                            peak = min(self.plateaus[pid])
                            for member in self.plateaus[pid]:
                                node_to_peak[member] = peak
                    # Singleton LOs are their own peaks
                    for node in self.lo_index:
                        if node not in node_to_peak:
                            node_to_peak[node] = node

                    # Reindex lon_df to align with every row in data
                    peaks = [node_to_peak.get(n, n) for n in data.index]
                    expanded = lon_df.reindex(peaks)
                    expanded.index = data.index
                    for col in expanded.columns:
                        data[col] = expanded[col].values

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

        if not include_pagerank and "pagerank" in data.columns:
            data.drop(columns="pagerank", inplace=True)

        if self.data_types is not None and self.n_vars is not None:
            keys = list(self.data_types.keys())
            # Feature columns built from data already carry their input names in
            # original input order; only fall back to positionally labelling the
            # first n_vars columns when those names are absent (e.g. a graph
            # loaded via build_from_graph with generic vertex attributes).
            if not all(k in data.columns for k in keys):
                data.rename(
                    columns={
                        old: new
                        for old, new in zip(data.columns[: self.n_vars], keys)
                    },
                    inplace=True,
                )

        return data

    def get_lon(
        self,
        mlon: bool = True,
        min_edge_freq: int = 3,
        trim: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> ig.Graph:
        """Constructs and returns the Local Optima Network (LON).

        The LON is a coarse-grained representation of the fitness landscape
        where nodes are the local optima of the original landscape, and edges
        represent the possibility of transitions between their basins of
        attraction, typically weighted by the fitness difference or distance
        between the optima. This method requires the landscape graph to be
        built and local optima to be identified.

        The landscape's own graph, configurations, optima and ``config_dict``
        are supplied automatically; the parameters below control the LON itself
        and are forwarded to :func:`graphfla.lon.get_lon`.

        Parameters
        ----------
        mlon : bool, default=True
            If True, also build the monotonic LON (edges restricted to
            non-worsening transitions).
        min_edge_freq : int, default=3
            Keep a LON edge only when the number of basin transitions between
            two optima is strictly greater than this threshold.
        trim : int, optional
            If given, keep only the ``trim`` strongest outgoing edges per node.
        verbose : bool, optional
            Verbosity override; defaults to the landscape's own ``verbose``.

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

        if self.configs is None or self._peak_index is None or self.config_dict is None:
            raise RuntimeError(
                "Cannot compute LON: Required attributes missing "
                "(configs, _peak_index, config_dict). Ensure landscape was built "
                "from data or these were provided to build_from_graph."
            )

        if not self._peak_index:
            warnings.warn("No local optima found, LON will be empty.", RuntimeWarning)
            self.lon = ig.Graph()
            self.has_lon = True
            return self.lon

        # The LON is built from basin membership; compute basins lazily if needed.
        if not self._basin_calculated:
            self.basins

        if self.verbose:
            print("Constructing Local Optima Network (LON)...")

        self.lon = get_lon(
            graph=self.graph,
            configs=self.configs,
            lo_index=self._peak_index,
            config_dict=self.config_dict,
            maximize=self.maximize,
            mlon=mlon,
            min_edge_freq=min_edge_freq,
            trim=trim,
            verbose=self.verbose if verbose is None else verbose,
        )
        self.has_lon = True

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

    @timeit
    def _resolve_strategies(self) -> InputHandler:
        """Resolve and cache the type-specific strategies for data builds."""
        handler = self._input_handlers.get(self._strategy_key)
        if handler is None:
            raise InvalidParameterError(
                f"No input handler for landscape kind: {self.kind}"
            )

        neighbor_generator = self._neighbor_generators.get(self._strategy_key)
        if neighbor_generator is None:
            raise InvalidParameterError(
                f"No neighbor generator available for landscape kind: {self.kind}"
            )

        self._neighbor_generator = neighbor_generator
        return handler

    @timeit
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
        # Store the numeric matrix (source of truth) and index; leave the tuple
        # Series cache empty so ``configs`` builds it lazily only if read.
        self._configs_array = prepared.configs_array
        self._configs_index = prepared.configs_index
        self._configs = None
        self.config_dict = prepared.config_dict

        processed_data = prepared.data_for_attributes
        self._n_configs = len(processed_data)
        return processed_data

    @timeit
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

    @timeit
    def _postprocess_graph(
        self,
        *,
        neutral_pairs: List[Tuple[int, int]],
        tau: Optional[float],
        filter_mode: str,
        verbose: Optional[bool],
    ) -> List[Tuple[int, int]]:
        """Apply graph pruning and remap cached metadata when vertices are removed."""
        self.graph, self._n_configs, self._n_edges, kept_indices = filter_graph(
            self.graph, self.maximize, tau, filter_mode, verbose
        )

        # Protect plateau-interior nodes (linked only by neutral/tied edges)
        # from isolation pruning, which runs before the plateau layer is built
        # and would otherwise drop them as "isolated".
        protected = None
        if neutral_pairs:
            if kept_indices is not None:
                tau_map = {old: new for new, old in enumerate(kept_indices)}
                protected = {
                    tau_map[n]
                    for pair in neutral_pairs
                    for n in pair
                    if n in tau_map
                }
            else:
                protected = {n for pair in neutral_pairs for n in pair}

        iso_result = remove_isolated_nodes(
            self.graph, self.verbose, protected=protected
        )
        if iso_result is not None:
            self.graph, self._n_configs, self._n_edges, iso_kept = iso_result
            if kept_indices is not None:
                kept_indices = [kept_indices[i] for i in iso_kept]
            else:
                kept_indices = iso_kept

        return self._remap_metadata(kept_indices, neutral_pairs)

    def _remap_metadata(
        self,
        kept_indices: Optional[List[int]],
        neutral_pairs: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Remap configs and neutral pairs after graph filtering changes indices."""
        if kept_indices is None:
            return neutral_pairs

        kept_arr = np.asarray(kept_indices, dtype=np.int64)
        n_kept = kept_arr.size

        # Remap the numeric matrix and reset the index to a contiguous range.
        # Don't touch the tuple Series unless already built (that would force an
        # unnecessary materialisation); when empty, ``configs`` rebuilds it
        # lazily from the remapped array with identical tuples/order.
        if self._configs_array is not None:
            self._configs_array = self._configs_array[kept_arr]
        self._configs_index = range(n_kept)
        if self._configs is not None:
            remapped = self._configs.take(kept_indices)
            remapped.index = range(n_kept)
            self._configs = remapped

        if not neutral_pairs:
            return neutral_pairs

        # Vectorised old->new remap of neutral pairs (the Python-dict +
        # comprehension was the dominant cost on large sparse graphs). ``inv``
        # maps each surviving old index to its new contiguous index, -1 for
        # dropped; the mask drops pairs touching a removed vertex.
        pairs = np.asarray(neutral_pairs, dtype=np.int64)
        # ``inv`` must span every old index used to index it (kept vertices and
        # pair endpoints); use ``.max()`` so sizing holds even if unsorted.
        n_inv = int(max(int(kept_arr.max()), int(pairs.max()))) + 1
        inv = np.full(n_inv, -1, dtype=np.int64)
        inv[kept_arr] = np.arange(n_kept, dtype=np.int64)

        u = inv[pairs[:, 0]]
        v = inv[pairs[:, 1]]
        keep = (u >= 0) & (v >= 0)
        return list(zip(u[keep].tolist(), v[keep].tolist()))

    @timeit
    def _finalize_build(self) -> None:
        """Mark the instance as built and emit the standard completion output."""
        self._is_built = True
        if self.verbose:
            print("Landscape built successfully.\n")
            self.describe()

    def _check_built(self) -> None:
        """Raise :class:`NotBuiltError` if the landscape hasn't been built yet."""
        if not self._is_built:
            raise NotBuiltError(
                "Landscape has not been built yet. Call build_from_data() or "
                "build_from_graph() first."
            )

    def _build_edges(self, data, n_edit, strategy="auto"):
        """Build improving edges and neutral pairs for the current dataset."""
        if self._neighbor_generator is None:
            raise RuntimeError("Neighbor generator not set before build.")

        # Pass the cached ``self._configs`` (empty during a fresh build), not the
        # ``configs`` property, so fast strategies use ``configs_array`` and the
        # tuple Series is never materialised on the construction path;
        # ``build_edges`` builds tuples on demand only for the generic fallback.
        result = build_edges(
            configs=self._configs,
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
        """Build the igraph representation from nodes and improving edges.

        ``edges`` is the directed ``(source, target)`` edge list and
        ``delta_fits`` the aligned ``|Δfitness|`` weights, as produced by
        :func:`graphfla._neighbors.build_edges`. Depending on the neighbourhood
        strategy these are either numpy arrays (``active``: an ``(E, 2)`` int64
        edge array + 1-D float64 weights) or Python lists of tuples/floats
        (``pairwise`` / ``broadcast``).

        Edge *list* ingest uses the ``(E, 2)`` int64 ndarray directly (igraph
        0.11's fastest path; an array->list conversion is a net loss). The
        per-edge ``delta_fit`` *attribute*, however, ingests ~2x faster when
        igraph reads it through the buffer protocol than when it iterates a
        float64 ndarray element-by-element: the ndarray path boxes each element
        as a Python ``np.float64`` object, which a ``memoryview`` over the same
        (zero-copy) buffer avoids. So a contiguous 1-D ``delta_fits`` array is
        wrapped in a ``memoryview`` here (no data copy, unlike ``.tolist()``).
        igraph then stores the values as plain Python ``float`` objects --
        matching what the ``pairwise``/``broadcast`` producers already emit, and
        identical in value (``float(x) == np.float64(x)``). Edge order is
        preserved, keeping ``delta_fits[i]`` aligned with edge ``i``.
        """
        if self.verbose:
            print(" - Constructing graph object...")

        if self.verbose:
            print(" - Adding node attributes (fitness, etc.)...")

        n_edges = len(edges)
        if n_edges:
            # igraph reads the per-edge float attribute faster from a buffer than
            # from a float64 ndarray (see docstring); zero-copy wrap, contiguous
            # 1-D only, anything else passed through unchanged.
            delta_attr = delta_fits
            if (
                isinstance(delta_fits, np.ndarray)
                and delta_fits.ndim == 1
                and delta_fits.flags["C_CONTIGUOUS"]
            ):
                delta_attr = memoryview(delta_fits)
            edge_attrs = {"delta_fit": delta_attr}
        else:
            edge_attrs = {}

        graph = ig.Graph(
            n=len(data),
            edges=edges if n_edges else None,
            directed=True,
            vertex_attrs={
                str(column): data[column].to_numpy(copy=False)
                for column in data.columns
            },
            edge_attrs=edge_attrs,
        )

        self._n_edges = graph.ecount()

        return graph

    def _get_default_distance_metric(self):
        """Get the appropriate distance metric based on the landscape kind."""
        if self.kind in ["boolean", "dna", "rna", "protein"]:
            return hamming_distance
        else:
            return mixed_distance

    def _compute_local_optima(self) -> "Landscape":
        """Identify local optima with plateau-aware semantics (build-time step).

        A node is a local optimum if it has no way to improve fitness,
        taking neutral plateaus into account.  Results are stored at the
        node level in ``n_lo_members`` / ``lo_index`` / ``is_lo``, at the
        plateau level in ``n_plateau_lo`` / ``plateau_lo_index``, and at the
        distinct-optimum level in ``n_lo`` (number of local optima, each
        plateau-LO counted once) with ``_peak_index`` (one representative
        node per optimum, used by LON construction).

        Internal: the built landscape is treated as immutable, so this runs
        during construction (and as an idempotent recompute hook); it is not a
        public, user-facing operation. Returns ``self``.
        """
        _optima.determine_local_optima(self)
        return self

    def _compute_global_optimum(self) -> "Landscape":
        """Identify the global optimum node (build-time step).

        Internal recompute hook; see :meth:`_compute_local_optima`. Returns ``self``.
        """
        _compute.determine_global_optimum(self)
        return self

    def describe(self) -> Dict[str, Any]:
        """Return a structured summary of the landscape as a dict.

        Unlike a printed summary, the returned mapping is composable and
        testable -- callers can log it, assert on it, or render it. For a quick
        human-readable view use ``print(landscape)`` (see :meth:`__str__`).

        Returns
        -------
        dict
            ``class``, ``kind``, ``built``, ``maximize`` and ``epsilon`` are
            always present. When built, size/optima fields (``n_vars``,
            ``n_configs``, ``n_edges``, ``n_lo``, ``go_index``), the calculation
            flags, and -- if a plateau layer exists -- the plateau counts are
            added.
        """
        report: Dict[str, Any] = {
            "class": self.__class__.__name__,
            "kind": self.kind,
            "built": self._is_built,
            "maximize": self.maximize,
            "epsilon": self.epsilon,
        }
        if not self._is_built:
            return report
        report.update(
            n_vars=self.n_vars,
            n_configs=self.n_configs,
            n_edges=self.n_edges,
            n_lo=self.n_lo,
            go_index=self.go_index,
            has_plateaus=self._has_plateaus,
            basins_calculated=self._basin_calculated,
            paths_calculated=self._path_calculated,
            lon_calculated=self.has_lon,
        )
        if self._has_plateaus:
            report.update(
                n_lo_members=self.n_lo_members,
                n_plateau=self.n_plateau,
                n_plateau_lo=self.n_plateau_lo,
            )
        return report

    @timeit
    def _analyze(self) -> None:
        """Run the mandatory analysis steps after the graph is constructed.

        This computes network metrics, local optima and the global optimum.
        The more expensive, optional analyses (basins, accessible paths,
        distance-to-optimum, neighbour fitness) are computed lazily on first
        access via the ``.basins`` / ``.accessible_paths`` / ``.dist_to_go`` /
        ``.neighbor_fitness`` properties.
        """
        if self.graph is None:
            raise RuntimeError("Graph is None, cannot analyze.")

        if self.graph.vcount() == 0:
            warnings.warn("Cannot analyze an empty graph.", RuntimeWarning)
            self.n_lo = 0
            self.n_lo_members = 0
            self.lo_index = []
            self._peak_index = []
            self.plateau_lo_index = []
            self.n_plateau_lo = 0
            self.go_index = None
            self.go = None
            self.lon = None
            self.has_lon = False
            return

        if self.verbose:
            print("Calculating landscape properties...")

        # In/out degree stay eager: cheap and needed for local-optimum
        # detection. PageRank (70-90% of the old cost here, used by nothing on
        # this path) is deferred to the lazy ``pagerank`` property.
        if "out_degree" not in self.graph.vs.attributes():
            if self.verbose:
                print(" - Calculating network metrics (degrees)...")
            self.graph.vs["in_degree"] = self.graph.indegree()
            self.graph.vs["out_degree"] = self.graph.outdegree()

        # Determine optima (basins / paths / distance / neighbour fitness are lazy).
        self._compute_local_optima()
        self._compute_global_optimum()

