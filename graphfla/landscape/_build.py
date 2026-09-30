"""Construction pipeline for :class:`~graphfla.landscape.landscape.Landscape`.

The private build internals — strategy resolution, preprocessing, graph
construction, pruning/remap, edge building and the post-construction analysis —
mixed into the core class as ``_BuildMixin``. The public ``build_from_data``
entry point stays on ``Landscape`` (it is the "fit"); these are the helpers it
orchestrates, kept here so landscape.py reads as the class surface.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import igraph as ig

from .._data import (
    InputHandler,
    PreparedData,
    filter_data,
    prepare_data,
    clean_data,
    encode_data,
)
from .._neighbors import build_edges
from ..utils import _largest_weak_component, remove_isolated_nodes, timeit
from ..exceptions import InvalidParameterError, NotBuiltError
import logging

logger = logging.getLogger(__name__)
_ATTRIBUTE_BUFFER_BYTES = 16 * 1024**2


class _BuildMixin:
    """Private graph-construction methods for :class:`Landscape`."""

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
        self._configs_index = range(len(prepared.data_for_attributes))
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
        tau: Optional[float] = None,
        filter_mode: str = "any",
    ) -> List[Tuple[int, int]]:
        """Construct the graph from preprocessed data and return neutral pairs."""
        if self.verbose:
            logger.info("Constructing landscape graph...")

        edges, delta_fits, neutral_pairs = self._build_edges(
            processed_data, n_edit=n_edit, strategy=neighborhood_strategy
        )
        if tau is not None and filter_mode == "both" and len(edges):
            edges = np.asarray(edges, dtype=np.int64)
            # The target is the fitter endpoint of every improving edge.
            target_fitness = processed_data["fitness"].to_numpy()[edges[:, 1]]
            keep = target_fitness >= tau if self.maximize else target_fitness <= tau
            edges = edges[keep]
            delta_fits = np.asarray(delta_fits)[keep]
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
        kept_indices = None
        if tau is not None and filter_mode == "both":
            self.graph, kept_indices = _largest_weak_component(self.graph, verbose)
        self._n_configs, self._n_edges = self.graph.vcount(), self.graph.ecount()

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
            logger.info("Landscape built successfully.\n")
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
        """Build a directed graph and attach aligned fitness and variant data."""
        if self.verbose:
            logger.info(" - Constructing graph object and attributes...")

        edge_attrs = {}
        if len(edges):
            values = delta_fits
            if isinstance(values, np.ndarray) and values.flags.c_contiguous:
                values = memoryview(values)
            edge_attrs["delta_fit"] = values

        stream = data.size * np.dtype(object).itemsize > _ATTRIBUTE_BUFFER_BYTES
        vertex_attrs = {} if stream else {
            str(column): data[column].to_numpy(copy=False) for column in data.columns
        }
        graph = ig.Graph(
            n=len(data), edges=edges if len(edges) else None, directed=True,
            vertex_attrs=vertex_attrs, edge_attrs=edge_attrs,
        )
        # Bound temporary decoded arrays for long genetic backgrounds.
        if stream:
            for column in data.columns:
                graph.vs[str(column)] = data[column].to_numpy(copy=False)
        self._n_edges = graph.ecount()
        return graph

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
            logger.info("Calculating landscape properties...")

        # In/out degree stay eager: cheap and needed for local-optimum
        # detection. PageRank (70-90% of the old cost here, used by nothing on
        # this path) is deferred to the lazy ``pagerank`` property.
        if "out_degree" not in self.graph.vs.attributes():
            if self.verbose:
                logger.info(" - Calculating network metrics (degrees)...")
            self.graph.vs["in_degree"] = self.graph.indegree()
            self.graph.vs["out_degree"] = self.graph.outdegree()

        # Determine optima (basins / paths / distance / neighbour fitness are lazy).
        self._compute_local_optima()
        self._compute_global_optimum()
