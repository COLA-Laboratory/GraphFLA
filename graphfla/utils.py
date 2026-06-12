from __future__ import annotations

import logging
import time

import numpy as np
import igraph as ig
import warnings

from functools import wraps


logger = logging.getLogger(__name__)


def timeit(method):
    """
    A decorator to measure and log the execution time of a method.

    The timing is emitted at ``DEBUG`` level on the ``graphfla`` logger, so it
    is silent by default (logging's default level is ``WARNING``) and honours
    ``verbose``. To see per-step timings, raise the package log level::

        import logging
        logging.getLogger("graphfla").setLevel(logging.DEBUG)
        logging.basicConfig()  # if no handler is configured yet
    """

    @wraps(method)
    def timed(*args, **kwargs):
        start_time = time.perf_counter()
        result = method(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logger.debug(
            "Method %s executed in %.4f seconds.", method.__name__, elapsed_time
        )
        return result

    return timed


def filter_graph(graph, maximize, tau, filter_mode, verbose):
    """Apply post-construction filtering to the landscape graph.

    Returns
    -------
    graph : ig.Graph
        The (possibly pruned) graph.
    n_configs : int
        Number of vertices in the final graph.
    n_edges : int
        Number of edges in the final graph.
    kept_vertex_indices : list[int] or None
        Original vertex indices that survived the filter, in new-index order.
        ``None`` when no vertices were removed (identity mapping).
    """
    kept_vertex_indices = None

    if tau is not None:
        if filter_mode == "both":
            if verbose:
                logger.info(
                    f" - Applying post-construction functional filter "
                    f"(tau={tau}, filter_mode='both')..."
                )

            initial_edges = graph.ecount()

            fitness_arr = np.array(graph.vs["fitness"])
            edge_list = graph.get_edgelist()

            if edge_list:
                edge_array = np.asarray(edge_list, dtype=np.int32)
                src_fit = fitness_arr[edge_array[:, 0]]
                tgt_fit = fitness_arr[edge_array[:, 1]]

                if maximize:
                    both_below = (src_fit < tau) & (tgt_fit < tau)
                else:
                    both_below = (src_fit > tau) & (tgt_fit > tau)

                edges_to_remove = np.where(both_below)[0].tolist()
            else:
                edges_to_remove = []

            if edges_to_remove:
                graph.delete_edges(edges_to_remove)

                final_edges = graph.ecount()
                removed_edges = initial_edges - final_edges

                if verbose:
                    logger.info(
                        f"   - Removed {removed_edges} edges connecting "
                        f"non-functional configurations"
                    )
                    logger.info(f"   - Kept {final_edges}/{initial_edges} edges")
            else:
                if verbose:
                    logger.info("   - No edges removed (all connect functional configs)")

            # Keep only the largest weakly connected component
            initial_nodes = graph.vcount()
            components = graph.connected_components(mode="weak")
            membership = np.asarray(components.membership, dtype=np.int32)

            if membership.size > 0:
                component_sizes = np.bincount(membership)
                giant_component = int(component_sizes.argmax())
                kept_mask = membership == giant_component

                if not np.all(kept_mask):
                    kept_vertex_indices = np.flatnonzero(kept_mask).tolist()
                    graph = graph.induced_subgraph(kept_vertex_indices)

            if verbose:
                removed_nodes = initial_nodes - graph.vcount()
                if removed_nodes > 0:
                    logger.info(
                        f"   - Kept largest connected component: "
                        f"{graph.vcount()} nodes "
                        f"({removed_nodes} isolated/minor-component nodes removed)"
                    )
                else:
                    logger.info("   - Graph remains fully connected")

    n_configs = graph.vcount()
    n_edges = graph.ecount()

    if n_configs == 0:
        raise RuntimeError(
            "Landscape graph construction resulted in an empty graph."
        )

    return graph, n_configs, n_edges, kept_vertex_indices


def remove_isolated_nodes(graph, verbose=False, protected=None):
    """Remove nodes with no incident edges from the landscape graph.

    Parameters
    ----------
    graph : ig.Graph
        The directed landscape graph.
    verbose : bool, default=False
        Whether to print removal information.
    protected : set[int] or None, default=None
        Vertex indices to exempt from removal even if they have directed
        degree 0. Used to keep plateau-interior nodes that are connected to
        the landscape only by neutral (tied / within-epsilon) edges, which
        live outside the directed improving-edge graph and would otherwise
        look isolated before the plateau layer is built.

    Returns
    -------
    tuple (graph, n_configs, n_edges, kept_indices) or None
        ``None`` when no isolated nodes exist; otherwise the pruned graph
        and the original vertex indices that were retained.

    Raises
    ------
    ValueError
        If the graph contains no edges at all (fully disconnected).
    """
    if graph.ecount() == 0:
        raise ValueError(
            "Landscape graph has no edges. No neighboring configurations "
            "were detected in the dataset. This usually means the "
            "configuration space is too sparsely sampled or the "
            "neighborhood definition (n_edit, encoding) does not match "
            "the data."
        )

    total_degree = np.asarray(graph.degree())
    isolated_mask = total_degree == 0
    if protected:
        isolated_mask[list(protected)] = False
    n_isolated = int(isolated_mask.sum())

    if n_isolated == 0:
        return None

    warnings.warn(
        f"{n_isolated} isolated configuration(s) with no neighbors "
        f"detected and removed from the landscape graph. These nodes "
        f"had no mutational neighbors in the dataset under the current "
        f"neighborhood definition.",
        UserWarning,
    )

    # kept_indices (ascending) is the new-index -> old-index map for remapping
    # cached metadata. Below the ~0.5 removed-fraction switchover, igraph's
    # induced_subgraph(auto) just copy-and-deletes, so in-place delete_vertices
    # is byte-identical but cheaper; above it igraph reorders edges, so fall
    # back to induced_subgraph to keep output identical.
    kept_indices = np.flatnonzero(~isolated_mask).tolist()
    if n_isolated <= 0.4 * total_degree.size:
        graph.delete_vertices(np.flatnonzero(isolated_mask).tolist())
    else:
        graph = graph.induced_subgraph(kept_indices)

    if verbose:
        logger.info(
            f" - Removed {n_isolated} isolated node(s); "
            f"{graph.vcount()} nodes remain."
        )

    return graph, graph.vcount(), graph.ecount(), kept_indices


def autocorr_numpy(x, lag=1):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x_mean = np.mean(x)

    x_centered = x - x_mean

    numerator = np.dot(x_centered[: n - lag], x_centered[lag:])

    denominator = np.dot(x_centered, x_centered)

    return numerator / denominator if denominator != 0 else np.nan


def add_network_metrics(graph: ig.Graph, weight: str = "delta_fit") -> ig.Graph:
    """
    Calculate basic network metrics for nodes in an igraph directed graph.

    Parameters
    ----------
    graph : ig.Graph
        The directed graph for which the network metrics are to be calculated.

    weight : str, default='delta_fit'
        The edge attribute key to be considered for weighting.

    Returns
    -------
    ig.Graph
        The graph with node attributes added: in-degree, out-degree, and PageRank.
    """
    # Compute in-degree and out-degree
    graph.vs["in_degree"] = graph.indegree()
    graph.vs["out_degree"] = graph.outdegree()

    # Pass the edge-attribute name (not a Python list) so igraph reads weights
    # from its C store, avoiding materialising millions of weights in Python.
    weights = weight if weight in graph.edge_attributes() else None
    pagerank = graph.pagerank(weights=weights, directed=True)

    graph.vs["pagerank"] = pagerank

    return graph


def infer_graph_properties(graph, data_types=None, configs=None, verbose=False):
    """Infer basic landscape properties from a graph and optional metadata.

    Parameters
    ----------
    graph : ig.Graph
        The graph to inspect.
    data_types : dict, optional
        Mapping of variable names to data types. When provided, its length is
        used as the primary source for ``n_vars``.
    configs : pandas.Series, optional
        Configuration sequence aligned with graph vertices. When available, the
        first configuration is used to infer ``n_vars``.
    verbose : bool, default=False
        Whether to emit a warning when ``n_vars`` cannot be inferred reliably.

    Returns
    -------
    tuple[int, int, int | None]
        ``(n_configs, n_edges, n_vars)`` inferred from the provided inputs.
    """
    if graph is None:
        raise RuntimeError("infer_graph_properties called before graph assignment.")

    n_configs = graph.vcount()
    n_edges = graph.ecount()

    # Infer dimensionality (n_vars), preferring the most reliable source.
    if data_types:
        n_vars = len(data_types)
    elif configs is not None and len(configs) > 0:
        try:
            # configs entries are assumed to be tuples/lists of variables
            n_vars = len(configs.iloc[0])
        except Exception:
            n_vars = None
    else:
        # Last resort: guess from vertex attributes named var_*/pos_*/bit_*
        try:
            if graph.vcount() > 0:
                vertex_attrs = graph.vs.attributes()
                potential_var_keys = [
                    k
                    for k in vertex_attrs
                    if isinstance(k, str) and (k.startswith(("var_", "pos_", "bit_")))
                ]
                if potential_var_keys:
                    n_vars = len(potential_var_keys)
                else:
                    n_vars = None
            else:
                n_vars = None
        except Exception:
            n_vars = None

    if n_vars is None and verbose:
        warnings.warn(
            "Could not reliably determine 'n_vars' (number of variables) "
            "from the provided graph and parameters. Distance calculations "
            "or analyses requiring dimensionality might fail.",
            UserWarning,
        )

    return n_configs, n_edges, n_vars
