import numpy as np
import igraph as ig
import pandas as pd
import warnings

from ._data import filter_data


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
                print(
                    f" - Applying post-construction functional filter "
                    f"(tau={tau}, filter_mode='both')..."
                )

            initial_edges = graph.ecount()

            # Vectorized edge filtering using numpy
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
                    print(
                        f"   - Removed {removed_edges} edges connecting "
                        f"non-functional configurations"
                    )
                    print(f"   - Kept {final_edges}/{initial_edges} edges")
            else:
                if verbose:
                    print("   - No edges removed (all connect functional configs)")

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
                    print(
                        f"   - Kept largest connected component: "
                        f"{graph.vcount()} nodes "
                        f"({removed_nodes} isolated/minor-component nodes removed)"
                    )
                else:
                    print("   - Graph remains fully connected")

    n_configs = graph.vcount()
    n_edges = graph.ecount()

    if n_configs == 0:
        raise RuntimeError(
            "Landscape graph construction resulted in an empty graph."
        )

    return graph, n_configs, n_edges, kept_vertex_indices


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

    # Compute PageRank (with weights if the attribute exists)
    weights = graph.es[weight] if weight in graph.edge_attributes() else None
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

    # Attempt to infer the number of variables (dimensionality)
    if data_types:
        n_vars = len(data_types)
    elif configs is not None and len(configs) > 0:
        try:
            # Assumes configs series contains tuples/lists of variables
            n_vars = len(configs.iloc[0])
        except Exception:
            n_vars = None  # Failed inference
    else:
        # Fallback: try to guess from node attributes (less reliable)
        try:
            if graph.vcount() > 0:
                vertex_attrs = graph.vs.attributes()
                # Heuristic: look for attributes like 'var_0', 'pos_1', etc.
                potential_var_keys = [
                    k
                    for k in vertex_attrs
                    if isinstance(k, str) and (k.startswith(("var_", "pos_", "bit_")))
                ]
                if potential_var_keys:
                    n_vars = len(potential_var_keys)
                else:
                    n_vars = None  # No obvious variable attributes
            else:
                n_vars = None  # No vertices to examine
        except Exception:
            n_vars = None  # Failed inference

    if n_vars is None and verbose:
        warnings.warn(
            "Could not reliably determine 'n_vars' (number of variables) "
            "from the provided graph and parameters. Distance calculations "
            "or analyses requiring dimensionality might fail.",
            UserWarning,
        )

    return n_configs, n_edges, n_vars


# def is_ancestor_fast(G: nx.DiGraph, start_node: Any, target_node: Any) -> bool:
#     """
#     Checks if target_node is reachable from start_node by following directed
#     edges (successors) in graph G using Depth-First Search.

#     Parameters
#     ----------
#     G : nx.DiGraph
#         The directed graph.
#     start_node : Any
#         The node to start the search from.
#     target_node : Any
#         The node to check reachability for.

#     Returns
#     -------
#     bool
#         True if target_node is reachable from start_node, False otherwise.
#     """
#     if start_node == target_node:
#         # Consistent with nx.ancestors, a node isn't its own ancestor.
#         # However, for basin definition, a node *is* in its own basin.
#         # The logic in global_optima_accessibility handles this by checking
#         # reachability *to* the GO. If start_node *is* GO, it's trivially reachable.
#         # Let's return True here if start==target for basin logic.
#         return True  # Modified: node is reachable from itself

#     stack = [start_node]
#     visited = {start_node}  # Add start node to visited immediately

#     while stack:
#         node = stack.pop()
#         # Check successors only - follows the directed path forward
#         for successor in G.successors(node):
#             if successor == target_node:
#                 return True
#             if successor not in visited:
#                 visited.add(successor)
#                 stack.append(successor)
#     return False

# def get_embedding(
#     graph: nx.Graph, data: pd.DataFrame, model: Any, reducer: Any
# ) -> pd.DataFrame:
#     """
#     Processes a graph to generate embeddings using a specified model and then reduces the dimensionality
#     of these embeddings using a given reduction technique. The function then augments the reduced embeddings
#     with additional data provided.

#     Parameters
#     ----------
#     graph : nx.Graph
#         The graph structure from which to generate embeddings. This is used as input to the model.

#     data : pd.DataFrame
#         Additional data to be joined with the dimensionally reduced embeddings.

#     model : Any
#         The embedding model to be applied on the graph. This model should have fit and get_embedding methods.

#     reducer : Any
#         The dimensionality reduction model to apply on the high-dimensional embeddings. This model should
#         have fit_transform methods.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame containing the dimensionally reduced embeddings, now augmented with the additional data.
#         Each embedding is represented in two components ('cmp1' and 'cmp2').
#     """
#     model.fit(graph)
#     embeddings = model.get_embedding()
#     embeddings = pd.DataFrame(data=embeddings)

#     embeddings_low = reducer.fit_transform(embeddings)
#     embeddings_low = pd.DataFrame(data=embeddings_low)
#     embeddings_low.columns = ["cmp1", "cmp2"]
#     embeddings_low = embeddings_low.join(data)

#     return embeddings_low


# def relabel(graph: nx.Graph) -> nx.Graph:
#     """
#     Relabels the nodes of a graph to use sequential numerical indices starting from zero. This function
#     creates a new graph where each node's label is replaced by a numerical index based on its position
#     in the node enumeration.

#     Parameters
#     ----------
#     graph : nx.Graph
#         The graph whose nodes are to be relabeled.

#     Returns
#     -------
#     nx.Graph
#         A new graph with nodes relabeled as consecutive integers, maintaining the original graph's structure.
#     """
#     mapping = {node: idx for idx, node in enumerate(graph.nodes())}
#     new_graph = nx.relabel_nodes(graph, mapping)
#     return new_graph
