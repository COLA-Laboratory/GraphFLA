import numpy as np
import igraph as ig
import pandas as pd


def apply_pre_construction_filter(X, f, maximize, functional_threshold, functional_filter_strategy, verbose):
    if functional_threshold:
        if functional_filter_strategy == "any":
            if verbose:
                print(
                    f" - Applying functional threshold filter "
                    f"(threshold={functional_threshold})..."
                )

            initial_count = len(f)

            if maximize:
                mask = f >= functional_threshold
                comparison_op = ">="
            else:
                mask = f <= functional_threshold
                comparison_op = "<="

            X = X[mask]
            f = f[mask]

            X.reset_index(drop=True, inplace=True)
            f.reset_index(drop=True, inplace=True)

            final_count = len(f)
            removed_count = initial_count - final_count

            if verbose:
                opposite_op = "<" if comparison_op[0] == ">" else ">"
                opposite_op += "=" if len(comparison_op) > 1 else ""

                print(
                    f"   - Removed {removed_count} configurations with fitness "
                    f"{opposite_op} {functional_threshold}"
                )
                print(f"   - Kept {final_count}/{initial_count} configurations")

            if final_count == 0:
                raise ValueError(
                    f"All configurations removed by functional threshold filter "
                    f"(threshold={functional_threshold})"
                )

    return X, f


def apply_post_construction_filter(graph, maximize, functional_threshold, functional_filter_strategy, verbose):
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

    if functional_threshold:
        if functional_filter_strategy == "both":
            if verbose:
                print(
                    f" - Applying post-construction functional filter "
                    f"(threshold={functional_threshold}, strategy='both')..."
                )

            initial_edges = graph.ecount()

            # Get fitness values for all nodes
            fitness_values = graph.vs["fitness"]

            # Identify edges to remove based on optimization direction
            edges_to_remove = []

            for edge_idx in range(graph.ecount()):
                edge = graph.es[edge_idx]
                source_fitness = fitness_values[edge.source]
                target_fitness = fitness_values[edge.target]

                # Check if both endpoints are below threshold
                if maximize:
                    both_below = (
                        source_fitness < functional_threshold
                        and target_fitness < functional_threshold
                    )
                else:
                    both_below = (
                        source_fitness > functional_threshold
                        and target_fitness > functional_threshold
                    )

                if both_below:
                    edges_to_remove.append(edge_idx)

            # Remove edges in reverse order to maintain correct indices
            if edges_to_remove:
                edges_to_remove.sort(reverse=True)
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

            # Keep only the largest connected component
            initial_nodes = graph.vcount()

            # Tag each vertex with its pre-filter index so we can recover
            # the mapping after giant() re-indexes vertices.
            graph.vs["_orig_idx"] = list(range(initial_nodes))

            components = graph.connected_components(mode="weak")
            graph = components.giant()

            if graph.vcount() < initial_nodes:
                kept_vertex_indices = list(graph.vs["_orig_idx"])

            # Remove the temporary attribute
            del graph.vs["_orig_idx"]

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
