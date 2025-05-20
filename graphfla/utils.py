import numpy as np
import igraph as ig


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
