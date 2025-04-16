import numpy as np
import pandas as pd
import igraph as ig
from typing import List, Any, Dict, Tuple
from itertools import combinations, product
from tqdm import tqdm
from .utils import add_network_metrics


def get_lon(
    graph: ig.Graph,
    configs: pd.Series,
    lo_index: List[int],
    config_dict: Dict[Any, Any],
    maximize: bool = True,
    mlon: bool = True,
    min_edge_freq: int = 3,
    trim: int = None,
    verbose: bool = True,
) -> ig.Graph:
    """
    Construct the local optima network (LON) of the fitness landscape.

    Parameters
    ----------
    graph : ig.Graph
        The igraph Graph of the landscape.

    configs : pd.Series
        Series of configurations (as tuples).

    lo_index : List[int]
        List of indices of local optima.

    config_dict : Dict[Any, Any]
        Configuration dictionary specifying variable types and max values.

    maximize : bool, default=True
        Whether the fitness is to be maximized or minimized.

    mlon : bool, default=True
        Whether to use monotonic-LON (M-LON), which will only have improving edges.

    min_edge_freq : int, default=3
        Minimal escape frequency needed to construct an edge between two local optima.

    trim : int, default=None
        The number of edges with the highest transition probability to retain for each node.

    verbose : bool, default=True
        Whether to print verbose messages.

    Returns
    -------
    lon : ig.Graph
        The constructed local optimum network (LON) as an igraph Graph.
    """

    if verbose:
        print("Constructing local optima network...")

    lo_configs = configs.iloc[lo_index].tolist()
    lo_neighbors_list = batched_find_neighbors(
        lo_configs, config_dict, n_edit=2, verbose=verbose
    )

    # Extract basin_index attribute from graph vertices
    basin_index = pd.Series([graph.vs[i]["basin_index"] for i in range(graph.vcount())])
    basin_index = basin_index.sort_index()

    n_lo = len(lo_index)
    lo_to_index_mapping = dict(zip(lo_index, range(n_lo)))
    basin_index = basin_index.map(lo_to_index_mapping)
    config_to_basin_mapping = dict(zip(configs.tolist(), basin_index))

    lo_adj = calculate_lon_adj(
        lo_neighbors_list,
        config_to_basin_mapping,
        n_lo=n_lo,
        min_edge_freq=min_edge_freq,
        verbose=verbose,
    )

    # Create LON as igraph Graph
    lon = create_lon(graph, lo_adj, lo_index, verbose=verbose)

    escape_difficulty = calculate_escape_rate(
        lo_adj, lo_index, n_lo=n_lo, verbose=verbose
    )
    # Set vertex attributes in igraph Graph
    for lo_idx, difficulty in escape_difficulty.items():
        vertex_idx = lon.vs.find(name=lo_idx).index
        lon.vs[vertex_idx]["escape_difficulty"] = difficulty

    improvement_measure = calculate_improve_rate(lon, maximize, verbose=verbose)
    for lo_idx, improve_rate in improvement_measure.items():
        vertex_idx = lon.vs.find(name=lo_idx).index
        lon.vs[vertex_idx]["improve_rate"] = improve_rate

    if mlon:
        lon = get_mlon(lon, maximize, "fitness")
        if verbose:
            print(
                " - The LON has been reduced to M-LON by keeping only improving edges"
            )

    if trim:
        lon = trim_lon(lon, trim, "fitness")
        if verbose:
            print(
                f" - The LON has been trimmed to keep only {trim} edges for each node."
            )

    accessibility = calculate_lo_accessibility(lon, verbose=verbose)
    for lo_idx, access_val in accessibility.items():
        vertex_idx = lon.vs.find(name=lo_idx).index
        lon.vs[vertex_idx]["accessibility"] = access_val

    if verbose:
        print("# Adding further node attributes...")
    lon = add_network_metrics(lon, weight="weight")

    return lon


def batched_find_neighbors(
    configs: List[Tuple[Any, ...]],
    config_dict: Dict[Any, Any],
    n_edit: int = 1,
    verbose: bool = True,
) -> List[List[Tuple]]:
    """Finding the neighbors for a list of configurations"""

    neighbor_list = []
    iterator = (
        configs
        if not verbose
        else tqdm(configs, total=len(configs), desc="# Calculating neighborhoods")
    )
    for config in iterator:
        neighbor_list.append(generate_neighbors(config, config_dict, n_edit=n_edit))
    return neighbor_list


def generate_neighbors(
    config: Tuple[Any, ...], config_dict: Dict[Any, Any], n_edit: int = 1
) -> List[Tuple[Any, ...]]:
    """Finding the neighbors of a given configuration"""

    def get_neighbors(index, value):
        config_type = config_dict[index]["type"]
        config_max = config_dict[index]["max"]

        if config_type == "categorical":
            return [i for i in range(config_max + 1) if i != value]
        elif config_type == "ordinal":
            neighbors = []
            if value > 0:
                neighbors.append(value - 1)
            if value < config_max:
                neighbors.append(value + 1)
            return neighbors
        elif config_type == "boolean":
            return [1 - value]
        else:
            raise ValueError(f"Unknown variable type: {config_type}")

    def k_edit_combinations():
        original_config = config
        for indices in combinations(range(len(config)), n_edit):
            current_config = list(original_config)
            possible_values = [get_neighbors(i, current_config[i]) for i in indices]
            for changes in product(*possible_values):
                for idx, new_value in zip(indices, changes):
                    current_config[idx] = new_value
                yield tuple(current_config)

    return list(k_edit_combinations())


def calculate_lon_adj(
    neighbors_list: List[List[Tuple]],
    config_to_basin_mapping: Dict[Tuple[Any, ...], int],
    n_lo: int,
    min_edge_freq: int = 3,
    verbose: bool = True,
) -> np.ndarray:
    """
    Calculate the adjacency matrix for LON.

    Parameters
    ----------
    neighbors_list : List[List[Tuple]]
        List of lists of neighbor configurations for each local optimum.

    config_to_basin_mapping : Dict[Tuple[Any, ...], int]
        Mapping from configurations to basin indices.

    n_lo : int
        Number of local optima.

    min_edge_freq : int, default=3
        Minimal escape frequency needed to construct an edge between two local optima.

    Returns
    -------
    lo_adj : np.ndarray
        Adjacency matrix of the LON.
    """

    lo_adj = np.zeros((n_lo, n_lo), dtype=np.int16)
    iterator = (
        enumerate(neighbors_list)
        if not verbose
        else tqdm(
            enumerate(neighbors_list), total=n_lo, desc=" - Creating adjacency matrix"
        )
    )
    for i, lo_neighbors in iterator:
        for neighbor in lo_neighbors:
            basin_j = config_to_basin_mapping.get(neighbor)
            if basin_j is not None:
                lo_adj[i, basin_j] += 1

    if verbose:
        print(f" - Masking positions with transition frequency <= {min_edge_freq}")
    lo_adj = np.where(lo_adj <= min_edge_freq, 0, lo_adj)

    return lo_adj


def create_lon(
    graph: ig.Graph, lo_adj: np.ndarray, lo_index: List[int], verbose: bool = True
) -> ig.Graph:
    """
    Create LON based on adjacency matrix.

    Parameters
    ----------
    graph : ig.Graph
        Original igraph Graph of the landscape.

    lo_adj : np.ndarray
        Adjacency matrix for the LON.

    lo_index : List[int]
        List of indices of local optima.

    Returns
    -------
    lon : ig.Graph
        The local optima network as an igraph Graph.
    """

    if verbose:
        print("# Creating LON from adjacency matrix...")
    n_lo = len(lo_index)

    # Create igraph Graph
    lon = ig.Graph(directed=True)

    # Add vertices with names matching the original local optima indices
    lon.add_vertices(n_lo)
    lon.vs["name"] = lo_index

    # Add edges from adjacency matrix
    edges = []
    weights = []
    for i in range(n_lo):
        for j in range(n_lo):
            if lo_adj[i, j] > 0 and i != j:  # Skip self-loops
                edges.append((i, j))
                weights.append(lo_adj[i, j])

    lon.add_edges(edges)
    if edges:
        lon.es["weight"] = weights

    # Create a subgraph of local optima from the original graph
    lo_subgraph = graph.subgraph(lo_index)

    # Transfer vertex attributes from igraph subgraph to LON graph
    for i, lo_idx in enumerate(lo_index):
        # Match vertex in LON to corresponding vertex in subgraph
        lon_vertex = lon.vs[i]

        # Extract required vertex attributes
        for attr in ["fitness", "size_basin", "max_radius_basin", "config"]:
            if attr in lo_subgraph.vs.attributes():
                lon_vertex[attr] = lo_subgraph.vs[i][attr]

    return lon


def calculate_escape_rate(
    lo_adj: np.ndarray, lo_index: List[int], n_lo: int, verbose: bool = True
) -> Dict[Any, float]:
    """
    Calculate the probability of escaping from a local optimum.

    Parameters
    ----------
    lo_adj : np.ndarray
        Adjacency matrix of the LON.

    lo_index : List[int]
        List of indices of local optima.

    n_lo : int
        Number of local optima.

    Returns
    -------
    escape_difficulty : Dict[Any, float]
        Dictionary mapping local optimum node to its escape difficulty.
    """

    column_sums = np.sum(lo_adj, axis=1) - np.diag(lo_adj)
    escape_difficulty_values = np.zeros(n_lo)
    iterator = (
        range(n_lo)
        if not verbose
        else tqdm(range(n_lo), total=n_lo, desc="# Calculating escape probability")
    )
    for i in iterator:
        if column_sums[i] != 0:
            escape_difficulty_values[i] = lo_adj[i, i] / (column_sums[i] + lo_adj[i, i])
        else:
            escape_difficulty_values[i] = 1
    escape_difficulty = dict(zip(lo_index, escape_difficulty_values))
    return escape_difficulty


def calculate_improve_rate(
    lon: ig.Graph, maximize: bool = True, verbose: bool = True
) -> Dict[Any, float]:
    """
    Calculate the improve rate for each node in the LON.

    Parameters
    ----------
    lon : ig.Graph
        The local optima network.

    maximize : bool, default=True
        Whether the fitness is to be maximized or minimized.

    Returns
    -------
    improvement_measure : Dict[Any, float]
        Dictionary mapping nodes to their improve rates.
    """

    if verbose:
        print("# Calculating improve rate...")
    improvement_measure = {}
    iterator = lon.vs if not verbose else tqdm(lon.vs, total=lon.vcount())

    for vertex in iterator:
        node = vertex["name"]
        total_outgoing_weight = 0
        improving_moves_weight = 0
        current_fitness = vertex["fitness"]

        # Get all outgoing edges
        out_edges = lon.es.select(_source=vertex.index)

        for edge in out_edges:
            edge_weight = edge["weight"]
            total_outgoing_weight += edge_weight

            # Get target vertex
            target_vertex = lon.vs[edge.target]
            target_fitness = target_vertex["fitness"]

            if maximize:
                if target_fitness > current_fitness:
                    improving_moves_weight += edge_weight
            else:
                if target_fitness < current_fitness:
                    improving_moves_weight += edge_weight

        if total_outgoing_weight > 0:
            improvement_measure[node] = improving_moves_weight / total_outgoing_weight
        else:
            improvement_measure[node] = 0
    return improvement_measure


def calculate_lo_accessibility(lon: ig.Graph, verbose: bool = True) -> Dict[Any, int]:
    """
    Calculate the accessibility of each local optimum in the LON.

    Parameters
    ----------
    lon : ig.Graph
        The local optima network.

    Returns
    -------
    accessibility : Dict[Any, int]
        Dictionary mapping nodes to their accessibility.
    """

    access_lon = {}
    iterator = (
        lon.vs
        if not verbose
        else tqdm(
            lon.vs,
            total=lon.vcount(),
            desc="# Calculating accessibility of LOs:",
        )
    )
    for vertex in iterator:
        node = vertex["name"]
        # Find all vertices that have paths to this vertex (ancestors)
        ancestors = lon.subcomponent(vertex.index, mode="in")
        access_lon[node] = len(ancestors)
    return access_lon


def get_mlon(
    graph: ig.Graph, maximize: bool = True, attribute: str = "fitness"
) -> ig.Graph:
    """
    Generates a Monotonic Local Optima Network (M-LON) from a given igraph Graph.

    Parameters
    ----------
    graph : ig.Graph
        The LON to be trimmed.

    maximize : bool
        Whether the fitness is to be optimized.

    attribute : str, default = "fitness"
        The vertex attribute key based on which the edges are filtered.

    Return
    ------
    ig.Graph: The resulting M-LON
    """

    # Create list of edges to remove
    edges_to_remove = []

    for edge in graph.es:
        source_idx = edge.source
        target_idx = edge.target

        source_fitness = graph.vs[source_idx][attribute]
        target_fitness = graph.vs[target_idx][attribute]

        if maximize:
            if source_fitness > target_fitness:
                edges_to_remove.append(edge.index)
        else:
            if source_fitness < target_fitness:
                edges_to_remove.append(edge.index)

    # Create a copy of the graph and remove edges
    mlon_graph = graph.copy()
    mlon_graph.delete_edges(edges_to_remove)

    return mlon_graph


def trim_lon(graph: ig.Graph, k: int = 10, attribute: str = "weight") -> ig.Graph:
    """
    Trim the LON to keep only k out-going edges from each local optimum with the largest transition probability.

    Parameters
    ----------
    graph : ig.Graph
        The LON to be trimmed.

    k : int, default=10
        The number of edges to retain for each node. Default is 10.

    attribute : str, default = "weight"
        The edge attribute key based on which the edges are sorted. Default is 'weight'.

    Return
    ------
    ig.Graph: The resulting trimmed LON.
    """

    # Create a copy of the graph to modify
    trimmed_graph = graph.copy()

    # Process each vertex
    for vertex in trimmed_graph.vs:
        # Get all outgoing edges for this vertex
        out_edges = trimmed_graph.es.select(_source=vertex.index)

        if len(out_edges) <= k:
            continue  # Skip if there are k or fewer edges

        # Sort edges by weight in descending order
        edge_weights = [(e.index, e[attribute]) for e in out_edges]
        edge_weights.sort(key=lambda x: x[1], reverse=True)

        # Identify edges to remove (all except the top k)
        edges_to_remove = [idx for idx, _ in edge_weights[k:]]

        # Remove edges
        trimmed_graph.delete_edges(edges_to_remove)

    return trimmed_graph
