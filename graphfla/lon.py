import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Any, Dict, Tuple
from itertools import combinations, product
from tqdm import tqdm
from .utils import add_network_metrics

def get_lon(
    graph: nx.DiGraph,
    configs: pd.Series,
    lo_index: List[int],
    config_dict: Dict[Any, Any],
    maximize: bool = True,
    mlon: bool = True,
    min_edge_freq: int = 3,
    trim: int = None,
    verbose: bool = True
) -> nx.DiGraph:
    """
    Construct the local optima network (LON) of the fitness landscape.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph of the landscape.

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
    lon : nx.DiGraph
        The constructed local optimum network (LON).
    """
    
    if verbose:
        print("Constructing local optima network...")

    lo_configs = configs.iloc[lo_index].tolist()
    lo_neighbors_list = batched_find_neighbors(lo_configs, config_dict, n_edit=2, verbose=verbose)

    basin_index = pd.Series(nx.get_node_attributes(graph, 'basin_index')).sort_index()
    n_lo = len(lo_index)
    lo_to_index_mapping = dict(zip(lo_index, range(n_lo)))
    basin_index = basin_index.map(lo_to_index_mapping)
    config_to_basin_mapping = dict(zip(configs.tolist(), basin_index))

    lo_adj = calculate_lon_adj(
        lo_neighbors_list,
        config_to_basin_mapping,
        n_lo=n_lo,
        min_edge_freq=min_edge_freq,
        verbose=verbose
    )

    lon = create_lon(
        graph,
        lo_adj,
        lo_index,
        verbose=verbose
    )

    escape_difficulty = calculate_escape_rate(lo_adj, lo_index, n_lo=n_lo, verbose=verbose)
    nx.set_node_attributes(lon, escape_difficulty, 'escape_difficulty')

    improvement_measure = calculate_improve_rate(lon, maximize, verbose=verbose)
    nx.set_node_attributes(lon, improvement_measure, 'improve_rate')

    if mlon:
        lon = get_mlon(lon, maximize, 'fitness')
        if verbose:
            print(" - The LON has been reduced to M-LON by keeping only improving edges")

    if trim:
        lon = trim_lon(lon, trim, 'fitness')
        if verbose:
            print(f" - The LON has been trimmed to keep only {trim} edges for each node.")

    accessibility = calculate_lo_accessibility(lon, verbose=verbose)
    nx.set_node_attributes(lon, accessibility, 'accessibility')

    if verbose:
        print("# Adding further node attributes...")
    lon = add_network_metrics(lon, weight="weight")

    return lon

def batched_find_neighbors(
    configs: List[Tuple[Any, ...]],
    config_dict: Dict[Any, Any],
    n_edit: int = 1,
    verbose: bool = True
) -> List[List[Tuple]]:
    """Finding the neighbors for a list of configurations"""

    neighbor_list = []
    iterator = configs if not verbose else tqdm(
        configs, total=len(configs), desc="# Calculating neighborhoods"
    )
    for config in iterator:
        neighbor_list.append(generate_neighbors(config, config_dict, n_edit=n_edit))
    return neighbor_list

def generate_neighbors(
    config: Tuple[Any, ...],
    config_dict: Dict[Any, Any],
    n_edit: int = 1
) -> List[Tuple[Any, ...]]:
    """Finding the neighbors of a given configuration"""

    def get_neighbors(index, value):
        config_type = config_dict[index]['type']
        config_max = config_dict[index]['max']
        
        if config_type == 'categorical':
            return [i for i in range(config_max + 1) if i != value]
        elif config_type == 'ordinal':
            neighbors = []
            if value > 0:
                neighbors.append(value - 1)
            if value < config_max:
                neighbors.append(value + 1)
            return neighbors
        elif config_type == 'boolean':
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
    verbose: bool = True
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
    iterator = enumerate(neighbors_list) if not verbose else tqdm(
        enumerate(neighbors_list),
        total=n_lo,
        desc=" - Creating adjacency matrix"
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
    graph: nx.DiGraph,
    lo_adj: np.ndarray,
    lo_index: List[int],
    verbose: bool = True
) -> nx.DiGraph:
    """
    Create LON based on adjacency matrix.

    Parameters
    ----------
    graph : nx.DiGraph
        Original graph of the landscape.

    lo_adj : np.ndarray
        Adjacency matrix for the LON.

    lo_index : List[int]
        List of indices of local optima.

    Returns
    -------
    lon : nx.DiGraph
        The local optima network.
    """

    if verbose:
        print("# Creating LON from adjacency matrix...")
    n_lo = len(lo_index)
    index_to_lo_mapping = dict(zip(range(n_lo), lo_index))
    lon = nx.DiGraph(lo_adj)
    lon = nx.relabel_nodes(lon, index_to_lo_mapping)

    graph_lon_ = graph.subgraph(lo_index)
    for attribute in ['fitness', 'size_basin', 'max_radius_basin', 'config']:
        attr_dict = nx.get_node_attributes(graph_lon_, attribute)
        nx.set_node_attributes(lon, attr_dict, attribute)

    self_loops = list(nx.selfloop_edges(lon))
    lon.remove_edges_from(self_loops)

    return lon

def calculate_escape_rate(
    lo_adj: np.ndarray,
    lo_index: List[int],
    n_lo: int,
    verbose: bool = True
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
    iterator = range(n_lo) if not verbose else tqdm(
        range(n_lo), total=n_lo, desc="# Calculating escape probability"
    )
    for i in iterator:
        if column_sums[i] != 0:  
            escape_difficulty_values[i] = lo_adj[i, i] / (column_sums[i] + lo_adj[i, i])
        else:
            escape_difficulty_values[i] = 1
    escape_difficulty = dict(zip(lo_index, escape_difficulty_values))
    return escape_difficulty

def calculate_improve_rate(
    lon: nx.DiGraph,
    maximize: bool = True,
    verbose: bool = True
) -> Dict[Any, float]:
    """
    Calculate the improve rate for each node in the LON.

    Parameters
    ----------
    lon : nx.DiGraph
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
    iterator = lon.nodes() if not verbose else tqdm(
        lon.nodes(), total=lon.number_of_nodes()
    )

    for node in iterator:
        total_outgoing_weight = 0
        improving_moves_weight = 0
        current_fitness = lon.nodes[node]['fitness']
        
        for target in lon.successors(node):
            edge_data = lon.get_edge_data(node, target)
            edge_weight = edge_data['weight']
            total_outgoing_weight += edge_weight
            target_fitness = lon.nodes[target]['fitness']
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

def calculate_lo_accessibility(
    lon: nx.DiGraph,
    verbose: bool = True
) -> Dict[Any, int]:
    """
    Calculate the accessibility of each local optimum in the LON.

    Parameters
    ----------
    lon : nx.DiGraph
        The local optima network.

    Returns
    -------
    accessibility : Dict[Any, int]
        Dictionary mapping nodes to their accessibility.
    """

    access_lon = {}
    iterator = lon.nodes() if not verbose else tqdm(
        lon.nodes(), total=lon.number_of_nodes(),
        desc="# Calculating accessibility of LOs:"
    )
    for node in iterator:
        access_lon[node] = len(nx.ancestors(lon, node))
    return access_lon

def get_mlon(
        graph: nx.DiGraph, 
        maximize: bool = True, 
        attribute: str = 'fitness'
    ) -> nx.DiGraph:
    """
    Generates a Monotonic Local Optima Network (M-LON) from a given directed graph.
    
    Parameters
    ----------
    G : nx.DiGraph
        The LON to be trimmed.

    maximize : bool
        Whether the fitness is to be optimized.
    
    attribute : str, default = "weight"
        The edge attribute key based on which the edges are sorted. Default is 'weight'.

    Return
    ------
    nx.DiGraph: The resulting M-LON
    """
    
    if maximize:
        edges_to_remove = [(source, end) for source, end in graph.edges()
                        if graph.nodes[source][attribute] > graph.nodes[end][attribute]]
    else:
        edges_to_remove = [(source, end) for source, end in graph.edges()
                        if graph.nodes[source][attribute] < graph.nodes[end][attribute]]

    graph.remove_edges_from(edges_to_remove)

    return graph

def trim_lon(
        graph: nx.DiGraph,
        k: int = 10, 
        attribute: str = 'weight'
    ) -> nx.DiGraph:
    """
    Trim the LON to keep only k out-goging edges from each local optiam with the largest transition probability.

    Parameters
    ----------
    G : nx.DiGraph
        The LON to be trimmed.

    k : int, default=10
        The number of edges to retain for each node. Default is 10.

    attribute : str, default = "weight"
        The edge attribute key based on which the edges are sorted. Default is 'weight'.
    
    Return
    ------
    nx.DiGraph: The resulting trimmed LON.
    """

    for node in graph.nodes():
        edges = sorted(graph.out_edges(node, data=True), key=lambda x: x[2][attribute], reverse=True)
        edges_to_keep = edges[:k]
        edges_to_remove = [edge for edge in graph.out_edges(node) if edge not in [e[:2] for e in edges_to_keep]]
        graph.remove_edges_from(edges_to_remove)
    
    return graph