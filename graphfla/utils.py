import networkx as nx
import pandas as pd
from typing import Any

def add_network_metrics(graph: nx.DiGraph, weight: str) -> nx.DiGraph:
    """
    Calculate basic network metrics for nodes in the graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph for which the network metrics are to be calculated.

    weight : str, default='delta_fit'
        The edge attribute key to be considered for weighting. Default is 'delta_fit'.

    Returns
    -------
    nx.DiGraph
        The directed graph with node attributes added.
    """
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph, weight=weight)

    nx.set_node_attributes(graph, in_degree, "in_degree")
    nx.set_node_attributes(graph, out_degree, "out_degree")
    nx.set_node_attributes(graph, pagerank, "pagerank")

    return graph

def get_embedding(
        graph: nx.Graph,
        data: pd.DataFrame,
        model: Any,
        reducer: Any
    ) -> pd.DataFrame:
    """
    Processes a graph to generate embeddings using a specified model and then reduces the dimensionality
    of these embeddings using a given reduction technique. The function then augments the reduced embeddings
    with additional data provided.

    Parameters
    ----------
    graph : nx.Graph
        The graph structure from which to generate embeddings. This is used as input to the model.

    data : pd.DataFrame
        Additional data to be joined with the dimensionally reduced embeddings.

    model : Any
        The embedding model to be applied on the graph. This model should have fit and get_embedding methods.

    reducer : Any
        The dimensionality reduction model to apply on the high-dimensional embeddings. This model should
        have fit_transform methods.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the dimensionally reduced embeddings, now augmented with the additional data.
        Each embedding is represented in two components ('cmp1' and 'cmp2').
    """
    model.fit(graph)
    embeddings = model.get_embedding()
    embeddings = pd.DataFrame(data=embeddings)

    embeddings_low = reducer.fit_transform(embeddings)
    embeddings_low = pd.DataFrame(data=embeddings_low)
    embeddings_low.columns=["cmp1","cmp2"]
    embeddings_low = embeddings_low.join(data)
    
    return embeddings_low

def relabel(graph: nx.Graph) -> nx.Graph:
    """
    Relabels the nodes of a graph to use sequential numerical indices starting from zero. This function
    creates a new graph where each node's label is replaced by a numerical index based on its position
    in the node enumeration.

    Parameters
    ----------
    graph : nx.Graph
        The graph whose nodes are to be relabeled. 

    Returns
    -------
    nx.Graph
        A new graph with nodes relabeled as consecutive integers, maintaining the original graph's structure.
    """
    mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    new_graph = nx.relabel_nodes(graph, mapping)
    return new_graph

