import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mplt
import numpy as np
import pandas as pd
import umap.umap_ as umap

from palettable.lightbartlein.diverging import BlueOrangeRed_3
from typing import Any
from scipy.interpolate import griddata
from scipy.stats import norm
from karateclub import HOPE
from ..utils import relabel


def draw_landscape_2d(
    landscape: Any,
    metric: str,
    embedding_model: Any = HOPE(),
    reducer: Any = umap.UMAP(n_neighbors=15, n_epochs=500, min_dist=1),
    rank: bool = True,
    n_grids: int = 100,
    cmap: Any = BlueOrangeRed_3,
) -> None:
    """
    Draws a 2D visualization of a landscape by plotting reduced graph embeddings and coloring them
    according to the fitness values.

    Parameters
    ----------
    landscape : Any
        The landscape object that contains the graph and data for visualization.

    metric : str
        The name of the fitness column in the landscape data that will be visualized on the contour plot.

    embedding_model : Any, default=HOPE()
        The model used to generate embeddings from the landscape's graph. It should implement fit and
        get_embedding methods.

    reducer : Any, default=umap.UMAP(...)
        The dimensionality reduction technique to be applied on the embeddings.
    rank : bool, default=True
        If True, ranks the metric values across the dataset.

    n_grids : int, default=100
        The number of divisions along each axis of the plot grid. Higher numbers increase the
        resolution of the contour plot.

    cmap : Any, default=BlueOrangeRed_3
        The color map from 'palettable' used for coloring the contour plot.
    """
    df = prepare_visualization_data(
        landscape, metric, embedding_model, reducer, rank=rank
    )
    cmap = cmap.mpl_colormap

    l_area = mplt.MaxNLocator(nbins=20)
    l_area = l_area.tick_values(df[metric].min(), df[metric].max())

    l_line = mplt.MaxNLocator(nbins=5)
    l_line = l_line.tick_values(df[metric].min(), df[metric].max())

    x_range = np.linspace(df["cmp1"].min(), df["cmp1"].max(), n_grids)
    y_range = np.linspace(df["cmp2"].min(), df["cmp2"].max(), n_grids)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = griddata((df["cmp1"], df["cmp2"]), df[metric], (xx, yy), method="linear")

    plt.contourf(xx, yy, zz, cmap=cmap, levels=l_area, alpha=1)
    plt.colorbar()
    plt.contour(
        xx, yy, zz, levels=5, linewidths=0.35, colors="black", linestyles="solid"
    )
    plt.show()


def draw_landscape_3d(
    landscape: Any,
    metric: str,
    embedding_model=HOPE(),
    reducer=umap.UMAP(n_neighbors=15, n_epochs=500, min_dist=1),
    rank: bool = True,
    n_grids: int = 100,
    cmap=BlueOrangeRed_3,
):
    """
    Draws a 3D interactive visualization of a landscape by plotting reduced graph embeddings and coloring
    them according to a specified metric.

    Parameters
    ----------
    landscape : Any
        The landscape object that contains the graph and data for visualization.

    metric : str
        The name of the fitness score in the landscape data that will be visualized on the contour plot.

    embedding_model : Any, default=HOPE()
        The model used to generate embeddings from the landscape's graph. It should implement fit and
        get_embedding methods.

    reducer : Any, default=umap.UMAP(...)
        The dimensionality reduction technique to be applied on the embeddings.

    rank : bool, default=True
        If True, ranks the metric values across the dataset.

    n_grids : int, default=100
        The number of divisions along each axis of the plot grid. Higher numbers increase the
        resolution of the contour plot.

    cmap : Any, default=BlueOrangeRed_3
        The color map from 'palettable' used for coloring the contour plot.
    """

    df = prepare_visualization_data(
        landscape, metric, embedding_model, reducer, rank=rank
    )
    colorscale = [
        (float(i) / (len(cmap.colors) - 1), color)
        for i, color in enumerate(cmap.hex_colors)
    ]

    x_range = np.linspace(df["cmp1"].min(), df["cmp1"].max(), n_grids)
    y_range = np.linspace(df["cmp2"].min(), df["cmp2"].max(), n_grids)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = griddata((df["cmp1"], df["cmp2"]), df[metric], (xx, yy), method="linear")

    fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale=colorscale)])
    fig.show()


def draw_neighborhood(
    G: nx.DiGraph,
    node: Any,
    radius: int = 1,
    node_size: int = 300,
    with_labels: bool = True,
    font_weight: str = "bold",
    font_size: str = 12,
    font_color: str = "black",
    node_label: str = None,
    node_color: Any = None,
    edge_label: str = None,
    colormap=plt.cm.Blues,
    alpha: float = 1.0,
) -> None:
    """
    Visualizes the neighborhood of a node in a directed graph within a specified radius.

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph.

    node : Any
        The target node whose neighborhood is to be visualized.

    radius : int, optional, default=1
        The radius within which to consider neighbors.

    node_size : int, optional, default=300
        The size of the nodes in the visualization.

    with_labels : bool, optional, default=True
        Whether to display node labels.

    font_weight : str, optional, default='bold'
        Font weight for node labels.

    font_size : str, optional, default=12
        Font size for labels.

    font_color : str, optional, default='black'
        Font color for node labels.

    node_label : str, optional, default=None
        The node attribute to use for labeling, if not the node itself.

    node_color : Any, optional, default=None
        The node attribute to determine node colors.

    edge_label : str, optional, default=None
        The edge attribute to use for labeling edges.

    colormap : matplotlib colormap, optional, default=plt.cm.Blues
        The Matplotlib colormap to use for node coloring.

    alpha : float, optional, default=1.0
        The alpha value for node colors.
    """

    nodes_within_radius = set(
        nx.single_source_shortest_path_length(G, node, radius).keys()
    )
    nodes_within_radius |= set(
        nx.single_source_shortest_path_length(G.reverse(), node, radius).keys()
    )
    H = G.subgraph(nodes_within_radius)

    pos = nx.circular_layout(H)
    pos[node] = (0, 0)

    if node_color:
        attr_values = [H.nodes[n].get(node_color, 0) for n in H.nodes()]
        min_val = min(attr_values)
        max_val = max(attr_values)
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        node_colors = [colormap(norm(val)) for val in attr_values]
    else:
        node_colors = "skyblue"

    nx.draw_networkx_nodes(
        H,
        pos,
        node_size=node_size,
        node_color=node_colors,
        alpha=alpha,
        edgecolors="black",
    )

    outgoing_edges = [(node, n) for n in H.successors(node) if n != node]
    incoming_edges = [(n, node) for n in H.predecessors(node) if n != node]
    other_edges = [(u, v) for u, v in H.edges() if u != node and v != node]

    nx.draw_networkx_edges(
        H,
        pos,
        edgelist=outgoing_edges,
        edge_color="#FF7F50",
        arrows=True,
        connectionstyle="arc3, rad=0.1",
    )
    nx.draw_networkx_edges(
        H,
        pos,
        edgelist=incoming_edges,
        edge_color="#008080",
        arrows=True,
        connectionstyle="arc3, rad=-0.1",
    )
    nx.draw_networkx_edges(
        H,
        pos,
        edgelist=other_edges,
        edge_color="lightgray",
        arrows=True,
        connectionstyle="arc3, rad=-0.1",
    )

    if with_labels:
        labels = {}
        for n in H.nodes():
            label_value = H.nodes[n].get(node_label, n)
            if isinstance(label_value, float):
                labels[n] = "{:.4f}".format(label_value)
            elif isinstance(label_value, int):
                labels[n] = str(label_value)
            else:
                labels[n] = str(label_value)

        label_pos = {node: (pos[node][0], pos[node][1] + 0.1) for node in H.nodes()}

        nx.draw_networkx_labels(
            H,
            label_pos,
            labels,
            font_weight=font_weight,
            font_size=font_size,
            font_color=font_color,
        )

    if edge_label:
        edge_labels = {}
        for u, v in H.edges():
            label_value = H.edges[u, v].get(edge_label, "")
            if isinstance(label_value, float):
                edge_labels[(u, v)] = "{:.4f}".format(label_value)
            elif isinstance(label_value, int):
                edge_labels[(u, v)] = str(label_value)
            else:
                edge_labels[(u, v)] = str(label_value)

        nx.draw_networkx_edge_labels(
            H,
            pos,
            edge_labels=edge_labels,
            font_weight=font_weight,
            font_size=font_size - 4,
            font_color="gray",
        )

    plt.axis("off")
    plt.show()


def prepare_visualization_data(
    landscape: Any, metric: str, embedding_model: Any, reducer: Any, rank: bool = True
) -> pd.DataFrame:
    """
    Prepares data for visualization by generating embeddings of the graph, reducing their dimensionality,
    and optionally ranking the specified metric. The function returns a DataFrame containing the reduced
    embeddings along with the metric values.

    Parameters
    ----------
    landscape : Any
        The landscape object containing the graph and data to be visualized. The graph's structure is used
        to generate embeddings, and the data is augmented with metric values.

    metric : str
        The name of the fitness column in the landscape data to be included in the visualization.

    embedding_model : Any
        The model used to generate node embeddings from the graph. It should have a fit method and a
        get_embedding method.

    reducer : Any
        The dimensionality reduction model applied to the high-dimensional embeddings. It should have a
        fit_transform method.

    rank : bool, default=True
        Whether to rank the metric values. If True, the metric values are ranked in ascending order.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the dimensionally reduced embeddings and the metric values, optionally ranked.
    """

    embedding_model.fit(relabel(landscape.graph))
    embeddings = embedding_model.get_embedding()
    embeddings = pd.DataFrame(data=embeddings)
    embeddings_low = reducer.fit_transform(embeddings)
    embeddings_low = pd.DataFrame(data=embeddings_low, columns=["cmp1", "cmp2"])

    data_ = embeddings_low.join(landscape.get_data()[metric])

    df = data_.copy()
    if rank:
        df[metric] = df[metric].rank()

    return df
