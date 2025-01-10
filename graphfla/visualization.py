import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mplt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import umap.umap_ as umap

from palettable.lightbartlein.diverging import BlueOrangeRed_3
from typing import Any
from scipy.interpolate import griddata
from scipy.stats import norm
from karateclub import HOPE
from .utils import relabel
from .algorithms import hill_climb

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


def draw_epistasis(
    epistasis_df,
    p_threshold=0.05,
    cohen_d_threshold=0.5,
    figsize=(8, 8),
    node_color="#f2f2f2",
    label_font_size=10,
    node_size=500,
    legend_loc="upper right",
    edge_width_scale=2,
):
    """
    Visualize epistatic interactions between all pairs of genetic positions with customizable
    plot formatting.

    Parameters
    ----------
    epistasis_df : pd.DataFrame
        Aggregated epistasis results for all position pairs. This can be generated by the
        `all_pairwise_epistasis` function in the metrics module.

    p_threshold : float, default=0.05
        p-value threshold for significance.

    cohen_d_threshold : float, default=0.5
        Threshold for Cohen's d to define strong interactions.

    figsize : tuple, default=(8, 8)
        Size of the plot figure.

    node_color : str, default='#f2f2f2'
        Color of the nodes in the plot.

    label_font_size : int, default=10
        Font size for the node labels.

    node_size : int, default=500
        Size of the nodes in the plot.

    legend_loc : str, default='upper right'
        Location of the legend.

    edge_width_scale : float, default=2
        Scale factor for edge width based on `average_cohen_d`.

    Returns
    -------
    None
        Displays the plot with enhanced formatting options.
    """

    graph = nx.from_pandas_edgelist(
        epistasis_df,
        source="pos1",
        target="pos2",
        edge_attr=[
            "average_cohen_d",
            "average_mean_diff",
            "most_significant_p",
            "total_mutation_pairs",
        ],
    )

    edge_colors = []
    edge_widths = []

    for u, v, data in graph.edges(data=True):
        if (
            data["average_mean_diff"] > 0
            and data["most_significant_p"] <= p_threshold
            and data["average_cohen_d"] >= cohen_d_threshold
        ):
            rgb = (233, 110, 34)
        elif (
            data["average_mean_diff"] < 0
            and data["most_significant_p"] <= p_threshold
            and data["average_cohen_d"] >= cohen_d_threshold
        ):
            rgb = (50, 143, 157)
        else:
            rgb = (221, 221, 221)

        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        edge_colors.append(hex_color)

        edge_width = (
            data["average_cohen_d"] * edge_width_scale
            if not np.isnan(data["average_cohen_d"])
            else 0.5
        )
        edge_widths.append(edge_width)

    pos = nx.circular_layout(graph, scale=1)
    _, ax = plt.subplots(figsize=figsize)

    nx.draw(
        graph,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        node_color=node_color,
        node_size=node_size,
        with_labels=True,
        font_size=label_font_size,
        font_weight="bold",
        edgecolors="black",
        ax=ax,
    )

    positive_patch = mpatches.Patch(color="#e96e22", label="Positive Epistasis")
    negative_patch = mpatches.Patch(color="#328f9d", label="Negative Epistasis")
    non_significant_patch = mpatches.Patch(color="#dddddd", label="Non-significant")

    ax.legend(
        handles=[positive_patch, negative_patch, non_significant_patch],
        loc=legend_loc,
        title="Epistasis type",
    )

    ax.axis("off")
    plt.show()


def draw_fdc(
    data: pd.DataFrame,
    distance: str,
    fitness: str,
    confidence_level: float = 0.95,
) -> None:
    """
    Plot the average fitness trend as a function of distance to global optimum.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset as a pandas DataFrame.
        Contains at least the columns specified by `distance` and `fitness`.

    distance : str
        The column name of the distances to global optimum.

    fitness : str
        The column name of the fitness values.

    confidence_level : float, optional, default=0.95
        The desired confidence level for the interval, expressed as a value between 0 and 1
        (e.g., 0.95 for a 95% confidence interval).

    Returns
    -------
    None
        Displays a plot of the mean trend with shaded confidence intervals.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'distance': [1, 2, 3, 4], 'fitness': [10, 15, 10, 20]})
    >>> fdc_plot(data, distance='distance', fitness='fitness', confidence_level=0.95)
    """
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be a float between 0 and 1.")

    z_value = norm.ppf(1 - (1 - confidence_level) / 2)

    grouped = data.groupby(distance)[fitness]
    mean_values = grouped.mean()
    std_values = grouped.std()
    count_values = grouped.count()
    confidence_intervals = z_value * (std_values / np.sqrt(count_values))

    stats = (
        pd.DataFrame({"Mean": mean_values, "Confidence Interval": confidence_intervals})
        .dropna()
        .reset_index()
    )

    x = stats[distance].astype(float).values
    y_mean = stats["Mean"].astype(float).values
    y_lower = (stats["Mean"] - stats["Confidence Interval"]).astype(float).values
    y_upper = (stats["Mean"] + stats["Confidence Interval"]).astype(float).values

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_mean, label="Mean Fitness", marker="o", linestyle="-")
    plt.fill_between(
        x, y_lower, y_upper, alpha=0.3, label=f"{int(confidence_level * 100)}% CI"
    )
    plt.xlabel(distance.replace("_", " ").title())
    plt.ylabel(fitness.replace("_", " ").title())
    plt.legend()
    plt.grid()
    plt.show()


def draw_fitness_dist(
    fitness: str,
    type: str = "hist",
    bins: int = 50,
    color: str = "skyblue",
    edgecolor: str = "black",
    figsize: tuple = (5, 4),
    log: bool = False
) -> None:
    """
    Plot the distribution of fitness values in the dataset.

    Parameters
    ----------
    fitness : list or pd.Series or np.ndarray
        The fitness values to plot.

    type : str = {'hist', 'cdf'}, default='hist'
        The type of plot to display. 'hist' for histogram, 'cdf' for cumulative distribution.

    bins : int, default=50
        The number of bins to use for the histogram.

    color : str, default='skyblue'
        The color of the bars in the histogram.

    edgecolor : str, default='black'
        The color of the edges of the bars in the histogram.

    figsize : tuple, default=(5, 4)
        The size of the plot figure.

    log : bool, default=False
        If True, display both axes of the CDF plot on a logarithmic scale.

    Returns
    -------
    None
        Displays a histogram or CDF of the fitness values in the dataset.
    """
    if not isinstance(fitness, pd.Series):
        fitness = pd.Series(fitness)

    if type == "hist":
        plt.figure(figsize=figsize)
        plt.hist(fitness, bins=bins, color=color, edgecolor=edgecolor)
        plt.xlabel("Fitness")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Fitness Values")
        plt.grid(axis="y", alpha=0.75)
        plt.show()

    elif type == "cdf":
        cdf = fitness.value_counts().sort_index().cumsum() / len(fitness)
        plt.figure(figsize=figsize)
        cdf.plot()
        plt.title("Cumulative Distribution of Fitness Values")
        plt.xlabel("Fitness")
        plt.ylabel("Cumulative Proportion")
        if log:
            plt.xscale("log")
            plt.yscale("log")
        plt.grid()
        plt.show()


def draw_diminishing_return(
    landscape: object,
    sample: int = 10000,
    color: str = "skyblue",
    figsize: tuple = (5, 4),
) -> None:
    """
    Plot the relationship between fitness effects of each mutation and the background fitness
    under which the mutation occurs. This would usually lead to the so-called "diminishing-return"
    pattern observed in evolutionary biology. 

    Parameters
    ----------
    landscape : object
        The landscape object containing the graph structure with fitness and delta fitness values.

    sample : int, default=10000
        The number of data points to sample for plotting.

    color : str, default='skyblue'
        The color of the scatter plot points.

    figsize : tuple, default=(5, 4)
        The size of the plot figure.

    Returns
    -------
    None
        Displays a scatter plot of fitness versus delta fitness.
    """
    sample = landscape.n_configs if landscape.n_configs < sample else sample

    logger = []

    for u, v in landscape.graph.edges():
        fitness = landscape.graph.nodes[u]['fitness']
        delta_fit = landscape.graph.edges[u, v]['delta_fit']
        logger.append((fitness, delta_fit))

    df = pd.DataFrame(logger, columns=["fitness", "delta_fit"]).sample(sample)

    plt.figure(figsize=figsize)
    plt.scatter(df['fitness'], df['delta_fit'], color=color, alpha=0.8)
    plt.title('Delta Fitness vs Fitness')
    plt.xlabel('Fitness')
    plt.ylabel('Delta Fitness')
    plt.grid(True)
    plt.show()


def draw_ffi(
    landscape: object,
    figsize: tuple = (5, 4),
    min_len: int = 3
) -> None:
    """
    Plot all accessible mutational pathways to the global optimum.

    Parameters
    ----------
    landscape : object
        The landscape object containing the graph structure with fitness and delta fitness values.

    min_len : int, default=3
        Minimum length of an adaptive path for it to be considered.
    
    figsize : tuple, default=(5, 4)
        The size of the plot figure.

    Returns
    -------
    None
        Displays a plot where each grey line indicates an accessible mutation path to the global
        optimum, along with an averaged fitness line. 
    """

    data = landscape.get_data()
    idx_to_fitness = data["fitness"].to_dict()
    fitness_list = []

    for i in data.index:
        lo, _, trace = hill_climb(
            landscape.graph, i, "delta_fit", verbose=0, return_trace=True,
        )
        if len(trace) >= min_len and lo == landscape.go_index:
            fitnesses = list([idx_to_fitness[config] for config in trace])
            fitnesses.reverse()
            fitness_list.append(fitnesses)

    path_data = pd.DataFrame(fitness_list).T
    data_array = path_data.values
    mean_values = np.nanmean(data_array, axis=1)
    x_values = np.arange(data_array.shape[0])

    plt.figure(figsize=figsize)

    for column in path_data.columns:
        plt.plot(x_values, path_data[column], color='lightgrey', alpha=0.3)

    plt.plot(x_values, mean_values, color='blue', label='Mean')

    plt.xlabel('Index')
    plt.legend()
    plt.grid()
    plt.show()
