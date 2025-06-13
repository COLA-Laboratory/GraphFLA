import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
from matplotlib import colors


def draw_diminishing_return(
    landscape,
    figsize=(7, 6),
    alpha=1,
    color_scheme="viridis",
    show_regression=True,
    show_stats=True,
    save_path=None,
    dpi=300,
    gridsize=50,  # New parameter for hexbin grid size
):
    """
    Visualizes diminishing returns epistasis by plotting fitness improvement vs background fitness.

    Creates a hexbin plot showing the relationship between node fitness (x-axis) and
    individual fitness improvements from successors (y-axis), with optional linear fit line
    and statistical information.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    alpha : float, default=0.6
        Transparency level for hexbin plot (0-1).
    color_scheme : str, default='viridis'
        Matplotlib colormap name for hexbin coloring.
    show_regression : bool, default=True
        Whether to show the linear fit line.
    show_stats : bool, default=True
        Whether to display regression statistics on the plot.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.
    gridsize : int, default=30
        Number of hexagons in the x-direction for hexbin plot.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If insufficient data is available for plotting.
    """

    # Validate landscape
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Extract data - collect all raw successor improvements
    node_fitnesses = []
    successor_improvements = []

    for v in landscape.graph.vs:
        current_fitness = v["fitness"]
        successors = v.successors()
        if successors:
            for s in successors:
                improvement = s["fitness"] - current_fitness
                if improvement > 0:  # Only positive improvements
                    node_fitnesses.append(current_fitness)
                    successor_improvements.append(improvement)

    # Convert to arrays
    x_data = np.array(node_fitnesses)
    y_data = np.array(successor_improvements)

    if len(x_data) < 2:
        raise ValueError("Not enough valid data points for visualization.")

    # Set up the plot with beautiful styling
    plt.style.use("default")
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Create hexbin plot
    hexbin = ax.hexbin(
        x_data,
        y_data,
        gridsize=gridsize,
        cmap=color_scheme,
        alpha=alpha,
        mincnt=1,  # Minimum count to display a hexagon
        edgecolors="white",
        linewidths=0.2,
        norm=colors.LogNorm(),
    )

    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label("Count", fontsize=12)

    # Calculate and plot linear fit line if requested
    slope = np.nan
    intercept = np.nan
    r_squared = np.nan
    p_value = np.nan

    if show_regression and len(x_data) > 1:
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x_data, y_data
            )
            r_squared = r_value**2

            # Plot linear fit line
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                "r-",
                linewidth=2,
                label=f"Linear fit (slope={slope:.3f})",
            )

        except Exception as e:
            print(f"Warning: Could not calculate linear fit: {e}")

    # Add statistical information if requested
    if show_stats and not np.isnan(slope):
        stats_text = f"Slope: {slope:.3f}\nR²: {r_squared:.3f}\np-value: {p_value:.3e}\nn = {len(x_data)}"

        # Position text box in upper right if negative slope, upper left if positive
        if slope < 0:
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=10,
            )
        else:
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=10,
            )

    # Customize plot appearance
    ax.set_xlabel("Background Fitness", fontsize=14)
    ax.set_ylabel("Fitness Improvement", fontsize=14)
    ax.set_title("Diminishing Returns Epistasis", fontsize=16, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Improve tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    return None


def draw_fitness_distance_corr(
    landscape,
    figsize=(10, 6),
    show_std=True,
    alpha_std=0.3,
    color="steelblue",
    std_color=None,
    save_path=None,
    dpi=300,
):
    """
    Visualizes fitness-distance correlation by plotting average fitness vs distance to global optimum.

    Creates a line plot showing the relationship between distance to global optimum (x-axis) and
    average fitness (y-axis), with optional standard deviation bands.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    show_std : bool, default=True
        Whether to show standard deviation bands around the mean.
    alpha_std : float, default=0.3
        Transparency level for standard deviation bands (0-1).
    color : str, default='steelblue'
        Color for the main line plot.
    std_color : str, optional
        Color for standard deviation bands. If None, uses same color as main line.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built or distance to global optimum cannot be calculated.
    ValueError
        If insufficient data is available for plotting.
    """

    # Validate landscape
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Check if the landscape has dist_go calculated
    if "dist_go" not in landscape.graph.vs.attributes():
        # If dist_go is not available, calculate it
        landscape.determine_dist_to_go()

        # Check again in case calculation failed
        if "dist_go" not in landscape.graph.vs.attributes():
            raise RuntimeError(
                "Could not calculate distance to global optimum. Make sure the landscape "
                "has proper configuration data and a valid global optimum."
            )

    # Get the data
    data = landscape.get_data()
    distances = data["dist_go"]
    fitnesses = data["fitness"]

    if len(distances) < 2:
        raise ValueError("Not enough valid data points for visualization.")

    # Calculate average fitness and std for each distance
    unique_distances = np.sort(np.unique(distances))
    mean_fitness = []
    std_fitness = []

    for dist in unique_distances:
        fitness_at_dist = fitnesses[distances == dist]
        mean_fitness.append(np.mean(fitness_at_dist))
        std_fitness.append(np.std(fitness_at_dist))

    mean_fitness = np.array(mean_fitness)
    std_fitness = np.array(std_fitness)

    # Set up the plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Set std_color if not provided
    if std_color is None:
        std_color = color

    # Plot the main line
    ax.plot(
        unique_distances,
        mean_fitness,
        color=color,
        linewidth=2,
        marker="o",
        markersize=4,
        label="Mean fitness",
    )

    # Add standard deviation bands if requested
    if show_std:
        ax.fill_between(
            unique_distances,
            mean_fitness - std_fitness,
            mean_fitness + std_fitness,
            alpha=alpha_std,
            color=std_color,
            label="± 1 std",
        )

    # Customize plot appearance
    ax.set_xlabel("Distance to Global Optimum", fontsize=14)
    ax.set_ylabel("Fitness", fontsize=14)
    ax.set_title("Fitness-Distance Correlation", fontsize=16, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Improve tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Add legend
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    return fig, ax


def draw_neighbor_fit_corr(
    landscape,
    figsize=(7, 6),
    alpha=1,
    color_scheme="viridis",
    show_regression=True,
    show_stats=True,
    save_path=None,
    dpi=300,
    gridsize=50,
    auto_calculate=True,
):
    """
    Visualizes neighbor fitness correlation by plotting node fitness vs mean neighbor fitness.

    Creates a hexbin plot showing the relationship between node fitness (x-axis) and
    mean neighbor fitness (y-axis), with optional linear fit line and statistical information.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object.
    figsize : tuple, default=(7, 6)
        Figure size as (width, height) in inches.
    alpha : float, default=1
        Transparency level for hexbin plot (0-1).
    color_scheme : str, default='viridis'
        Matplotlib colormap name for hexbin coloring.
    show_regression : bool, default=True
        Whether to show the linear fit line.
    show_stats : bool, default=True
        Whether to display regression statistics on the plot.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.
    gridsize : int, default=50
        Number of hexagons in the x-direction for hexbin plot.
    auto_calculate : bool, default=True
        If True, automatically runs determine_neighbor_fitness() if needed.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built or neighbor fitness cannot be calculated.
    ValueError
        If insufficient data is available for plotting.
    """

    # Validate landscape
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Check if neighbor fitness has been calculated
    if "mean_neighbor_fit" not in landscape.graph.vs.attributes():
        if auto_calculate:
            landscape.determine_neighbor_fitness()
        else:
            raise RuntimeError(
                "Neighbor fitness metrics haven't been calculated. "
                "Either call landscape.determine_neighbor_fitness() first "
                "or set auto_calculate=True."
            )

        # Check again in case calculation failed
        if "mean_neighbor_fit" not in landscape.graph.vs.attributes():
            raise RuntimeError(
                "Could not calculate neighbor fitness metrics. Make sure the landscape "
                "has proper graph structure."
            )

    # Extract fitness and mean neighbor fitness values
    fitness_values = np.array(landscape.graph.vs["fitness"])
    neighbor_fitness_values = np.array(landscape.graph.vs["mean_neighbor_fit"])

    # Remove NaN values (nodes with no neighbors)
    valid_mask = ~np.isnan(neighbor_fitness_values)
    x_data = fitness_values[valid_mask]
    y_data = neighbor_fitness_values[valid_mask]

    if len(x_data) < 2:
        raise ValueError("Not enough valid data points for visualization.")

    # Set up the plot with beautiful styling
    plt.style.use("default")
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Create hexbin plot
    hexbin = ax.hexbin(
        x_data,
        y_data,
        gridsize=gridsize,
        cmap=color_scheme,
        alpha=alpha,
        mincnt=1,  # Minimum count to display a hexagon
        edgecolors="white",
        linewidths=0.2,
        norm=colors.LogNorm(),
    )

    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label("Count", fontsize=12)

    # Calculate and plot linear fit line if requested
    slope = np.nan
    intercept = np.nan
    r_squared = np.nan
    p_value = np.nan

    if show_regression and len(x_data) > 1:
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x_data, y_data
            )
            r_squared = r_value**2

            # Plot linear fit line
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                "r-",
                linewidth=2,
                label=f"Linear fit (slope={slope:.3f})",
            )

        except Exception as e:
            print(f"Warning: Could not calculate linear fit: {e}")

    # Add statistical information if requested
    if show_stats and not np.isnan(slope):
        stats_text = f"Slope: {slope:.3f}\nR²: {r_squared:.3f}\np-value: {p_value:.3e}\nn = {len(x_data)}"

        # Position text box in upper right if negative slope, upper left if positive
        if slope < 0:
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=10,
            )
        else:
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=10,
            )

    # Customize plot appearance
    ax.set_xlabel("Node Fitness", fontsize=14)
    ax.set_ylabel("Mean Neighbor Fitness", fontsize=14)
    ax.set_title("Neighbor Fitness Correlation", fontsize=16, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Improve tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    return fig, ax


def draw_adaptive_walk(
    landscape,
    n_walks=50,
    walk_type="best-improvement",
    figsize=(8, 6),
    alpha_walks=0.3,
    color_walks="steelblue",
    color_mean="red",
    linewidth_walks=1,
    linewidth_mean=3,
    show_mean=True,
    save_path=None,
    dpi=300,
    verbose=0,
):
    """
    Visualizes adaptive walks on the fitness landscape by plotting fitness vs step number.

    Creates line plots for individual adaptive walks and optionally shows the mean fitness trajectory.
    The x-axis shows the step number in the walk and y-axis shows fitness.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object.
    n_walks : int or float, default=50
        Number of walks to perform. If int, exact number of walks.
        If float (0-1), fraction of total configurations to sample.
    walk_type : str, default="best-improvement"
        Type of adaptive walk. Options: "best-improvement", "first-improvement".
    figsize : tuple, default=(12, 8)
        Figure size as (width, height) in inches.
    alpha_walks : float, default=0.3
        Transparency level for individual walk lines (0-1).
    color_walks : str, default="steelblue"
        Color for individual walk lines.
    color_mean : str, default="red"
        Color for the mean fitness line.
    linewidth_walks : float, default=1
        Line width for individual walks.
    linewidth_mean : float, default=3
        Line width for mean fitness line.
    show_mean : bool, default=True
        Whether to show the mean fitness trajectory.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.
    verbose : int, default=0
        Verbosity level for walk progress.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If invalid parameters are provided.
    """

    # Validate landscape
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Validate and process n_walks parameter
    total_nodes = len(landscape.graph.vs)
    if isinstance(n_walks, float):
        if not 0 < n_walks <= 1:
            raise ValueError("n_walks as float must be between 0 and 1")
        n_walks = max(1, int(n_walks * total_nodes))
    elif isinstance(n_walks, int):
        if not 0 < n_walks <= total_nodes:
            raise ValueError(f"n_walks as int must be between 1 and {total_nodes}")
    else:
        raise ValueError("n_walks must be int or float")

    # Validate walk_type
    if walk_type not in ["best-improvement", "first-improvement"]:
        raise ValueError("walk_type must be 'best-improvement' or 'first-improvement'")

    # Import hill_climb function
    from ..algorithms.adaptive_walk import hill_climb

    # Sample starting nodes randomly
    starting_nodes = random.sample(range(total_nodes), n_walks)

    # Store all walk data
    all_walks = []

    if verbose > 0:
        print(f"Performing {n_walks} adaptive walks...")

    # Perform adaptive walks
    for i, start_node in enumerate(starting_nodes):
        if verbose > 0 and (i + 1) % max(1, n_walks // 10) == 0:
            print(f"Completed {i + 1}/{n_walks} walks")

        # Perform hill climb with trace
        final_node, steps, trace = hill_climb(
            landscape.graph,
            start_node,
            "fitness",  # weight parameter (though not directly used in our implementation)
            verbose=0,  # Keep individual walks quiet
            return_trace=True,
            search_method=walk_type,
        )

        # Extract fitness data for this walk
        walk_fitness = [landscape.graph.vs[node]["fitness"] for node in trace]
        # Create step numbers starting from 1
        walk_steps = list(range(1, len(trace) + 1))

        all_walks.append({"steps": walk_steps, "fitness": walk_fitness, "trace": trace})

    # Set up the plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Plot individual walks
    for i, walk in enumerate(all_walks):
        ax.plot(
            walk["steps"],
            walk["fitness"],
            color=color_walks,
            alpha=alpha_walks,
            linewidth=linewidth_walks,
            label="Individual walks" if i == 0 else "",
        )

    # Calculate and plot mean fitness trajectory if requested
    if show_mean and len(all_walks) > 0:
        # Find the maximum number of steps across all walks
        max_steps = max(len(walk["steps"]) for walk in all_walks)

        # Calculate mean fitness at each step
        mean_fitness_at_step = []
        for step in range(1, max_steps + 1):
            fitness_values = []
            for walk in all_walks:
                if step <= len(walk["steps"]):
                    fitness_values.append(
                        walk["fitness"][step - 1]
                    )  # step-1 for 0-indexed list

            if fitness_values:
                mean_fitness_at_step.append(np.mean(fitness_values))
            else:
                mean_fitness_at_step.append(np.nan)

        # Create step numbers for mean line
        step_numbers = list(range(1, max_steps + 1))

        # Remove NaN values
        valid_indices = [
            i for i, val in enumerate(mean_fitness_at_step) if not np.isnan(val)
        ]
        valid_steps = [step_numbers[i] for i in valid_indices]
        valid_mean_fitness = [mean_fitness_at_step[i] for i in valid_indices]

        if len(valid_steps) > 0:
            ax.plot(
                valid_steps,
                valid_mean_fitness,
                color=color_mean,
                linewidth=linewidth_mean,
                label="Mean fitness",
                zorder=10,  # Ensure mean line is on top
            )

    # Customize plot appearance
    ax.set_xlabel("Step Number", fontsize=14)
    ax.set_ylabel("Fitness", fontsize=14)
    ax.set_title(f"Adaptive Walks ({walk_type}, n={n_walks})", fontsize=16, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Improve tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels, frameon=True, fancybox=True, shadow=True, fontsize=10
        )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    return fig, ax


def draw_global_epistasis(
    landscape,
    figsize=(8, 6),
    alpha=1,
    color_scheme="viridis",
    show_nonlinear_fit=True,
    show_binned_mean=True,
    save_path=None,
    dpi=300,
    gridsize=50,
    n_bins=25,
):
    """
    Visualizes global epistasis by plotting fitness vs sum of linear predictors.

    Creates a hexbin plot showing the relationship between sum of first-order linear
    predictors (x-axis) and fitness values (y-axis), with optional nonlinear fit line
    and binned mean line to reveal global epistatic patterns.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object.
    figsize : tuple, default=(8, 6)
        Figure size as (width, height) in inches.
    alpha : float, default=1
        Transparency level for hexbin plot (0-1).
    color_scheme : str, default='viridis'
        Matplotlib colormap name for hexbin coloring.
    show_nonlinear_fit : bool, default=True
        Whether to show the nonlinear fit line (using polynomial regression).
    show_binned_mean : bool, default=True
        Whether to show binned mean fitness line.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.
    gridsize : int, default=50
        Number of hexagons in the x-direction for hexbin plot.
    n_bins : int, default=25
        Number of bins for calculating binned mean fitness.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If insufficient data is available for plotting.
    ImportError
        If required dependencies (sklearn) are not available.
    """

    # Validate landscape
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Import required libraries
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import (
            PolynomialFeatures,
            LabelEncoder,
            OneHotEncoder,
        )
        from sklearn.pipeline import Pipeline
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Global epistasis plotting requires scikit-learn and pandas. "
            "Please install them with: pip install scikit-learn pandas"
        )

    # Get the data
    data = landscape.get_data()

    # Extract genotype data (exclude non-genotype columns)
    exclude_cols = [
        "fitness",
        "in_degree",
        "out_degree",
        "pagerank",
        "is_lo",
        "basin_index",
        "size_basin_greedy",
        "radius_basin_greedy",
        "size_basin_accessible",
        "mean_neighbor_fit",
        "dist_go",
    ]

    # Get genotype columns (should be the variable columns)
    genotype_cols = [col for col in data.columns if col not in exclude_cols]

    if len(genotype_cols) == 0:
        raise ValueError("No genotype columns found in the data.")

    # Extract genotype data and fitness values
    X_genotype_raw = data[genotype_cols]
    y_fitness = data["fitness"].values

    if len(X_genotype_raw) < 2:
        raise ValueError("Not enough valid data points for visualization.")

    # Check data type and encode appropriately
    if landscape.verbose:
        print("Processing genotype data...")

    # Determine if we need categorical encoding
    needs_encoding = False
    for col in genotype_cols:
        # Check if column contains non-numeric data or if landscape data_type indicates categorical
        col_data = X_genotype_raw[col]
        if (
            hasattr(landscape, "data_type")
            and landscape.data_type in ["categorical", "sequence"]
        ) or not pd.api.types.is_numeric_dtype(col_data):
            needs_encoding = True
            break

    if needs_encoding:
        if landscape.verbose:
            print("Detected categorical data, applying one-hot encoding...")

        # Use pandas get_dummies for one-hot encoding
        X_encoded = pd.get_dummies(X_genotype_raw, prefix=genotype_cols)
        X_genotype = X_encoded.values

        if landscape.verbose:
            print(
                f"Encoded {len(genotype_cols)} categorical columns into {X_genotype.shape[1]} binary features"
            )

    else:
        # Data is already numeric (boolean/binary)
        X_genotype = X_genotype_raw.values

        if landscape.verbose:
            print(f"Using {X_genotype.shape[1]} numeric features directly")

    # Fit first-order linear model to get additive effects
    if landscape.verbose:
        print("Fitting first-order linear model...")

    linear_model = LinearRegression()
    linear_model.fit(X_genotype, y_fitness)

    # Calculate sum of linear predictors (phi)
    phi = linear_model.predict(X_genotype)

    # Set up the plot
    plt.style.use("default")
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Create hexbin plot
    hexbin = ax.hexbin(
        phi,
        y_fitness,
        gridsize=gridsize,
        cmap=color_scheme,
        alpha=alpha,
        mincnt=1,
        edgecolors="white",
        linewidths=0.2,
        norm=colors.LogNorm(),
    )

    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label("Count", fontsize=12)

    # Add binned mean line if requested
    if show_binned_mean:
        try:
            # Calculate binned statistics
            from scipy import stats as scipy_stats

            # Create bins
            bin_means, bin_edges, _ = scipy_stats.binned_statistic(
                phi, y_fitness, statistic="mean", bins=n_bins
            )

            # Calculate bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Remove NaN values
            valid_mask = ~np.isnan(bin_means)
            valid_centers = bin_centers[valid_mask]
            valid_means = bin_means[valid_mask]

            if len(valid_centers) > 0:
                ax.plot(
                    valid_centers,
                    valid_means,
                    color="black",
                    linewidth=2,
                    linestyle="--",
                    label=f"Binned mean (n={n_bins})",
                    alpha=0.8,
                )

        except Exception as e:
            print(f"Warning: Could not calculate binned mean: {e}")

    # Add nonlinear fit line if requested
    if show_nonlinear_fit:
        try:
            # Fit polynomial regression (degree 3) as approximation to spline
            poly_model = Pipeline(
                [("poly", PolynomialFeatures(degree=3)), ("linear", LinearRegression())]
            )

            # Reshape phi for sklearn
            phi_reshaped = phi.reshape(-1, 1)
            poly_model.fit(phi_reshaped, y_fitness)

            # Generate smooth line for plotting
            phi_range = np.linspace(phi.min(), phi.max(), 200)
            phi_range_reshaped = phi_range.reshape(-1, 1)
            y_nonlinear = poly_model.predict(phi_range_reshaped)

            ax.plot(
                phi_range,
                y_nonlinear,
                color="blue",
                linewidth=2,
                label="Nonlinear fit",
                alpha=0.9,
            )

        except Exception as e:
            print(f"Warning: Could not calculate nonlinear fit: {e}")

    # Calculate R-squared for linear model
    try:
        r_squared = linear_model.score(X_genotype, y_fitness)

        # Add statistics text
        n_features = X_genotype.shape[1]
        stats_text = (
            f"Linear R²: {r_squared:.3f}\nFeatures: {n_features}\nn = {len(y_fitness)}"
        )

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )
    except Exception as e:
        print(f"Warning: Could not calculate R-squared: {e}")

    # Customize plot appearance
    ax.set_xlabel("Sum of Linear Predictors", fontsize=14)
    ax.set_ylabel("Fitness", fontsize=14)
    ax.set_title("Global Epistasis", fontsize=16, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Improve tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Add legend if any lines were plotted
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels, frameon=True, fancybox=True, shadow=True, fontsize=10
        )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    return fig, ax


def draw_fitness_distribution(
    landscape,
    figsize=(8, 6),
    bins=30,
    color="steelblue",
    alpha=0.7,
    edgecolor="white",
    linewidth=0.5,
    show_stats=True,
    show_kde=False,
    kde_color="red",
    save_path=None,
    dpi=300,
):
    """
    Visualizes the fitness distribution of the landscape using a histogram.

    Creates a histogram showing the distribution of fitness values across all nodes
    in the landscape, with optional statistical information and kernel density estimation.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object.
    figsize : tuple, default=(8, 6)
        Figure size as (width, height) in inches.
    bins : int or str, default=30
        Number of histogram bins or binning strategy (e.g., 'auto', 'sqrt').
    color : str, default='steelblue'
        Color for the histogram bars.
    alpha : float, default=0.7
        Transparency level for histogram bars (0-1).
    edgecolor : str, default='white'
        Color for the edges of histogram bars.
    linewidth : float, default=0.5
        Width of the edges of histogram bars.
    show_stats : bool, default=True
        Whether to display statistical information on the plot.
    show_kde : bool, default=False
        Whether to overlay a kernel density estimation curve.
    kde_color : str, default='red'
        Color for the KDE curve.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If insufficient data is available for plotting.
    """

    # Validate landscape
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Extract fitness values
    fitness_values = np.array(landscape.graph.vs["fitness"])

    if len(fitness_values) < 2:
        raise ValueError("Not enough fitness data points for visualization.")

    # Set up the plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Create histogram
    n, bins_edges, patches = ax.hist(
        fitness_values,
        bins=bins,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        density=show_kde,  # Use density if showing KDE for proper scaling
    )

    # Add KDE curve if requested
    if show_kde:
        try:
            from scipy.stats import gaussian_kde

            # Create KDE
            kde = gaussian_kde(fitness_values)

            # Generate smooth x values for KDE curve
            x_kde = np.linspace(fitness_values.min(), fitness_values.max(), 200)
            y_kde = kde(x_kde)

            ax.plot(
                x_kde,
                y_kde,
                color=kde_color,
                linewidth=2,
                label="Kernel Density Estimate",
                alpha=0.9,
            )

        except ImportError:
            print("Warning: scipy required for KDE. Install with: pip install scipy")
        except Exception as e:
            print(f"Warning: Could not calculate KDE: {e}")

    # Calculate and display statistics if requested
    if show_stats:
        try:
            mean_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)
            median_fitness = np.median(fitness_values)
            min_fitness = np.min(fitness_values)
            max_fitness = np.max(fitness_values)

            # Calculate skewness and kurtosis
            from scipy import stats as scipy_stats

            skewness = scipy_stats.skew(fitness_values)
            kurtosis = scipy_stats.kurtosis(fitness_values)

            stats_text = (
                f"Mean: {mean_fitness:.3f}\n"
                f"Std: {std_fitness:.3f}\n"
                f"Median: {median_fitness:.3f}\n"
                f"Min: {min_fitness:.3f}\n"
                f"Max: {max_fitness:.3f}\n"
                f"Skewness: {skewness:.3f}\n"
                f"Kurtosis: {kurtosis:.3f}\n"
                f"n = {len(fitness_values)}"
            )

            # Position stats box in upper right
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=9,
            )

        except ImportError:
            # Basic stats without scipy
            mean_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)
            median_fitness = np.median(fitness_values)
            min_fitness = np.min(fitness_values)
            max_fitness = np.max(fitness_values)

            stats_text = (
                f"Mean: {mean_fitness:.3f}\n"
                f"Std: {std_fitness:.3f}\n"
                f"Median: {median_fitness:.3f}\n"
                f"Min: {min_fitness:.3f}\n"
                f"Max: {max_fitness:.3f}\n"
                f"n = {len(fitness_values)}"
            )

            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=9,
            )

        except Exception as e:
            print(f"Warning: Could not calculate statistics: {e}")

    # Add vertical lines for mean and median
    if show_stats:
        try:
            mean_fitness = np.mean(fitness_values)
            median_fitness = np.median(fitness_values)

            # Add mean line
            ax.axvline(
                mean_fitness,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Mean ({mean_fitness:.3f})",
            )

            # Add median line (only if different from mean)
            if abs(mean_fitness - median_fitness) > 0.001:
                ax.axvline(
                    median_fitness,
                    color="orange",
                    linestyle="-.",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Median ({median_fitness:.3f})",
                )

        except Exception as e:
            print(f"Warning: Could not add reference lines: {e}")

    # Customize plot appearance
    ax.set_xlabel("Fitness", fontsize=14)

    if show_kde:
        ax.set_ylabel("Density", fontsize=14)
    else:
        ax.set_ylabel("Frequency", fontsize=14)

    ax.set_title("Fitness Distribution", fontsize=16, pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Improve tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Add legend if there are labeled elements
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels, frameon=True, fancybox=True, shadow=True, fontsize=10
        )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    return fig, ax


def draw_fitness_effects(
    landscape,
    mutations,
    figsize=(8, 6),
    bins=30,
    color="steelblue",
    alpha=0.7,
    edgecolor="white",
    linewidth=0.5,
    show_stats=True,
    show_kde=False,
    kde_color="red",
    save_path=None,
    dpi=300,
):
    """
    Visualizes the distribution of fitness effects (DFE) of specified mutation(s) across genetic backgrounds.

    Creates a histogram showing the distribution of fitness effects when applying the specified
    mutation(s) across all possible genetic backgrounds in the landscape.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object.
    mutations : tuple or list of tuples
        Mutation(s) to analyze. Each mutation is a tuple of (original_allele, position, new_allele).
        Position must be a column in the landscape data. Both alleles must exist in the data.
        Multiple mutations should be provided as a list: [(orig1, pos1, new1), (orig2, pos2, new2), ...]
    figsize : tuple, default=(8, 6)
        Figure size as (width, height) in inches.
    bins : int or str, default=30
        Number of histogram bins or binning strategy (e.g., 'auto', 'sqrt').
    color : str, default='steelblue'
        Color for the histogram bars.
    alpha : float, default=0.7
        Transparency level for histogram bars (0-1).
    edgecolor : str, default='white'
        Color for the edges of histogram bars.
    linewidth : float, default=0.5
        Width of the edges of histogram bars.
    show_stats : bool, default=True
        Whether to display statistical information on the plot.
    show_kde : bool, default=False
        Whether to overlay a kernel density estimation curve.
    kde_color : str, default='red'
        Color for the KDE curve.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built.
    ValueError
        If invalid mutations or insufficient data is provided.
    """

    # Validate landscape
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Get the data
    data = landscape.get_data()

    # Normalize mutations input to always be a list of tuples
    if isinstance(mutations, tuple):
        mutations = [mutations]
    elif isinstance(mutations, list):
        if not all(isinstance(mut, tuple) and len(mut) == 3 for mut in mutations):
            raise ValueError(
                "All mutations must be tuples of format (original_allele, position, new_allele)"
            )
    else:
        raise ValueError(
            "mutations must be a tuple (original_allele, position, new_allele) "
            "or a list of such tuples"
        )

    # Validate mutations
    for original_allele, position, new_allele in mutations:
        if position not in data.columns:
            raise ValueError(
                f"Position '{position}' not found in landscape data columns"
            )

        unique_alleles = data[position].unique()
        if original_allele not in unique_alleles:
            raise ValueError(
                f"Original allele '{original_allele}' not found in column '{position}'. "
                f"Available alleles: {unique_alleles}"
            )
        if new_allele not in unique_alleles:
            raise ValueError(
                f"New allele '{new_allele}' not found in column '{position}'. "
                f"Available alleles: {unique_alleles}"
            )

    # Check for mutations at the same position
    positions = [mut[1] for mut in mutations]
    if len(positions) != len(set(positions)):
        raise ValueError("Multiple mutations cannot occur at the same position")

    # Create filters for original and new genotypes
    original_filter = True
    new_filter = True

    # Build filters for all mutations
    for original_allele, position, new_allele in mutations:
        original_filter = original_filter & (data[position] == original_allele)
        new_filter = new_filter & (data[position] == new_allele)

    # Filter data to get original and new genotype sets
    original_data = data[original_filter].copy()
    new_data = data[new_filter].copy()

    if len(original_data) == 0:
        raise ValueError("No genotypes found with the specified original alleles")
    if len(new_data) == 0:
        raise ValueError("No genotypes found with the specified new alleles")

    # Identify background columns (all columns except the mutation positions and fitness)
    exclude_cols = (
        ["fitness"] + positions + ["in_degree", "out_degree", "pagerank", "is_lo"]
    )
    background_cols = [col for col in data.columns if col not in exclude_cols]

    if len(background_cols) == 0:
        raise ValueError("No background columns found for comparison")

    # Align dataframes by common genetic backgrounds
    original_backgrounds = original_data[background_cols]
    new_backgrounds = new_data[background_cols]

    # Find common backgrounds using merge
    # First add fitness columns with suffixes to track them
    original_with_fit = original_data[background_cols + ["fitness"]].copy()
    original_with_fit = original_with_fit.rename(
        columns={"fitness": "fitness_original"}
    )

    new_with_fit = new_data[background_cols + ["fitness"]].copy()
    new_with_fit = new_with_fit.rename(columns={"fitness": "fitness_new"})

    # Merge on background columns to find common backgrounds
    aligned_data = original_with_fit.merge(
        new_with_fit, on=background_cols, how="inner"
    )

    if len(aligned_data) == 0:
        raise ValueError(
            "No common genetic backgrounds found between original and new allele genotypes"
        )

    # Calculate fitness effects (new fitness - original fitness)
    fitness_effects = aligned_data["fitness_new"] - aligned_data["fitness_original"]

    if len(fitness_effects) < 2:
        raise ValueError(
            "Not enough common backgrounds for meaningful fitness effects analysis"
        )

    # Set up the plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Create histogram
    n, bins_edges, patches = ax.hist(
        fitness_effects,
        bins=bins,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        density=show_kde,  # Use density if showing KDE for proper scaling
    )

    # Add KDE curve if requested
    if show_kde:
        try:
            from scipy.stats import gaussian_kde

            # Create KDE
            kde = gaussian_kde(fitness_effects)

            # Generate smooth x values for KDE curve
            x_kde = np.linspace(fitness_effects.min(), fitness_effects.max(), 200)
            y_kde = kde(x_kde)

            ax.plot(
                x_kde,
                y_kde,
                color=kde_color,
                linewidth=2,
                label="Kernel Density Estimate",
                alpha=0.9,
            )

        except ImportError:
            print("Warning: scipy required for KDE. Install with: pip install scipy")
        except Exception as e:
            print(f"Warning: Could not calculate KDE: {e}")

    # Calculate and display statistics if requested
    if show_stats:
        try:
            mean_effect = np.mean(fitness_effects)
            std_effect = np.std(fitness_effects)
            median_effect = np.median(fitness_effects)
            min_effect = np.min(fitness_effects)
            max_effect = np.max(fitness_effects)

            # Count beneficial, neutral, and deleterious mutations
            beneficial = np.sum(fitness_effects > 0)
            neutral = np.sum(fitness_effects == 0)
            deleterious = np.sum(fitness_effects < 0)

            # Calculate skewness and kurtosis if scipy available
            try:
                from scipy import stats as scipy_stats

                skewness = scipy_stats.skew(fitness_effects)
                kurtosis = scipy_stats.kurtosis(fitness_effects)

                stats_text = (
                    f"Mean: {mean_effect:.3f}\n"
                    f"Std: {std_effect:.3f}\n"
                    f"Median: {median_effect:.3f}\n"
                    f"Min: {min_effect:.3f}\n"
                    f"Max: {max_effect:.3f}\n"
                    f"Beneficial: {beneficial}\n"
                    f"Neutral: {neutral}\n"
                    f"Deleterious: {deleterious}\n"
                    f"Skewness: {skewness:.3f}\n"
                    f"Kurtosis: {kurtosis:.3f}\n"
                    f"n = {len(fitness_effects)}"
                )
            except ImportError:
                stats_text = (
                    f"Mean: {mean_effect:.3f}\n"
                    f"Std: {std_effect:.3f}\n"
                    f"Median: {median_effect:.3f}\n"
                    f"Min: {min_effect:.3f}\n"
                    f"Max: {max_effect:.3f}\n"
                    f"Beneficial: {beneficial}\n"
                    f"Neutral: {neutral}\n"
                    f"Deleterious: {deleterious}\n"
                    f"n = {len(fitness_effects)}"
                )

            # Position stats box in upper right or left depending on skew
            if mean_effect >= 0:
                ha = "right"
                x_pos = 0.95
            else:
                ha = "left"
                x_pos = 0.05

            ax.text(
                x_pos,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment=ha,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=9,
            )

        except Exception as e:
            print(f"Warning: Could not calculate statistics: {e}")

    # Add vertical lines for mean and zero effect
    if show_stats:
        try:
            mean_effect = np.mean(fitness_effects)

            # Add zero line (neutral)
            ax.axvline(
                0,
                color="black",
                linestyle="-",
                linewidth=1,
                alpha=0.8,
                label="Neutral (0)",
            )

            # Add mean line
            ax.axvline(
                mean_effect,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Mean ({mean_effect:.3f})",
            )

        except Exception as e:
            print(f"Warning: Could not add reference lines: {e}")

    # Create mutation description for title
    if len(mutations) == 1:
        original_allele, position, new_allele = mutations[0]
        mutation_desc = f"{original_allele}→{new_allele} at {position}"
    else:
        mutation_parts = []
        for original_allele, position, new_allele in mutations:
            mutation_parts.append(f"{original_allele}→{new_allele}@{position}")
        mutation_desc = ", ".join(mutation_parts)

    # Customize plot appearance
    ax.set_xlabel("Fitness Effect", fontsize=14)

    if show_kde:
        ax.set_ylabel("Density", fontsize=14)
    else:
        ax.set_ylabel("Frequency", fontsize=14)

    ax.set_title(
        f"Distribution of Fitness Effects\n({mutation_desc})", fontsize=16, pad=20
    )

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Improve tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Add legend if there are labeled elements
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels, frameon=True, fancybox=True, shadow=True, fontsize=10
        )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    return fig, ax


def draw_basin_fit_corr(
    landscape,
    figsize=(8, 6),
    alpha=0.6,
    color="steelblue",
    show_regression=True,
    show_confidence=True,
    confidence_alpha=0.2,
    method="spearman",
    save_path=None,
    dpi=300,
):
    """
    Visualizes the correlation between basin size and local optima fitness.

    Creates a scatter plot showing the relationship between local optima fitness (x-axis) and
    basin size (y-axis), with optional linear fit line and 95% confidence interval.

    Parameters
    ----------
    landscape : Landscape
        An initialized and built fitness landscape object.
    figsize : tuple, default=(8, 6)
        Figure size as (width, height) in inches.
    alpha : float, default=0.6
        Transparency level for scatter points (0-1).
    color : str, default='steelblue'
        Color for the scatter points and regression line.
    show_regression : bool, default=True
        Whether to show the linear fit line.
    show_confidence : bool, default=True
        Whether to show the 95% confidence interval around the regression line.
    confidence_alpha : float, default=0.2
        Transparency level for confidence interval (0-1).
    method : str, default='spearman'
        Correlation method to display in statistics ('spearman' or 'pearson').
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.

    Raises
    ------
    RuntimeError
        If the landscape object has not been built or basin data cannot be calculated.
    ValueError
        If insufficient data is available for plotting or invalid method specified.
    """

    # Validate landscape
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    # Validate method
    if method not in ["spearman", "pearson"]:
        raise ValueError(f"Invalid method '{method}'. Choose 'spearman' or 'pearson'.")

    # Check if basins have been calculated
    if "size_basin_greedy" not in landscape.graph.vs.attributes():
        # If basin sizes are not available, calculate them
        if landscape.verbose:
            print("Basin sizes not found. Calculating basins of attraction...")
        landscape.determine_basin_of_attraction()

        # Check again in case calculation failed
        if "size_basin_greedy" not in landscape.graph.vs.attributes():
            raise RuntimeError(
                "Could not calculate basin sizes. Make sure the landscape "
                "has a valid graph structure for basin calculation."
            )

    # Get local optima data
    lo_data = landscape.get_data(lo_only=True)

    if len(lo_data) < 2:
        raise ValueError("Not enough local optima for correlation analysis.")

    # Extract data for plotting
    fitness_values = lo_data["fitness"].values
    basin_sizes_greedy = lo_data["size_basin_greedy"].values

    # Check if accessible basin sizes are available
    has_accessible = "size_basin_accessible" in lo_data.columns
    if has_accessible:
        basin_sizes_accessible = lo_data["size_basin_accessible"].values

    # Set up the plot
    plt.style.use("default")

    # Create subplots - one or two depending on available data
    if has_accessible:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(figsize[0] * 2, figsize[1]), dpi=100
        )
        axes = [ax1, ax2]
        basin_data = [basin_sizes_greedy, basin_sizes_accessible]
        titles = ["Greedy Basin Size", "Accessible Basin Size"]
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        axes = [ax]
        basin_data = [basin_sizes_greedy]
        titles = ["Greedy Basin Size"]

    # Plot data for each basin type
    for idx, (current_ax, basin_sizes, title) in enumerate(
        zip(axes, basin_data, titles)
    ):

        # Create scatter plot
        current_ax.scatter(
            fitness_values,
            basin_sizes,
            alpha=alpha,
            color=color,
            s=50,
            edgecolors="white",
            linewidths=0.5,
        )

        # Calculate and plot regression line if requested
        slope = np.nan
        intercept = np.nan
        r_value = np.nan
        p_value = np.nan

        if show_regression and len(fitness_values) > 1:
            try:
                # Calculate linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    fitness_values, basin_sizes
                )

                # Plot regression line
                x_line = np.linspace(fitness_values.min(), fitness_values.max(), 100)
                y_line = slope * x_line + intercept
                current_ax.plot(
                    x_line,
                    y_line,
                    color="red",
                    linewidth=2,
                    label=f"Linear fit (R²={r_value**2:.3f})",
                    alpha=0.8,
                )

                # Add 95% confidence interval if requested
                if show_confidence:
                    try:
                        # Calculate confidence interval
                        n = len(fitness_values)
                        t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence interval

                        # Calculate standard error of prediction
                        x_mean = np.mean(fitness_values)
                        sxx = np.sum((fitness_values - x_mean) ** 2)
                        s_yx = np.sqrt(
                            np.sum(
                                (basin_sizes - (slope * fitness_values + intercept))
                                ** 2
                            )
                            / (n - 2)
                        )

                        # Standard error for each point on the line
                        se = s_yx * np.sqrt(1 / n + (x_line - x_mean) ** 2 / sxx)

                        # Confidence interval
                        ci = t_val * se

                        current_ax.fill_between(
                            x_line,
                            y_line - ci,
                            y_line + ci,
                            alpha=confidence_alpha,
                            color=color,
                            label="95% CI",
                        )

                    except Exception as e:
                        if landscape.verbose:
                            print(
                                f"Warning: Could not calculate confidence interval: {e}"
                            )

            except Exception as e:
                if landscape.verbose:
                    print(f"Warning: Could not calculate linear regression: {e}")

        # Calculate correlation coefficient
        try:
            from scipy.stats import spearmanr, pearsonr

            if method == "spearman":
                corr_coef, corr_p = spearmanr(fitness_values, basin_sizes)
                corr_name = "Spearman ρ"
            else:  # pearson
                corr_coef, corr_p = pearsonr(fitness_values, basin_sizes)
                corr_name = "Pearson r"

            # Add statistical information
            if not np.isnan(corr_coef):
                stats_text = (
                    f"{corr_name}: {corr_coef:.3f}\n"
                    f"p-value: {corr_p:.3e}\n"
                    f"n = {len(fitness_values)}"
                )

                if not np.isnan(slope):
                    stats_text += f"\nLinear R²: {r_value**2:.3f}"

                # Position text box
                current_ax.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=current_ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=10,
                )

        except ImportError:
            if landscape.verbose:
                print("Warning: scipy required for correlation calculation")
        except Exception as e:
            if landscape.verbose:
                print(f"Warning: Could not calculate correlation: {e}")

        # Customize plot appearance
        current_ax.set_xlabel("Local Optima Fitness", fontsize=14)
        current_ax.set_ylabel("Basin Size", fontsize=14)
        current_ax.set_title(
            f"Basin-Fitness Correlation\n({title})", fontsize=14, pad=15
        )

        # Add grid for better readability
        current_ax.grid(True, alpha=0.3, linestyle="--")

        # Improve tick labels
        current_ax.tick_params(axis="both", which="major", labelsize=11)

        # Add legend if there are labeled elements
        handles, labels = current_ax.get_legend_handles_labels()
        if handles:
            current_ax.legend(
                handles, labels, frameon=True, fancybox=True, shadow=True, fontsize=9
            )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Figure saved to: {save_path}")

    if has_accessible:
        return fig, (ax1, ax2)
    else:
        return fig, ax


def draw_pairwise_epistasis(landscape):
    raise NotImplementedError(
        "This function is not yet implemented. Please check back later or contribute to the codebase."
    )
