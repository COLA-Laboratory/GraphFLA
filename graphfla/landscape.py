import pandas as pd
import networkx as nx
import copy
import matplotlib.pyplot as plt
import umap.umap_ as umap
import palettable

from karateclub import HOPE
from typing import List, Any, Dict, Tuple
from itertools import product, combinations
from collections import defaultdict
from tqdm import tqdm

from ._base import BaseLandscape
from .lon import get_lon
from .algorithms import hill_climb
from .utils import add_network_metrics
from .distances import mixed_distance
from .metrics import *
from .visualization import *

import warnings

warnings.filterwarnings("ignore")


class Landscape(BaseLandscape):
    """
    Class implementing the fitness landscape object

    Notes
    -----
    In GraphFAL, a fitness landscape is represented as a genotype network in igraph, where 
    each node is a genotype, (entries in ``X``) with fitness values (values in ``f``) (and 
    any additional information as specified by ``extra_info``) are stored as node 
    attributes; Neighboring nodes are connected via directed edges, which always point to 
    fitter genotypes (as specified by ``maximize``). The absolute fitness differences are 
    used as edge weights. 

    For neutral configurations, which is controlled by the noise threshold ``epsilon``, 
    edges will be bi-directional with weights equal to zero. 

    Parameters
    ----------
    X : pd.DataFrame or np.array
        The data containing the configurations/sequences to construct the landscape.

    f : pd.Series or list or np.array
        The fitness values associated with the configurations.
        # TODO multiple fitness attributes can be allowed in the future. 

    graph : nx.DiGraph
        If provided, initialize the landscape with precomputed data as networkx directed graph.

    maximize : bool
        Indicates whether the fitness is to be maximized or minimized.

    data_types : dictionary
        A dictionary specifying the data type of each variable in X. Each variable can
        be {"boolean", "categorical", "ordinal"}. If

        - X is pd.DataFrame, then the keys of data_types should match with columns of X.
        - X is np.array, the keys of data_types can be in arbitrary format, but the order
          of the keys should be the same as in X.

    TODO epsilon : "auto" or float, default="auto"
        A tolerance threshold for compensating measurement noise in the fitness values. Only
        fitness differences greater than epsilon are considered significant, otherwise they
        are considered neutral. If "auto", epsilon is calculated as ...

    verbose : bool
        Controls the verbosity of output.

    Attributes
    ----------
    graph : nx.DiGraph
        A networkx directed graph representing the landscape. Fitness values and other
        calculated information are available as node attributes. Fitness differences between
        each pair of nodes (configurations) are stored as edge weights 'delta_fit'. The
        direction of the edge always points to fitter configurations.

    n_configs : int
        Number of total configurations in the constructed landscape.

    n_vars : int
        Number of variables in the constructed landscape.

    n_edges : int
        Number of total connections in the constructed landscape.

    n_lo : int
        Number of local optima in the constructed landscape.

    Examples
    --------
    Below is an example of how to create a `Landscape` object using a dataset of hyperparameter
    configurations and their corresponding test accuracy.

    ```python

    # Define the data types for each hyperparameter
    data_types = {
        "learning_rate": "ordinal",
        "max_bin": "ordinal",
        "max_depth": "ordinal",
        "n_estimators": "ordinal",
        "subsample": "ordinal",
    }

    >>> df = pd.read_csv("hpo_xgb.csv", index_col=0)

    >>> X = df.iloc[:, :5]  # Assuming the first five columns are the configuration parameters
    >>> f = df["acc_test"]  # Assuming 'acc_test' is the column for test accuracy

    # Create a Landscape object
    >>> landscape = Landscape(X, f, maximize=True, data_types=data_types)

    # General information regarding the landscape
    >>> landscape.describe()
    ```
    """

    def __init__(
        self,
        X: pd.DataFrame = None,
        f: pd.Series = None,
        graph: nx.DiGraph = None,
        maximize: bool = True,
        epsilon: float = "auto",
        data_types: Dict[str, str] = None,
        verbose: bool = True,
    ) -> None:

        self.maximize = maximize
        self.verbose = verbose
        self.epsilon = epsilon
        self.has_lon = False
        self.graph = None
        self.configs = None
        self.config_dict = None
        self.basin_index = None
        self.data_types = None
        self.lo_index = None
        self.n_configs = None
        self.n_edges = None
        self.n_lo = None
        self.n_vars = None

        if graph is None:
            if self.verbose:
                print("Creating landscape from scratch with X and f...")
            if X is None or f is None:
                raise ValueError("X and f cannot be None if graph is not provided.")
            if len(X) != len(f):
                raise ValueError("X and f must have the same length.")
            if data_types is None:
                raise ValueError("data_types cannot be None if graph is not provided.")

            self.n_configs = X.shape[0]
            self.n_vars = X.shape[1]
            self.data_types = data_types
            X, f, data_types = self._validate_data(X, f, data_types)
            data = self._prepare_data(X, f, data_types)
            edge_list = self._construct_neighborhoods(data, n_edit=1)
            self.graph = self._construct_landscape(data, edge_list)
            self.graph = self._add_network_metrics(self.graph, weight="delta_fit")
            self._determine_local_optima()
            self._determine_basin_of_attraction()
            self._determine_accessible_paths()
            self._determine_global_optimum()
            self._determine_dist_to_go(distance=mixed_distance)

        else:
            if self.verbose:
                print("Loading landscape from precomputed graph")
            self.graph = graph

        if self.verbose:
            print("Landscape constructed!\n")

    def __str__(self):
        """Return basic information of the landscape."""
        return (f"Landscape with {self.n_vars} variables, "
                f"{self.n_configs} configurations, "
                f"{self.n_edges} connections, and "
                f"{self.n_lo} local optima.")

    def __repr__(self):
        """Return summary of the landscape."""
        return self.__str__()
    
    def __len__(self):
        """Return the number of configurations in the landscape."""
        return self.n_configs
    
    def __getitem__(self, index):
        """Return the indexed genotype along with its fitness and attributes."""
        return self.graph.nodes[index]
    
    @property
    def shape(self):
        """Return the shape of the landscape"""
        # TODO I do not think this is a good idea for "shape" of a landscape.
        # it may cause confusions. 
        return (self.n_configs, self.n_edges)

    def _validate_data(self, X: pd.DataFrame, f: pd.Series, data_types: Dict[str, str]):
        if self.verbose:
            print("# Validating data...")

        if not isinstance(f, pd.Series):
            f = pd.Series(f)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"x{i}" for i in range(X.shape[1])]
            self.data_types = {
                f"x{i}": value for i, (key, value) in enumerate(data_types.items())
            }

        if X.duplicated().any():
            X = X.drop_duplicates()
            f = f[X.index]
            print("Warning: Duplicate configurations have been detected and removed.")

        if X.isnull().values.any():
            X = X.dropna()
            f = f[X.index]
            print("Warning: NaN values have been detected and removed.")

        if f.isnull().values.any():
            f = f.dropna()
            X = X.loc[f.index]
            print("Warning: NaN values have been detected and removed.")

        if set(data_types.keys()) != set(X.columns):
            raise ValueError("The keys of data_types must match with the columns of X.")

        data_types = {key: data_types[key] for key in X.columns}

        return X, f, data_types

    def _prepare_data(
        self, X: pd.DataFrame, f: pd.Series, data_types: Dict[str, str]
    ) -> Tuple[pd.DataFrame, dict]:
        """Preprocess the input data and generate domain dictionary for X"""

        if self.verbose:
            print("# Preparing data...")

        X = X[list(data_types.keys())]
        X.index = range(len(X))
        f.index = range(len(f))
        f.name = "fitness"

        X_raw = copy.deepcopy(X)

        for column in X.columns:
            dtype = data_types[column]
            if dtype == "boolean":
                X[column] = X[column].astype(bool)
                X_raw[column] = X_raw[column].astype(bool)
            elif dtype == "categorical":
                X[column] = pd.Categorical(X[column]).codes
            elif dtype == "ordinal":
                X[column] = pd.Categorical(X[column], ordered=True).codes
            else:
                raise ValueError(f"Unknown variable type: {dtype}")

        self.configs = pd.Series(X.apply(tuple, axis=1))
        self.config_dict = self._generate_config_dict(data_types, X)

        data = pd.concat([X_raw, f], axis=1)
        data.index = range(len(data))

        return data

    def _generate_config_dict(
        self, data_types: Dict[str, str], data: pd.DataFrame
    ) -> Dict[Any, Any]:
        """Generate a dictionary specifying the domain of X"""

        max_values = data[list(data_types.keys())].max()
        config_dict = {}
        for idx, (key, dtype) in enumerate(data_types.items()):
            config_dict[idx] = {"type": dtype, "max": max_values[key]}
        return config_dict

    def _construct_neighborhoods(self, data, n_edit: int = 1) -> List[List[Tuple]]:
        """Finding the neighbors for a list of configurations"""

        config_to_index = dict(zip(self.configs, data.index))
        config_to_fitness = dict(zip(self.configs, data["fitness"]))
        edge_list = []
        for config in tqdm(
            self.configs, total=self.n_configs, desc="# Constructing neighborhoods"
        ):
            current_fit = config_to_fitness[config]
            current_id = config_to_index[config]
            neighbors = self._generate_neighbors(config, self.config_dict, n_edit)
            for neighbor in neighbors:
                try:
                    neighbor_fit = config_to_fitness[neighbor]
                    delta_fit = current_fit - neighbor_fit
                    if (self.maximize and delta_fit < 0) or (
                        not self.maximize and delta_fit > 0
                    ):
                        edge_list.append(
                            (
                                current_id,
                                config_to_index[neighbor],
                                abs(delta_fit),
                            )
                        )
                    # TODO: neutrality is currently prohibited
                    # elif delta_fit == 0:
                    #     edge_list.append(
                    #         (current_id, config_to_index[neighbor],0,)
                    #     )
                    #     edge_list.append(
                    #         (config_to_index[neighbor], current_id,0,)
                    #     )
                except:
                    None

        return edge_list

    def _generate_neighbors(
        self,
        config: Tuple[Any, ...],
        config_dict: Dict[Any, Any],
        n_edit: int = 1,
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
                if value < config_max - 1:
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

    def _construct_landscape(
        self,
        data,
        edge_list: List[Tuple],
    ) -> nx.DiGraph:
        """Constructing the fitness landscape"""

        if self.verbose:
            print("# Constructing landscape...")

        graph = nx.DiGraph()
        graph.add_weighted_edges_from(edge_list, "delta_fit")

        if self.verbose:
            print(" - Adding node attributes...")
        for column in data.columns:
            nx.set_node_attributes(graph, data[column].to_dict(), column)

        self.n_edges = graph.number_of_edges()
        delta_n = self.n_configs - graph.number_of_nodes()
        if delta_n != 0:
            print(
                f"Warning: {delta_n} configurations are not connected to the giant component"
                + " of the landscape and have been removed."
            )
            self.configs = self.configs[graph.nodes]

        return graph

    def _add_network_metrics(
        self, graph: nx.DiGraph, weight: str = "delta_fit"
    ) -> nx.DiGraph:
        """Calculate basic network metrics for nodes"""

        if self.verbose:
            print("# Calculating network metrics...")

        graph = add_network_metrics(graph, weight=weight)

        return graph

    def _determine_local_optima(self) -> None:
        """Determine the local optima in the landscape."""

        if self.verbose:
            print("# Determining local optima...")

        out_degrees = dict(self.graph.out_degree())
        is_lo = {node: out_degrees[node] == 0 for node in self.graph.nodes}
        nx.set_node_attributes(self.graph, is_lo, "is_lo")
        self.n_lo = sum(is_lo.values())

        is_lo = pd.Series(nx.get_node_attributes(self.graph, "is_lo"))
        self.lo_index = list(is_lo[is_lo].sort_index().index)

    def _determine_basin_of_attraction(self) -> None:
        """Determine the basin of attraction of each local optimum."""

        if self.verbose:
            print("# Calculating basins of attraction...")

        basin_index = defaultdict(int)
        dict_size = defaultdict(int)
        dict_diameter = defaultdict(list)

        for i in tqdm(
            list(self.graph.nodes),
            total=self.n_configs,
            desc=" - Local searching from each config",
        ):
            lo, steps = hill_climb(
                self.graph,
                i,
                "delta_fit",
            )
            basin_index[i] = lo
            dict_size[lo] += 1
            dict_diameter[lo].append(steps)

        nx.set_node_attributes(self.graph, basin_index, "basin_index")
        nx.set_node_attributes(self.graph, dict_size, "size_basin_best")
        nx.set_node_attributes(
            self.graph,
            {k: max(v) for k, v in dict_diameter.items()},
            "max_radius_basin",
        )

        self.basin_index = basin_index

    def _determine_accessible_paths(self) -> None:
        """
        From a biological perspective, this function determines the the number of 
        configurations in the landscape that can access each local optimum
        through fitness-increasing (monotonic) mutational moves. In the evolutionary
        computation context, this equates to count the size of the basin of attraction 
        based on first-improvement local search.
        """
        if self.n_configs < 200000:
            dict_size = defaultdict(int)
            
            for lo in tqdm(
                self.lo_index,
                total=self.n_lo,
                desc="Determing accessible paths for each configuration"
            ):
                count = len(nx.ancestors(self.graph, lo))
                dict_size[lo] = count
                nx.set_node_attributes(self.graph, dict_size, "size_basin_first")
        else:
            if self.verbose:
                print("Accessible path analysis has been escaped given the size of the landscape.")

    def accessible_pathway_ratio(self) -> float:
        
        go_basin_size = len(nx.ancestors(self.graph, self.go_index))
        return go_basin_size / self.n_configs

    def _determine_global_optimum(self) -> None:
        """Determine global optimum of the landscape."""

        if self.verbose:
            print("# Determining global peak...")

        fitness_list = pd.Series(nx.get_node_attributes(self.graph, "fitness"))

        if self.maximize:
            self.go_index = fitness_list.idxmax()
        else:
            self.go_index = fitness_list.idxmin()

        self.go = self.graph.nodes[self.go_index]

    def _determine_dist_to_go(self, distance) -> None:
        """Calculate the distance to the global optimum for each configuration."""

        if self.verbose:
            print("# Calculating distances to global optimum...")

        data = self.get_data()
        configs = np.array(self.configs.to_list())
        go_config = configs[self.go_index]
        distances = distance(configs, go_config, self.data_types)
        nx.set_node_attributes(self.graph, dict(zip(data.index, distances)), "dist_go")

    def get_data(self, lo_only: bool = False) -> pd.DataFrame:
        """
        Get tabular landscape data as pd.DataFrame.

        Parameters
        ----------
        lo_only : bool, default=False
            Whether to return only local optima configurations.

        Returns
        -------
        pd.DataFrame : A pandas dataframe containing all information regarding each configuration.
        """

        if lo_only:
            if not self.has_lon:
                graph_lo_ = self.graph.subgraph(self.lo_index)
                data_lo = pd.DataFrame.from_dict(
                    dict(graph_lo_.nodes(data=True)), orient="index"
                ).sort_index()
                return data_lo.drop(
                    columns=["is_lo", "out_degree", "in_degree", "basin_index"]
                )
            else:
                data_lo = pd.DataFrame.from_dict(
                    dict(self.lon.nodes(data=True)), orient="index"
                ).sort_index()
                return data_lo
        else:
            data = pd.DataFrame.from_dict(
                dict(self.graph.nodes(data=True)), orient="index"
            ).sort_index()
            return data.drop(columns=["size_basin_first", "size_basin_best", "max_radius_basin"])

    def describe(self) -> None:
        """Print the basic information of the landscape."""

        print("---")
        print(f"number of variables: {self.n_vars}")
        print(f"number of configurations: {self.n_configs}")
        print(f"number of connections: {self.n_edges}")
        print(f"number of local optima: {self.n_lo}")

    def fdc(
        self,
        method: str = "spearman",
    ) -> float:
        """
        Calculate the fitness distance correlation of a landscape. It assesses how likely is it
        to encounter higher fitness values when moving closer to the global optimum.

        It will add an attribute `fdc` to the landscape object, and also create a "dist_go"
        column to both `data` and `data_lo`.

        The distance measure here is based on a combination of Hamming and Manhattan distances,
        to allow for mixed-type variables. See `Landscape._mixed_distance`.

        Parameters
        ----------
        method : str, one of {"spearman", "pearson"}, default="spearman"
            The correlation measure used to assess FDC.

        Returne
        -------
        float : An FDC value ranging from -1 to 1. A value close to 1 indicates positive correlation
            between the fitness values of a configuration and its distance to the global optimum.
        """

        return fdc(self, method=method)

    def ffi(self, frac: float = 1, min_len: int = 3, method: str = "spearman") -> float:
        """
        Calculate the fitness flatenning index (FFI) of the landscape. It assesses whether the
        landscape tends to be flatter around the global optimum. It operates by identifying
        (part of, controled by `frac`) adaptive paths leading to the global optimum, and
        checks whether the fitness gain in each step decreases as approaching the global peak.

        Parameters
        ----------
        frac : float, default=1
            The fraction of adapative paths to be assessed.

        min_len : int, default=3
            Minimum length of an adaptive path for it to be considered in evaluation.

        method : str, one of {"spearman", "pearson"}, default="spearman"
            The correlation measure used to assess FDC.

        Returns
        -------
        float : An FFI value ranging from -1 to 1. A value close to 1 indicates that the landscape
            is very likely to be flatter around the global optimum.
        """

        return ffi(self, frac=frac, min_len=min_len, method=method)

    def fitness_assortativity(self) -> float:
        """
        Calculate the assortativity of the landscape based on fitness values.

        Returns
        -------
        float : The assortativity value of the landscape.
        """

        if self.graph.number_of_nodes() > 100000:
            warnings.warn("The number of nodes in the graph is greater than 100,000.")

        assortativity = nx.numeric_assortativity_coefficient(self.graph, "fitness")
        return assortativity

    def autocorrelation(
        self, walk_length: int = 20, walk_times: int = 1000, lag: int = 1
    ) -> Tuple[float, float]:
        """
        A measure of landscape ruggedness. It operates by calculating the autocorrelation of
        fitness values over multiple random walks on a graph.

        Parameters:
        ----------
        walk_length : int, default=20
            The length of each random walk.

        walk_times : int, default=1000
            The number of random walks to perform.

        lag : int, default=1
            The distance lag used for calculating autocorrelation. See pandas.Series.autocorr.

        Returns:
        -------
        autocorr : Tuple[float, float]
            A tuple containing the mean and variance of the autocorrelation values.
        """

        return autocorrelation(
            self, walk_length=walk_length, walk_times=walk_times, lag=lag
        )

    def neutrality(self, threshold: float = 0.01) -> float:
        """
        Calculate the neutrality index of the landscape. It assesses the proportion of neighbors
        with fitness values within a given threshold, indicating the presence of neutral areas in
        the landscape.

        Parameters
        ----------
        threshold : float, default=0.01
            The fitness difference threshold for neighbors to be considered neutral.

        Returns
        -------
        neutrality : float
            The neutrality index, which ranges from 0 to 1, where higher values indicate more
            neutrality in the landscape.
        """

        return neutrality(self, threshold=threshold)

    def ruggedness(self) -> float:
        """
        Calculate the ruggedness index of the landscape. It is defined as the ratio of the number
        of local optima to the total number of configurations.

        Parameters
        ----------
        landscape : Landscape
            The fitness landscape object.

        Returns
        -------
        float
            The ruggedness index, ranging from 0 to 1.
        """

        return ruggedness(self)
    
    def r_s_ratio(self) -> float:
        """
        Calculate the roughness-to-slope (r/s) ratio of a fitness landscape. This metric quantifies
        the deviation from additivity, with higher values indicating greater ruggedness and epistasis.

        Parameters
        ----------
        landscape : object
            A landscape object with a `get_data()` method that provides sequence and fitness data.

        Returns
        -------
        float
            The roughness-to-slope (r/s) ratio.
        """
        return r_s_ratio(self)
    
    def idiosyncratic_index(self, mutation:tuple) -> float:
        """
        Calculates the idiosyncratic index for the fitness landscape.

        The idiosyncratic index of a specific genetic mutation quantifies the sensitivity 
        of a specific mutation to idiosyncratic epistasis. It is defined as the as the 
        variation in the fitness difference between genotypes that differ by the mutation, 
        relative to the variation in the fitness difference between random genotypes for 
        the same number of genotype pairs. We compute this for the entire fitness landscape 
        by averaging it across individual mutations. 

        The idiosyncratic index for a landscape varies from 0 to 1, corresponding to the 
        minimum and maximum levels of idiosyncrasy, respectively.

        For more information, please refer to the original paper:
        
        [1] Daniel M. Lyons et al, "Idiosyncratic epistasis creates universals in mutational 
        effects and evolutionary trajectories", Nat. Ecol. Evo., 2020.

        Parameters
        ----------
        landscape : Landscape
            The fitness landscape object.

        mutation : tuple(A, pos, B)
            A tuple containing:
            - A: The original variable value (allele) at the given position.
            - pos: The position in the configuration where the mutation occurs.
            - B: The new variable value (allele) after the mutation.

        Returns
        -------
        float
            The calculated idiosyncratic index.
        """
        return idiosyncratic_index(self, mutation=mutation)

    def basin_size_fit_corr(self, method: str = "spearman") -> tuple:
        """
        Calculate the correlation between the size of the basin of attraction and the fitness of local optima.

        Parameters
        ----------
        landscape : Landscape
            The fitness landscape object.

        method : str, one of {"spearman", "pearson"}, default="spearman"
            The correlation measure to use.

        Returns
        -------
        tuple
            A tuple containing the correlation coefficient and the p-value.
        """

        return basin_size_fit_corr(self, method=method)

    def gradient_intensity(self) -> float:
        """
        Calculate the gradient intensity of the landscape. It is defined as the average absolute
        fitness difference (delta_fit) across all edges.

        Parameters
        ----------
        landscape : Landscape
            The fitness landscape object.

        Returns
        -------
        float
            The gradient intensity.
        """

        return gradient_intensity(self)

    def single_mutation_effects(
        self, position: str, test_type: str = "positive", n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        Assess the fitness effects of all possible mutations at a single position across all genetic backgrounds.

        Parameters
        ----------
        position : str
            The name of the position (variable) to assess mutations for.

        test_type : str, default='positive'
            The type of significance test to perform. Must be 'positive' or 'negative'.

        n_jobs : int, default=1
            The number of parallel jobs to run.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing mutation pairs, median absolute fitness effect,
            p-values, and significance flags.
        """

        return single_mutation_effects(
            landscape=self, position=position, test_type=test_type, n_jobs=n_jobs
        )

    def all_mutation_effects(
        self, test_type: str = "positive", n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        Assess the fitness effects of all possible mutations across all positions in the landscape.

        Parameters
        ----------
        test_type : str, default='positive'
            The type of significance test to perform. Must be 'positive' or 'negative'.

        n_jobs : int, default=1
            The number of parallel jobs to run.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing, for each position and mutation pair, the median absolute fitness effect,
            p-values, and significance flags.
        """

        return all_mutation_effects(landscape=self, test_type=test_type, n_jobs=n_jobs)

    def pairwise_epistasis(self, pos1: str, pos2: str) -> pd.DataFrame:
        """
        Assess the pairwise epistasis effects between all unique unordered mutations
        at two specified positions within the landscape.

        This method leverages the `pairwise_epistasis` function to automatically enumerate all
        possible mutations at the given positions, compute epistatic interactions, and return
        the results in a structured DataFrame.

        Parameters
        ----------
        pos1 : str
            The name of the first genetic position to assess mutations for.

        pos2 : str
            The name of the second genetic position to assess mutations for.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the epistasis results for all mutation pairs between
            the two positions.

        Raises
        ------
        ValueError
            If either `pos1` or `pos2` is not a valid column in the landscape's genotype matrix.

        Examples
        --------
        ```python
        # Assuming you have a Landscape object named 'landscape'

        # Define the two positions to assess for epistasis
        position1 = 'position_3'
        position2 = 'position_5'

        # Compute pairwise epistasis between position1 and position2
        epistasis_results = landscape.pairwise_epistasis(pos1=position1, pos2=position2)

        # View the results
        print(epistasis_results)
        ```
        """

        data = self.get_data()
        X = data.iloc[:, : len(self.data_types)]
        f = data["fitness"]

        if pos1 not in X.columns:
            raise ValueError(
                f"Position '{pos1}' is not a valid column in the genotype matrix."
            )
        if pos2 not in X.columns:
            raise ValueError(
                f"Position '{pos2}' is not a valid column in the genotype matrix."
            )

        epistasis_df = pairwise_epistasis(X, f, pos1, pos2)

        return epistasis_df

    def all_pairwise_epistasis(self, n_jobs: int = 1) -> pd.DataFrame:
        """
        Compute epistasis effects between all unique pairs of positions within the landscape using parallel execution.

        This method leverages the `all_pairwise_epistasis` function to iterate over all possible
        pairs of genetic positions, compute their epistatic interactions in parallel, and compile the
        results into a comprehensive DataFrame.

        Parameters
        ----------
        n_jobs : int, default=1
            The number of parallel jobs to run. -1 means using all available cores.

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame containing epistasis results for all position pairs.
            Each row corresponds to a specific mutation pair between two positions.

        Raises
        ------
        ValueError
            If the genotype matrix or fitness data is not properly initialized.

        Examples
        --------
        ```python
        # Assuming you have a Landscape object named 'landscape'

        # Compute epistasis between all pairs of positions using 4 cores
        all_epistasis_results = landscape.all_pairwise_epistasis(n_jobs=4)

        # View the results
        print(all_epistasis_results)
        ```
        """

        data = self.get_data()
        X = data.iloc[:, : len(self.data_types)]
        f = data["fitness"]

        if X.empty:
            raise ValueError("Genotype matrix X is empty.")
        if f.empty:
            raise ValueError("Fitness data f is empty.")
        if len(X) != len(f):
            raise ValueError("Mismatch between number of genotypes and fitness values.")

        all_epistasis_df = all_pairwise_epistasis(X, f, n_jobs=n_jobs)

        return all_epistasis_df

    def draw_neighborhood(
        self,
        node: Any,
        radius: int = 1,
        node_size: int = 300,
        with_labels: bool = True,
        font_weight: str = "bold",
        font_size: int = 12,
        font_color: str = "black",
        node_label: str = "fitness",
        node_color: Any = "fitness",
        edge_label: str = "delta_fit",
        colormap=plt.cm.RdBu_r,
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

        edge_label : str, optional, default="delta_fit"
            The edge attribute to use for labeling edges. If None, then no edge labels
            are displayed.

        colormap : matplotlib colormap, optional, default=plt.cm.Blues
            The Matplotlib colormap to use for node coloring.

        alpha : float, optional, default=1.0
            The alpha value for node colors.
        """

        draw_neighborhood(
            G=self.graph,
            node=node,
            radius=radius,
            node_size=node_size,
            with_labels=with_labels,
            font_weight=font_weight,
            font_size=font_size,
            font_color=font_color,
            node_label=node_label,
            node_color=node_color,
            edge_label=edge_label,
            colormap=colormap,
            alpha=alpha,
        )

    def draw_landscape_2d(
        self,
        fitness: str = "fitness",
        embedding_model: Any = HOPE(),
        reducer: Any = umap.UMAP(n_neighbors=15, n_epochs=500, min_dist=1),
        rank: bool = True,
        n_grids: int = 100,
        cmap: Any = palettable.lightbartlein.diverging.BlueOrangeRed_3,
    ) -> None:
        """
        Draws a 2D visualization of a landscape by plotting reduced graph embeddings and coloring them
        according to the fitness values.

        Parameters
        ----------
        landscape : Any
            The landscape object that contains the graph and data for visualization.

        fitness : str, default="fitness"
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

        cmap : Any, default=palettable.lightbartlein.diverging.BlueOrangeRed_3
            The color map from 'palettable' used for coloring the contour plot.
        """

        draw_landscape_2d(
            self,
            metric=fitness,
            embedding_model=embedding_model,
            reducer=reducer,
            rank=rank,
            n_grids=n_grids,
            cmap=cmap,
        )

    def draw_landscape_3d(
        self,
        fitness: str = "fitness",
        embedding_model: Any = HOPE(),
        reducer: Any = umap.UMAP(n_neighbors=15, n_epochs=500, min_dist=1),
        rank: bool = True,
        n_grids: int = 100,
        cmap: Any = palettable.lightbartlein.diverging.BlueOrangeRed_3,
    ) -> None:
        """
        Draws a 3D interactive visualization of a landscape by plotting reduced graph embeddings and coloring
        them according to a specified metric.

        Parameters
        ----------
        landscape : Any
            The landscape object that contains the graph and data for visualization.

        fitness : str, default="fitness"
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

        cmap : Any, default=palettable.lightbartlein.diverging.BlueOrangeRed_3
            The color map from 'palettable' used for coloring the contour plot.
        """

        draw_landscape_3d(
            self,
            metric=fitness,
            embedding_model=embedding_model,
            reducer=reducer,
            rank=rank,
            n_grids=n_grids,
            cmap=cmap,
        )

    def draw_epistasis(
        self,
        epistasis_df=None,
        p_threshold=0.05,
        cohen_d_threshold=0.5,
        figsize=(5, 5),
        node_color="#f2f2f2",
        label_font_size=10,
        node_size=500,
        legend_loc="upper right",
        edge_width_scale=2,
    ) -> None:
        """
        Calls the external draw_epistasis function to visualize epistatic interactions.

        Parameters
        ----------
        aggregated_epistasis_df : pd.DataFrame
            Aggregated epistasis results for all position pairs.

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
            Displays the epistasis plot.
        """
        if epistasis_df is None:
            epistasis_df = self.all_pairwise_epistasis()

        draw_epistasis(
            epistasis_df=epistasis_df,
            p_threshold=p_threshold,
            cohen_d_threshold=cohen_d_threshold,
            figsize=figsize,
            node_color=node_color,
            label_font_size=label_font_size,
            node_size=node_size,
            legend_loc=legend_loc,
            edge_width_scale=edge_width_scale,
        )

    def draw_fdc(
        self,
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

        draw_fdc(
            data=self.get_data(),
            distance="dist_go",
            fitness="fitness",
            confidence_level=confidence_level,
        )

    def draw_fitness_dist(
        self,
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
            Displays a histogram of the fitness values in the dataset.
        """

        draw_fitness_dist(
            fitness=self.get_data()["fitness"],
            type=type,
            bins=bins,
            color=color,
            edgecolor=edgecolor,
            figsize=figsize,
            log=log
        )

    def get_lon(
        self,
        mlon: bool = True,
        min_edge_freq: int = 3,
        trim: int = None,
        verbose: bool = True,
    ) -> nx.DiGraph:
        """
        Construct the local optima network (LON) of the fitness landscape.

        Parameters
        ----------
        mlon : bool, default=True
            Whether to use monotonic-LON (M-LON), which will only have improving edges.

        min_edge_freq : int, default=3
            Minimal escape frequency needed to construct an edge between two local optima.

        trim : int, default=None
            The number of edges with the highest transition probability to retain for each node.

        Returns
        -------
        nx.DiGraph : The constructed local optimum network (LON).
        """

        if verbose:
            print("Constructing local optima network...")

        self.lon = get_lon(
            graph=self.graph,
            configs=self.configs,
            lo_index=self.lo_index,
            config_dict=self.config_dict,
            maximize=self.maximize,
            mlon=mlon,
            min_edge_freq=min_edge_freq,
            trim=trim,
            verbose=verbose,
        )
        self.has_lon = True
        return self.lon
    
    def draw_diminishing_return(
        self,
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

        draw_diminishing_return(
            self,
            sample=sample,
            color=color,
            figsize=figsize
        )

    def draw_ffi(
        self,
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

        draw_ffi(
            self,
            figsize=figsize,
            min_len=min_len
        )

    def classify_epistasis(self, type="pos_neg"):
        """
        Classify the type of epistasis present in a given fitness landscape.

        This function analyzes a fitness landscape to determine whether it exhibits positive/negative 
        epistasis or magnitude/sign/reciprocal sign epistasis. The classification is determined by 
        identifying 4-node cycles (squares) and analyzing the relationships between mutations.

        # TODO: support synergistic/antagonistic epistasis

        Parameters
        ----------
        landscape : Landscape
            The fitness landscape object.

        type : str, optional (default="pos_neg")
            The type of epistasis to classify. Supported options are:
            - "pos_neg": Classifies positive and negative epistasis.
            - "mag_sign": Classifies magnitude epistasis, sign epistasis, and reciprocal sign epistasis.

        Returns
        -------
        dict
            A dictionary with the proportion of squares exhibiting the specified type(s) of epistasis:
            - If `type="pos_neg"`, the dictionary contains:
                - "positive epistasis": Proportion of squares with positive epistasis.
                - "negative epistasis": Proportion of squares with negative epistasis.
            - If `type="mag_sign"`, the dictionary contains:
                - "magnitude epistasis": Proportion of squares with magnitude epistasis.
                - "sign epistasis": Proportion of squares with sign epistasis.
                - "reciprocal sign epistasis": Proportion of squares with reciprocal sign epistasis.

        Raises
        ------
        ValueError
            If the `type` argument is not "pos_neg" or "mag_sign".
        """
        return classify_epistasis(self, type=type)

    

