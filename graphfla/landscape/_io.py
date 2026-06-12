"""GraphML serialization for :class:`~graphfla.landscape.landscape.Landscape`.

Read/write of the landscape graph and its essential attributes. Mixed into the
core class as ``_IOMixin`` so ``landscape.to_graph(...)`` /
``Landscape.build_from_graph(...)`` keep their public signatures; the bodies
live here to keep landscape.py focused on the class surface.
"""

from __future__ import annotations

import ast
import warnings
from typing import TYPE_CHECKING

import igraph as ig
import pandas as pd

from .._data import encode_data
from ..utils import infer_graph_properties
from .._logging import enable_verbose_logging
from . import _plateaus
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .landscape import Landscape


class _IOMixin:
    """GraphML read/write methods for :class:`Landscape`."""

    @classmethod
    def build_from_graph(
        cls,
        filepath: str,
        *,
        verbose: bool = True,
    ) -> "Landscape":
        """Construct a landscape from a saved graph file.

        This class method creates a new landscape instance by loading a previously
        saved graph, avoiding the need to reconstruct the landscape from original
        configuration data. This is significantly faster than building from scratch.

        Parameters
        ----------
        filepath : str
            Path to the saved graph file (.graphml).
        verbose : bool, default=True
            Controls verbosity of output during loading and analysis.

        Returns
        -------
        Landscape
            A new instance populated with the graph and inferred properties.

        Raises
        ------
        ValueError
            If the file cannot be read or doesn't contain valid graph data.
        FileNotFoundError
            If the specified file doesn't exist.

        Notes
        -----
        This method will:
        1. Load the saved graph structure and attributes
        2. Infer essential landscape properties from the graph
        3. Recalculate local optima and global optimum from the graph structure

        Previously computed attributes (basins, accessible paths, distances,
        neighbor fitness) are preserved from the saved graph if present.

        Some specialized attributes from subclasses (like sequence_length in
        SequenceLandscape) will be inferred where possible.

        Only load GraphML from trusted sources: embedded configuration metadata
        is parsed with :func:`ast.literal_eval` (safe against arbitrary code
        execution, but not a substitute for validating untrusted files).
        """
        instance = cls()
        instance.verbose = verbose
        if verbose:
            enable_verbose_logging()

        if verbose:
            logger.info(f"Loading landscape from {filepath}...")

        try:
            graph = ig.Graph.Read_GraphML(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to load graph from {filepath}: {e}")

        instance.graph = graph

        if "maximize" in graph.attributes():
            instance.maximize = bool(graph["maximize"])
        else:
            instance.maximize = True
            warnings.warn(
                "'maximize' attribute not found in graph. Defaulting to True.",
                RuntimeWarning,
            )

        if "epsilon" in graph.attributes():
            try:
                epsilon_str = graph["epsilon"]
                if epsilon_str == "auto":
                    instance.epsilon = 0.0
                else:
                    instance.epsilon = float(epsilon_str)
            except Exception:
                instance.epsilon = 0.0
                warnings.warn(
                    "Could not parse 'epsilon' attribute. Defaulting to 0.",
                    RuntimeWarning,
                )
        else:
            instance.epsilon = 0.0

        if "landscape_type" in graph.attributes():
            instance._strategy_key = graph["landscape_type"]
        if "landscape_kind" in graph.attributes():
            instance.kind = graph["landscape_kind"]

        # --- data_types (parsed first; required to reconstruct configs) ---
        if "data_types_data" in graph.attributes():
            try:
                instance.data_types = ast.literal_eval(graph["data_types_data"])
            except Exception:
                instance.data_types = None
                warnings.warn(
                    "Could not parse data_types from graph attributes.", RuntimeWarning
                )
        else:
            instance.data_types = None

        # --- configs reconstruction ---
        # Preferred path: re-encode from the feature-column vertex attributes
        # _build_graph always writes, avoiding the huge configs_data string the
        # old format produced for large landscapes.
        instance.configs = None
        instance._configs_array = None
        instance.config_dict = None

        if instance.data_types is not None:
            try:
                feature_cols = [
                    c for c in instance.data_types.keys()
                    if c in graph.vs.attributes()
                ]
                if feature_cols and "fitness" in graph.vs.attributes():
                    X_rec = pd.DataFrame(
                        {c: graph.vs[c] for c in feature_cols}
                    )
                    f_rec = pd.Series(graph.vs["fitness"], name="fitness")
                    prepared = encode_data(
                        X_rec, f_rec, instance.data_types, verbose=False
                    )
                    instance.configs = prepared.configs
                    instance._configs_array = prepared.configs_array
                    instance.config_dict = prepared.config_dict
            except Exception as e:
                warnings.warn(
                    f"Could not reconstruct configs from vertex attributes: {e}",
                    RuntimeWarning,
                )
                instance.configs = None
                instance._configs_array = None
                instance.config_dict = None

        # Legacy fallback: parse the old single-string configs_data attribute
        # (only present in GraphML files saved before this refactor).
        if instance.configs is None and "configs_data" in graph.attributes():
            try:
                raw = ast.literal_eval(graph["configs_data"])
                parsed: dict = {}
                for idx_str, config_str in raw.items():
                    try:
                        parsed[int(idx_str)] = ast.literal_eval(config_str)
                    except Exception:
                        warnings.warn(
                            f"Could not parse config entry: {idx_str}", RuntimeWarning
                        )
                instance.configs = pd.Series(parsed)
            except Exception as e:
                warnings.warn(
                    f"Could not parse legacy configs_data: {e}", RuntimeWarning
                )
                instance.configs = None

        if instance.configs is None:
            warnings.warn(
                "No configs data found in graph. Some analyses may be limited.",
                RuntimeWarning,
            )

        # config_dict fallback from legacy attribute if reconstruction did not provide it
        if instance.config_dict is None and "config_dict_data" in graph.attributes():
            try:
                instance.config_dict = ast.literal_eval(graph["config_dict_data"])
            except Exception:
                warnings.warn(
                    "Could not parse config_dict from graph attributes.", RuntimeWarning
                )

        # Infer basic properties (n_configs, n_edges, n_vars)
        instance._n_configs, instance._n_edges, instance.n_vars = infer_graph_properties(
            instance.graph,
            data_types=instance.data_types,
            configs=instance.configs,
            verbose=instance.verbose,
        )

        # Restore subclass convenience attributes by landscape kind.
        restore_kind = instance.kind
        if (
            not isinstance(restore_kind, str) or restore_kind in ("", "default")
        ) and "landscape_class" in graph.attributes():
            # Legacy graphs predating landscape_kind: infer from class name.
            legacy_class = graph["landscape_class"]
            if legacy_class in (
                "SequenceLandscape",
                "DNALandscape",
                "RNALandscape",
                "ProteinLandscape",
            ):
                restore_kind = "sequence"
            elif legacy_class == "BooleanLandscape":
                restore_kind = "boolean"

        if instance.n_vars is not None and isinstance(restore_kind, str):
            if restore_kind in ("sequence", "dna", "rna", "protein"):
                instance.sequence_length = instance.n_vars
            elif restore_kind == "boolean":
                instance.bit_length = instance.n_vars

        # Reconstruct plateau data structures if saved in the graph
        if "plateau_id" in graph.vs.attributes():
            _plateaus.restore_plateaus(instance)

        # Determine local optima and global optimum from graph structure
        if instance.graph.vcount() > 0:
            instance._compute_local_optima()
            instance._compute_global_optimum()

        # Infer calculation status flags from saved graph attributes
        if "basin_index" in graph.vs.attributes():
            instance._basin_calculated = True
        if "size_basin_accessible" in graph.vs.attributes():
            instance._path_calculated = True
        if "dist_go" in graph.vs.attributes():
            instance._distance_calculated = True
        if "mean_neighbor_fit" in graph.vs.attributes():
            instance._neighbor_fit_calculated = True

        instance._is_built = True

        if verbose:
            logger.info(
                f"Landscape successfully loaded. Graph has {instance.n_configs} nodes and {instance.n_edges} edges."
            )
            instance.describe()

        return instance

    def to_graph(self, filepath: str) -> None:
        """Save the landscape graph and essential attributes to a file.

        This method serializes the landscape's graph structure and relevant
        attributes to a GraphML file, which can later be loaded using `build_from_graph`.
        This allows efficient storage and sharing of landscapes without requiring
        re-construction from scratch.

        Parameters
        ----------
        filepath : str
            The path where the graph file will be saved. If the file doesn't end
            with '.graphml', this extension will be added automatically.

        Raises
        ------
        NotBuiltError
            If the landscape has not been built.
        ValueError
            If the graph cannot be saved to the specified path.

        Notes
        -----
        The GraphML format preserves the graph structure and all vertex/edge attributes.
        In addition to the graph itself, essential landscape attributes like `maximize`
        and `epsilon` are stored as graph attributes.
        """
        self._check_built()

        if self.graph is None:
            raise ValueError("Cannot save an empty graph.")

        if not filepath.endswith(".graphml"):
            filepath = f"{filepath}.graphml"

        graph_copy = self.graph.copy()

        # Essential landscape attributes saved as graph attributes
        graph_copy["maximize"] = self.maximize
        graph_copy["epsilon"] = str(self.epsilon)
        graph_copy["landscape_class"] = self.__class__.__name__
        graph_copy["landscape_type"] = self._strategy_key
        graph_copy["landscape_kind"] = self.kind

        # Configs are preserved implicitly via the feature-column vertex
        # attributes _build_graph writes, so no separate configs_data attribute
        # is needed.

        if self.config_dict is not None:
            graph_copy["config_dict_data"] = str(self.config_dict)

        if self.data_types is not None:
            graph_copy["data_types_data"] = str(self.data_types)

        try:
            graph_copy.write_graphml(filepath)
            if self.verbose:
                logger.info(f"Landscape graph saved to {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to save graph to {filepath}: {e}")
