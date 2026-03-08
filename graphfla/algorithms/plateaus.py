from collections import defaultdict

import igraph as ig
import numpy as np

from ..utils import timeit


@timeit
def build_plateaus(landscape, neutral_pairs):
    """Build neutral network and identify connected components (plateaus).

    Each connected component of nodes linked by neutral (within-epsilon
    fitness) edges forms a plateau. Only multi-member plateaus are tracked;
    singleton nodes remain unaffected.

    Parameters
    ----------
    neutral_pairs : list[tuple[int, int]]
        Undirected neutral neighbor pairs from ``_build_edges``.
    """
    self = landscape

    if not neutral_pairs:
        self._has_plateaus = False
        self.n_plateau = 0
        return

    if self.graph is None or self.n_configs is None:
        self._has_plateaus = False
        self.n_plateau = 0
        return

    # Build a lightweight undirected graph just for component detection
    neutral_graph = ig.Graph(n=self.n_configs, edges=neutral_pairs, directed=False)
    components = neutral_graph.connected_components()

    # For singletons, plateau_id == node index (identity mapping).
    # Multi-member plateaus get IDs starting at n_configs to avoid collisions.
    node_to_plateau = np.arange(self.n_configs, dtype=np.int32)
    plateaus = {}
    next_plateau_id = self.n_configs

    for members in components:
        if len(members) > 1:
            member_list = sorted(members)
            plateaus[next_plateau_id] = member_list
            for node in member_list:
                node_to_plateau[node] = next_plateau_id
            next_plateau_id += 1

    if not plateaus:
        self._has_plateaus = False
        self.n_plateau = 0
        return

    self._node_to_plateau = node_to_plateau
    self._plateaus = plateaus
    self.n_plateau = len(plateaus)
    self._has_plateaus = True

    # Build per-node neutral neighbor adjacency
    neutral_neighbors = defaultdict(list)
    for u, v in neutral_pairs:
        neutral_neighbors[u].append(v)
        neutral_neighbors[v].append(u)
    self._neutral_neighbors = dict(neutral_neighbors)

    # Annotate graph vertices
    self.graph.vs["plateau_id"] = node_to_plateau.tolist()
    plateau_sizes = np.ones(self.n_configs, dtype=np.int32)
    for pid, members in plateaus.items():
        size = len(members)
        for node in members:
            plateau_sizes[node] = size
    self.graph.vs["plateau_size"] = plateau_sizes.tolist()

    if self.verbose:
        total_neutral_nodes = sum(len(m) for m in plateaus.values())
        print(
            f" - Identified {len(plateaus)} neutral plateaus "
            f"covering {total_neutral_nodes} nodes."
        )


def restore_plateaus(landscape):
    """Rebuild plateau data structures from saved vertex attributes.

    Called by ``build_from_graph`` when a graph with ``plateau_id``
    attributes is loaded.  When ``self.configs`` is available, neutral
    neighbor adjacency (``_neutral_neighbors``) is also reconstructed so
    that ``determine_neighbor_fitness`` remains plateau-aware.
    """
    self = landscape

    if self.graph is None or "plateau_id" not in self.graph.vs.attributes():
        self._has_plateaus = False
        self.n_plateau = 0
        return

    n = self.graph.vcount()
    node_to_plateau = np.array(self.graph.vs["plateau_id"], dtype=np.int32)

    plateaus = defaultdict(list)
    for node_idx in range(n):
        pid = int(node_to_plateau[node_idx])
        if pid != node_idx:
            plateaus[pid].append(node_idx)

    if not plateaus:
        self._has_plateaus = False
        self.n_plateau = 0
        return

    self._node_to_plateau = node_to_plateau
    self._plateaus = dict(plateaus)
    self.n_plateau = len(plateaus)
    self._has_plateaus = True

    # Rebuild _neutral_neighbors from configs (single-edit adjacency
    # within each plateau).  Falls back to None if configs are unavailable.
    if self.configs is not None and len(self.configs) == n:
        neutral_neighbors = defaultdict(list)
        for pid, members in self._plateaus.items():
            member_configs = [np.array(self.configs.iloc[m]) for m in members]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if np.sum(member_configs[i] != member_configs[j]) == 1:
                        neutral_neighbors[members[i]].append(members[j])
                        neutral_neighbors[members[j]].append(members[i])
        self._neutral_neighbors = (
            dict(neutral_neighbors) if neutral_neighbors else None
        )
    else:
        self._neutral_neighbors = None
