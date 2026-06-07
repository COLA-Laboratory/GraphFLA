from collections import defaultdict
from itertools import chain

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..utils import timeit


@timeit
def build_plateaus(landscape, neutral_pairs):
    """Build neutral network and identify connected components (plateaus).

    Each connected component of nodes linked by neutral (within-epsilon
    fitness) edges forms a plateau.  Only multi-member components are
    stored; singleton nodes receive ``plateau_id = -1``.

    Plateau IDs are assigned sequentially starting from 0.

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

    n_configs = self.n_configs

    # Connected components of the neutral network via SciPy, built directly from
    # ``neutral_pairs`` as numpy arrays -- avoids constructing a second igraph.
    #
    # ``neutral_pairs`` is a Python list of ``(i, j)`` tuples; flattening it with
    # ``itertools.chain`` into a single ``np.fromiter`` is ~3x faster than
    # ``np.asarray`` on the list-of-tuples (which boxes each tuple).  The edges
    # are loaded into a COO -> CSR matrix in their stored (directed) orientation
    # only -- ``connected_components(..., connection="weak")`` already treats the
    # graph as undirected, so materialising the transpose would only double the
    # nnz and the build cost for no change in the resulting partition.
    n_pairs = len(neutral_pairs)
    flat = np.fromiter(
        chain.from_iterable(neutral_pairs), dtype=np.int64, count=2 * n_pairs
    )
    rows = flat[0::2]
    cols = flat[1::2]
    data = np.ones(n_pairs, dtype=np.bool_)
    adjacency = coo_matrix(
        (data, (rows, cols)), shape=(n_configs, n_configs)
    ).tocsr()
    _, membership = connected_components(
        adjacency, directed=False, connection="weak"
    )
    membership = membership.astype(np.int64, copy=False)

    # ``membership[v]`` is the connected-component label of vertex ``v``.  Only
    # components with more than one member are plateaus.  Plateau ids must match
    # the previous ``for members in components`` loop, which assigned ids in the
    # order components were first encountered scanning vertices 0..n-1 (skipping
    # singletons).  SciPy's component labelling is not contractually tied to
    # that convention, so plateau ids are assigned explicitly by ascending
    # first-member (smallest node) index, which reproduces the old ordering
    # regardless of how SciPy numbers its components.
    comp_sizes = np.bincount(membership, minlength=n_configs)
    multi_ids = np.nonzero(comp_sizes > 1)[0]

    if multi_ids.size == 0:
        self._has_plateaus = False
        self.n_plateau = 0
        return

    # ``first_idx[label]`` is the smallest vertex index carrying that label.
    _, first_idx = np.unique(membership, return_index=True)
    plateau_order = np.argsort(first_idx[multi_ids], kind="stable")
    ordered_comp = multi_ids[plateau_order]

    n_plateau = ordered_comp.size
    comp_to_plateau = np.full(comp_sizes.shape[0], -1, dtype=np.int32)
    comp_to_plateau[ordered_comp] = np.arange(n_plateau, dtype=np.int32)
    node_to_plateau = comp_to_plateau[membership]

    # Group member nodes by plateau id.  ``in_plateau_nodes`` is ascending, and
    # a stable sort by plateau id preserves that ascending node order within
    # each group, so each member list is already sorted (matching the previous
    # ``sorted(members)``).
    in_plateau_nodes = np.nonzero(node_to_plateau >= 0)[0]
    member_pids = node_to_plateau[in_plateau_nodes]
    order = np.argsort(member_pids, kind="stable")
    sorted_nodes = in_plateau_nodes[order]
    sorted_pids = member_pids[order]
    split_points = np.nonzero(np.diff(sorted_pids))[0] + 1
    groups = np.split(sorted_nodes, split_points)
    plateaus = {pid: grp.tolist() for pid, grp in enumerate(groups)}

    self._node_to_plateau = node_to_plateau
    self.plateaus = plateaus
    self.n_plateau = n_plateau
    self._has_plateaus = True

    # Build per-node neutral neighbor adjacency.  The explicit Python loop over
    # ``neutral_pairs`` is faster here than vectorised regrouping because the
    # neutral degree per node is high and the list->ndarray conversion of the
    # pair list is itself costly.
    neutral_neighbors = defaultdict(list)
    for u, v in neutral_pairs:
        neutral_neighbors[u].append(v)
        neutral_neighbors[v].append(u)
    self._neutral_neighbors = dict(neutral_neighbors)

    # Annotate graph vertices.  Plateau size per node is the size of its
    # component (1 for singletons), assigned via vectorised scatter.
    self.graph.vs["plateau_id"] = node_to_plateau.tolist()
    plateau_sizes = np.ones(n_configs, dtype=np.int32)
    plateau_sizes[in_plateau_nodes] = comp_sizes[membership[in_plateau_nodes]]
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
    attributes is loaded.  Expects the 0-based ID scheme where
    ``plateau_id >= 0`` denotes a multi-member plateau and ``-1``
    denotes a singleton node.

    When ``self.configs`` is available, neutral neighbor adjacency
    (``_neutral_neighbors``) is also reconstructed so that
    ``determine_neighbor_fitness`` remains plateau-aware.
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
        if pid >= 0:
            plateaus[pid].append(node_idx)

    if not plateaus:
        self._has_plateaus = False
        self.n_plateau = 0
        return

    self._node_to_plateau = node_to_plateau
    self.plateaus = dict(plateaus)
    self.n_plateau = len(plateaus)
    self._has_plateaus = True

    if self.configs is not None and len(self.configs) == n:
        neutral_neighbors = defaultdict(list)
        for pid, members in self.plateaus.items():
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
