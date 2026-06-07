import numpy as np


def determine_local_optima(landscape):
    """Identify local optima with plateau-aware semantics.

    A node is a local optimum if it has no way to improve fitness, taking
    neutral plateaus into account:

    * **Single-point LO** — ``out_degree == 0`` and not a member of any
      multi-member plateau.
    * **Plateau-LO member** — belongs to a multi-member plateau where
      *no* member has an improving edge to a node outside the plateau
      whose fitness beats the plateau's best by more than ``epsilon``
      (same threshold as neutral-vs-improving in graph construction).

    Nodes that have ``out_degree == 0`` but belong to a plateau with only
    “small” external gains (within ``epsilon`` of the plateau best) do
    **not** disqualify the plateau from being a local optimum.

    Attributes set on *landscape*
    -----------------------------
    is_lo : vertex attribute (bool)
        True for every LO node (single-point LOs + all plateau-LO members).
    lo_index : list[int]
        Sorted list of all LO *node* indices.
    n_lo : int
        Number of distinct local optima (each plateau-LO counts once, plus
        every single-point LO). This is the "number of local optima".
    n_lo_members : int
        ``len(lo_index)`` — total number of LO member *nodes* (every member of
        every plateau-LO plus single-point LOs).
    plateau_lo_index : list[int]
        Plateau IDs (0-based) of plateaus that are local optima.
        Does **not** include single-point LOs.
    n_plateau_lo : int
        ``len(plateau_lo_index)``.
    _peak_index : list[int]
        One representative node per peak (``min(members)`` for each
        plateau-LO, the node index itself for single-point LOs).
        Used internally by LON construction.
    """
    self = landscape

    if self.verbose:
        print(" - Determining local optima...")

    out_degrees = np.asarray(self.graph.outdegree())

    if self._has_plateaus:
        # --- Plateau-aware LO detection ---
        # ``fitness`` is only needed on the plateau path, so it is read here
        # rather than unconditionally (the no-plateau branch never uses it).
        eps = float(getattr(self, "epsilon", 0) or 0)
        fitness = np.asarray(self.graph.vs["fitness"], dtype=np.float64)
        node_to_plateau = self._node_to_plateau

        # Per-plateau best fitness over its members, computed in one grouped
        # reduction instead of a per-plateau ``np.max`` call. The array is
        # sized by the largest plateau id so it stays correct even if ids are
        # not perfectly dense (e.g. restored from a saved graph).
        in_plateau_mask = node_to_plateau >= 0
        member_pids = node_to_plateau[in_plateau_mask]
        member_fits = fitness[in_plateau_mask]
        n_pid = int(member_pids.max()) + 1
        if self.maximize:
            plateau_best_arr = np.full(n_pid, -np.inf, dtype=np.float64)
            np.maximum.at(plateau_best_arr, member_pids, member_fits)
        else:
            plateau_best_arr = np.full(n_pid, np.inf, dtype=np.float64)
            np.minimum.at(plateau_best_arr, member_pids, member_fits)

        successors = self.graph.successors
        plateau_is_lo = {}
        for pid, members in self.plateaus.items():
            plateau_best = float(plateau_best_arr[pid])
            has_better_exit = False
            for node in members:
                for successor in successors(node):
                    if int(node_to_plateau[successor]) == pid:
                        continue
                    successor_fit = float(fitness[successor])
                    if self.maximize:
                        if successor_fit > plateau_best + eps:
                            has_better_exit = True
                            break
                    elif successor_fit < plateau_best - eps:
                        has_better_exit = True
                        break
                if has_better_exit:
                    break
            plateau_is_lo[pid] = not has_better_exit

        in_plateau = in_plateau_mask
        is_lo = (out_degrees == 0) & ~in_plateau  # single-point LOs

        for pid, is_lo_val in plateau_is_lo.items():
            if is_lo_val:
                members = self.plateaus[pid]
                is_lo[members] = True

        self.graph.vs["is_lo"] = is_lo.tolist()
        self.lo_index = sorted(np.where(is_lo)[0].tolist())
        self.n_lo_members = int(is_lo.sum())

        self.plateau_lo_index = sorted(
            pid for pid, v in plateau_is_lo.items() if v
        )
        self.n_plateau_lo = len(self.plateau_lo_index)

        singleton_los = sorted(
            np.where((out_degrees == 0) & ~in_plateau & is_lo)[0].tolist()
        )
        plateau_reps = sorted(
            min(self.plateaus[pid]) for pid in self.plateau_lo_index
        )
        # n_lo counts distinct optima: each plateau-LO is one optimum.
        self.n_lo = self.n_plateau_lo + len(singleton_los)
        self._peak_index = sorted(plateau_reps + singleton_los)

        if self.n_lo == 0 and self.graph.vcount() > 0:
            raise RuntimeError(
                "Plateau-aware local-optimum detection found zero optima "
                "in a non-empty graph. This indicates inconsistent plateau "
                "state or a bug in the plateau classification logic."
            )

        if self.verbose:
            print(
                f"   - Found {self.n_lo} local optima "
                f"({self.n_lo_members} member nodes: "
                f"{self.n_plateau_lo} plateau-LOs + "
                f"{len(singleton_los)} single-point LOs)."
            )
    else:
        # --- No plateaus: classic out_degree == 0 ---
        is_lo = out_degrees == 0
        self.graph.vs["is_lo"] = is_lo.tolist()
        self.lo_index = sorted(np.where(is_lo)[0].tolist())
        # No plateaus: every LO is a single node, so optima == member nodes.
        self.n_lo_members = int(is_lo.sum())
        self.n_lo = self.n_lo_members

        self.plateau_lo_index = []
        self.n_plateau_lo = 0
        self._peak_index = list(self.lo_index)

        if self.n_lo == 0 and self.graph.vcount() > 0:
            raise RuntimeError(
                "Local-optimum detection found zero optima in a non-empty "
                "graph. This should be impossible for an acyclic improving "
                "graph and indicates a bug upstream."
            )

        if self.verbose:
            print(f"   - Found {self.n_lo} local optima.")
