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
        Sorted list of all LO node indices.
    n_lo : int
        ``len(lo_index)`` — total number of LO *nodes*.
    plateau_lo_index : list[int]
        Plateau IDs (0-based) of plateaus that are local optima.
        Does **not** include single-point LOs.
    n_plateau_lo : int
        ``len(plateau_lo_index)``.
    n_peak : int
        Number of distinct peaks (``n_plateau_lo`` + number of
        single-point LOs).
    _peak_index : list[int]
        One representative node per peak (``min(members)`` for each
        plateau-LO, the node index itself for single-point LOs).
        Used internally by LON construction.
    """
    self = landscape

    if self.verbose:
        print(" - Determining local optima...")

    out_degrees = np.array(self.graph.outdegree())
    fitness = np.asarray(self.graph.vs["fitness"])

    if self._has_plateaus:
        # --- Plateau-aware LO detection ---
        eps = float(getattr(self, "epsilon", 0) or 0)
        plateau_is_lo = {}
        for pid, members in self.plateaus.items():
            plateau_best = (
                float(np.max(fitness[members]))
                if self.maximize
                else float(np.min(fitness[members]))
            )
            has_better_exit = False
            for node in members:
                for successor in self.graph.successors(node):
                    if int(self._node_to_plateau[successor]) == pid:
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

        in_plateau = self._node_to_plateau >= 0
        is_lo = (out_degrees == 0) & ~in_plateau  # single-point LOs

        for pid, is_lo_val in plateau_is_lo.items():
            if is_lo_val:
                members = self.plateaus[pid]
                is_lo[members] = True

        self.graph.vs["is_lo"] = is_lo.tolist()
        self.n_lo = int(is_lo.sum())
        self.lo_index = sorted(np.where(is_lo)[0].tolist())

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
        self.n_peak = self.n_plateau_lo + len(singleton_los)
        self._peak_index = sorted(plateau_reps + singleton_los)

        if self.n_lo == 0 and self.graph.vcount() > 0:
            raise RuntimeError(
                "Plateau-aware local-optimum detection found zero optima "
                "in a non-empty graph. This indicates inconsistent plateau "
                "state or a bug in the plateau classification logic."
            )

        if self.verbose:
            print(
                f"   - Found {self.n_lo} LO nodes across "
                f"{self.n_peak} distinct peaks "
                f"({self.n_plateau_lo} plateau-LOs, "
                f"{len(singleton_los)} single-point LOs)."
            )
    else:
        # --- No plateaus: classic out_degree == 0 ---
        is_lo = out_degrees == 0
        self.graph.vs["is_lo"] = is_lo.tolist()
        self.n_lo = int(is_lo.sum())
        self.lo_index = sorted(np.where(is_lo)[0].tolist())

        self.plateau_lo_index = []
        self.n_plateau_lo = 0
        self.n_peak = self.n_lo
        self._peak_index = list(self.lo_index)

        if self.n_lo == 0 and self.graph.vcount() > 0:
            raise RuntimeError(
                "Local-optimum detection found zero optima in a non-empty "
                "graph. This should be impossible for an acyclic improving "
                "graph and indicates a bug upstream."
            )

        if self.verbose:
            print(f"   - Found {self.n_lo} local optima.")
