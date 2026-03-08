import numpy as np


def determine_local_optima(landscape):
    """Identifies local optima nodes in the landscape graph.

    A node is a per-node local optimum if its out-degree is 0 (no strictly
    improving neighbor). When plateaus are present (``epsilon > 0``), an
    additional plateau-level classification is computed: a plateau is a
    local optimum if **no** member has an improving edge to a node outside
    the plateau.

    Per-node results are stored in ``n_lo`` / ``lo_index`` / ``is_lo``
    for backward compatibility. Plateau-level results are stored in
    ``n_plateau_lo`` / ``plateau_lo_index`` / ``is_plateau_lo``.
    """
    self = landscape

    if self.verbose:
        print(" - Determining local optima...")

    # --- Per-node LO detection (unchanged) ---
    out_degrees = np.array(self.graph.outdegree())
    is_lo = out_degrees == 0
    self.graph.vs["is_lo"] = is_lo.tolist()
    self.n_lo = int(is_lo.sum())
    self.lo_index = np.where(is_lo)[0].tolist()

    if self.verbose:
        print(f"   - Found {self.n_lo} per-node local optima.")

    # --- Plateau-level LO detection ---
    if self._has_plateaus:
        n_vertices = self.graph.vcount()
        plateau_is_lo = {}

        for pid, members in self._plateaus.items():
            has_external_exit = False
            for node in members:
                for successor in self.graph.successors(node):
                    if int(self._node_to_plateau[successor]) != pid:
                        has_external_exit = True
                        break
                if has_external_exit:
                    break
            plateau_is_lo[pid] = not has_external_exit

        # For nodes not in any multi-member plateau, fall back to per-node is_lo
        is_plateau_lo_arr = is_lo.copy()
        for pid, is_lo_val in plateau_is_lo.items():
            members = self._plateaus[pid]
            is_plateau_lo_arr[members] = is_lo_val

        self.graph.vs["is_plateau_lo"] = is_plateau_lo_arr.tolist()

        plateau_lo_pids = [pid for pid, v in plateau_is_lo.items() if v]
        singleton_lo_mask = is_lo.copy()
        for pid in plateau_is_lo:
            members = self._plateaus[pid]
            singleton_lo_mask[members] = False
        singleton_los = np.where(singleton_lo_mask)[0].tolist()

        self.n_plateau_lo = len(plateau_lo_pids) + len(singleton_los)
        self.plateau_lo_index = plateau_lo_pids + singleton_los

        if self.verbose:
            print(f"   - Found {self.n_plateau_lo} plateau-level local optima.")
