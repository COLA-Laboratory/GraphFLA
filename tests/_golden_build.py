"""Build each golden-catalog landscape in graphfla and extract every feature
value, in the same shape as the frozen references (tests/_golden_references.py).
Used only by tests/test_golden_landscapes.py."""
import io, contextlib
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

from graphfla.landscape import (
    BooleanLandscape, OrdinalLandscape, Landscape,
    DNALandscape, RNALandscape, ProteinLandscape,
)
from graphfla.analysis import (
    lo_ratio, autocorrelation, r_s_ratio, gradient_intensity,
    fitness_distance_corr, neighbor_fit_corr, basin_fit_corr, fitness_flattening_index,
    gamma_statistic, gamma_star, classify_epistasis, higher_order_epistasis,
    walsh_hadamard_coefficient, global_idiosyncratic_index, extradimensional_bypass_analysis,
    diminishing_returns_index, increasing_costs_index,
    global_optima_accessibility, local_optima_accessibility, mean_path_lengths,
    mean_path_lengths_go, mean_dist_go, neutrality, evol_enhance_mutations,
)


def build(e):
    k = e["kind"]; gts = e["genotypes"]; f = list(e["fitness"])
    mx = e["maximize"]; eps = e["epsilon"]; npos = e["n_positions"]
    kw = dict(epsilon=eps, verbose=False)
    if k == "boolean":
        ls = BooleanLandscape(maximize=mx); ls.build_from_data([tuple(g) for g in gts], f, **kw)
    elif k == "ordinal":
        X = pd.DataFrame(gts, columns=[f"x{i}" for i in range(npos)])
        ls = OrdinalLandscape(maximize=mx); ls.build_from_data(X, f, **kw)
    elif k in ("categorical", "mixed"):
        X = pd.DataFrame(gts, columns=[f"x{i}" for i in range(npos)])
        dt = {f"x{i}": e["position_types"][i] for i in range(npos)}
        ls = Landscape(kind="default", maximize=mx)
        ls.build_from_data(X, f, data_types=dt, neighborhood_strategy="active", **kw)
    else:
        ab = e["alphabet"]; seqs = ["".join(ab[i] for i in g) for g in gts]
        cls = {"dna": DNALandscape, "rna": RNALandscape, "protein": ProteinLandscape}[k]
        ls = cls(maximize=mx); ls.build_from_data(seqs, f, **kw)
    # Trigger the lazy, cached analyses the extractor reads directly (this
    # replaces the deprecated eager calculate_* build flags).
    ls.basins; ls.accessible_paths; ls.dist_to_go; ls.neighbor_fitness
    return ls


def _try(d, key, fn):
    try:
        d[key] = fn()
    except Exception as ex:
        d[key] = f"ERR:{type(ex).__name__}:{str(ex)[:60]}"


def extract(e, seed=0):
    """Return {feature_key: graphfla_value} for one landscape entry."""
    ls = build(e)
    g = ls.graph
    out = {}
    # --- structure / optima ---
    out["n_configs"] = ls.n_configs
    out["n_edges"] = g.ecount()
    out["edges"] = sorted([list(t) for t in g.get_edgelist()])
    nn = getattr(ls, "_neutral_neighbors", None) or {}
    npairs = set()
    for v, us in nn.items():
        for u in us:
            npairs.add((min(v, u), max(v, u)))
    out["neutral_pairs"] = sorted([list(p) for p in npairs])
    out["out_degree"] = list(g.outdegree())
    out["in_degree"] = list(g.indegree())
    out["go"] = ls.go_index
    out["n_lo"] = ls.n_lo
    out["lo_index"] = list(ls.lo_index)
    out["n_lo_members"] = ls.n_lo_members
    _try(out, "lo_ratio", lambda: lo_ratio(ls))
    # --- basins ---
    bidx = g.vs["basin_index"]
    out["basin_index"] = list(bidx)
    sg = g.vs["size_basin_greedy"]; rg = g.vs["radius_basin_greedy"]
    out["size_basin_greedy"] = {int(b): int(sg[i]) for i, b in enumerate(bidx) if b == i or i in ls.lo_index}
    # size per representative
    rep_size = {}; rep_rad = {}
    for i, b in enumerate(bidx):
        rep_size[int(b)] = int(sg[i]); rep_rad[int(b)] = max(rep_rad.get(int(b), 0), int(rg[i]))
    out["size_basin_greedy"] = rep_size
    out["radius_basin_greedy"] = rep_rad
    if "size_basin_accessible" in g.vs.attributes():
        sa = g.vs["size_basin_accessible"]
        out["size_basin_accessible"] = {int(i): int(sa[i]) for i in ls.lo_index}
    # --- navigability ---
    if "dist_go" in g.vs.attributes():
        out["dist_to_go"] = list(g.vs["dist_go"])
    _try(out, "global_optima_accessibility", lambda: global_optima_accessibility(ls))
    _try(out, "mean_path_lengths_go", lambda: mean_path_lengths(ls, lo=ls.go_index))
    _try(out, "mean_dist_go", lambda: mean_dist_go(ls))
    # --- gamma ---
    _try(out, "gamma", lambda: gamma_statistic(ls, n_jobs=1))
    _try(out, "gamma_star", lambda: gamma_star(ls, n_jobs=1))
    # --- classify + bypass ---
    _try(out, "classify", lambda: classify_epistasis(ls))
    _try(out, "bypass", lambda: extradimensional_bypass_analysis(ls))
    # --- higher-order / walsh ---
    ho = {}
    for o in (1, 2, 3):
        if o <= ls.n_vars:
            try:
                ho[str(o)] = higher_order_epistasis(ls, order=o)
            except Exception as ex:
                ho[str(o)] = f"ERR:{type(ex).__name__}"
    out["higher_order"] = ho
    if e["kind"] == "boolean":
        def walsh_maxabs():
            with contextlib.redirect_stdout(io.StringIO()):
                c = walsh_hadamard_coefficient(ls, max_order=2)
            return {str(o): (max(abs(v) for v in c[o].values()) if c.get(o) else 0.0) for o in (0, 1, 2)}
        _try(out, "walsh_maxabs_by_order", walsh_maxabs)
    else:
        out["walsh_maxabs_by_order"] = None
    # --- idiosyncratic ---
    _try(out, "global_idiosyncratic_index", lambda: global_idiosyncratic_index(ls, n_jobs=1))
    # --- ruggedness ---
    _try(out, "autocorrelation", lambda: autocorrelation(ls, walk_length=30, walk_times=3000, seed=seed))
    _try(out, "r_s_ratio", lambda: r_s_ratio(ls))
    _try(out, "gradient_intensity", lambda: gradient_intensity(ls))
    # --- correlations ---
    _try(out, "fitness_distance_corr", lambda: fitness_distance_corr(ls))
    _try(out, "neighbor_fit_corr", lambda: neighbor_fit_corr(ls))
    _try(out, "basin_fit_corr", lambda: basin_fit_corr(ls))
    _try(out, "fitness_flattening_index", lambda: fitness_flattening_index(ls))
    # --- robustness ---
    _try(out, "neutrality", lambda: neutrality(ls))
    _try(out, "evol_enhance_mutations", lambda: evol_enhance_mutations(ls))
    # --- diminishing / increasing ---
    _try(out, "diminishing_returns_index", lambda: diminishing_returns_index(ls))
    _try(out, "increasing_costs_index", lambda: increasing_costs_index(ls))
    return out


