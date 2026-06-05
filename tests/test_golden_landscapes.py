"""Golden test bed: graphfla vs hand-verified, graphfla-blind ground truth.

For each of 30 small, diverse landscapes (tests/_golden_catalog.py), the
ground-truth feature values were computed INDEPENDENTLY of graphfla by blind
solver agents and adjudicated by hand (tests/_golden_references.py). This test
builds each landscape in graphfla and asserts every computed feature against
that independent reference — so a passing run means the calculations are
*correct*, not merely that they run.

Adjudication during construction fixed two real graphfla discrepancies (FDC now
uses the nearest of tied global optima; DRI/ICI now handle minimisation) and
corrected a handful of reference values; see _golden_references.py.
"""
import math
import pytest

from _golden_catalog import LANDSCAPES
from _golden_references import REFERENCES
from _golden_build import extract

# ----------------------------------------------------------------------
# build + extract once per landscape (cached)
# ----------------------------------------------------------------------
_CACHE = {}
def gf(lid):
    if lid not in _CACHE:
        e = next(x for x in LANDSCAPES if x["id"] == lid)
        _CACHE[lid] = extract(e, seed=0)
    return _CACHE[lid]

# ----------------------------------------------------------------------
# comparison helpers
# ----------------------------------------------------------------------
def _coerce(x):
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("inf", "infinity"): return math.inf
        if s in ("-inf", "-infinity"): return -math.inf
        if s in ("nan", "none", "null"): return None
    return x
def _undef(x):
    return x is None or (isinstance(x, float) and math.isnan(x))
def feq(a, b, atol=1e-6, rtol=1e-5):
    a, b = _coerce(a), _coerce(b)
    if _undef(a) and _undef(b): return True
    if _undef(a) or _undef(b): return False
    if math.isinf(a) or math.isinf(b):
        return a == b or (abs(a) > 1e11 and abs(b) > 1e11 and (a > 0) == (b > 0))
    return abs(a - b) <= atol + rtol * abs(b)
def listeq(a, b):
    return [list(x) if isinstance(x, (list, tuple)) else x for x in a] == \
           [list(x) if isinstance(x, (list, tuple)) else x for x in b]
def seteq(a, b):
    return sorted(map(tuple, a)) == sorted(map(tuple, b))
def dicteq(a, b, atol=1e-6):
    a = {str(k): v for k, v in a.items()}; b = {str(k): v for k, v in b.items()}
    return set(a) == set(b) and all(feq(a[k], b[k], atol=atol) for k in a)

# ----------------------------------------------------------------------
# reference-key -> (graphfla getter, comparison)
# ----------------------------------------------------------------------
def _cls(name): return lambda ex: ex.get("classify", {}).get(name)
SPEC = {
    "n_configs": (lambda ex: ex["n_configs"], "exact"),
    "n_edges": (lambda ex: ex["n_edges"], "exact"),
    "edges": (lambda ex: ex["edges"], "set"),
    "neutral_pairs": (lambda ex: ex["neutral_pairs"], "set"),
    "out_degree": (lambda ex: ex["out_degree"], "list"),
    "in_degree": (lambda ex: ex["in_degree"], "list"),
    "go": (lambda ex: ex["go"], "exact"),
    "n_lo": (lambda ex: ex["n_lo"], "exact"),
    "lo_index": (lambda ex: ex["lo_index"], "list"),
    "n_lo_members": (lambda ex: ex["n_lo_members"], "exact"),
    "lo_ratio": (lambda ex: ex["lo_ratio"], "float"),
    "dist_to_go": (lambda ex: ex["dist_to_go"], "list"),
    "mean_dist_go": (lambda ex: ex["mean_dist_go"], "float"),
    "global_optima_accessibility": (lambda ex: ex["global_optima_accessibility"], "float"),
    "mean_path_lengths_go": (lambda ex: ex["mean_path_lengths_go"], "dict"),
    "gamma": (lambda ex: ex["gamma"], "float"),
    "gamma_star": (lambda ex: ex["gamma_star"], "float"),
    "global_idiosyncratic_index": (lambda ex: ex["global_idiosyncratic_index"], "float"),
    "r_s_ratio": (lambda ex: ex["r_s_ratio"], "float"),
    "gradient_intensity": (lambda ex: ex["gradient_intensity"], "float"),
    "autocorrelation": (lambda ex: ex["autocorrelation"], "loose"),
    "neighbor_fit_corr": (lambda ex: ex["neighbor_fit_corr"], "float"),
    "fitness_distance_corr": (lambda ex: ex["fitness_distance_corr"], "float"),
    "neutrality": (lambda ex: ex["neutrality"], "float"),
    "evol_enhance_mutations": (lambda ex: ex["evol_enhance_mutations"], "float"),
    "diminishing_returns_index": (lambda ex: ex["diminishing_returns_index"], "float"),
    "increasing_costs_index": (lambda ex: ex["increasing_costs_index"], "float"),
    "higher_order": (lambda ex: ex["higher_order"], "dict"),
    "magnitude": (_cls("magnitude epistasis"), "float"),
    "sign": (_cls("sign epistasis"), "float"),
    "reciprocal": (_cls("reciprocal sign epistasis"), "float"),
    "positive": (_cls("positive epistasis"), "float"),
    "negative": (_cls("negative epistasis"), "float"),
    "bypass_proportion": (lambda ex: ex.get("bypass", {}).get("bypass_proportion"), "frac0"),
    "bypass_mean_length": (lambda ex: ex.get("bypass", {}).get("average_bypass_length"), "float"),
    "basin_fit_corr": (lambda ex: ex["basin_fit_corr"], "float"),
    "fitness_flattening_index": (lambda ex: ex["fitness_flattening_index"], "float"),
    "size_basin_greedy": (lambda ex: ex["size_basin_greedy"], "dict"),
    "radius_basin_greedy": (lambda ex: ex["radius_basin_greedy"], "dict"),
    "walsh_order2_zero": (lambda ex: (abs(ex["walsh_maxabs_by_order"]["2"]) < 1e-9)
                          if ex.get("walsh_maxabs_by_order") else None, "exact"),
}

def compare(kind, gfv, ref):
    if kind == "exact": return gfv == ref
    if kind == "float": return feq(gfv, ref)
    if kind == "loose": return feq(gfv, ref, atol=0.08, rtol=0)
    if kind == "frac0":  # a proportion where "no instances" may be None or 0.0
        return feq(0.0 if _undef(gfv) else gfv, 0.0 if _undef(ref) else ref)
    if kind == "list": return listeq(gfv, ref)
    if kind == "set": return seteq(gfv, ref)
    if kind == "dict": return dicteq(gfv, ref)
    raise AssertionError(kind)

# parametrize over every (landscape, frozen feature) pair we know how to check
PARAMS = [(lid, feat) for lid, ref in REFERENCES.items() for feat in ref if feat in SPEC]


@pytest.mark.parametrize("lid,feat", PARAMS, ids=[f"{l}-{f}" for l, f in PARAMS])
def test_golden_feature(lid, feat):
    getter, kind = SPEC[feat]
    ex = gf(lid)
    gfv = getter(ex)
    assert not (isinstance(gfv, str) and gfv.startswith("ERR")), f"graphfla error: {gfv}"
    ref = REFERENCES[lid][feat]
    assert compare(kind, gfv, ref), f"{lid}.{feat}: graphfla={gfv!r} vs reference={ref!r}"


@pytest.mark.parametrize("lid", [e["id"] for e in LANDSCAPES])
def test_golden_basin_invariants(lid):
    """Greedy basins partition the configuration space onto local optima."""
    ex = gf(lid)
    bidx = ex["basin_index"]
    lo = set(ex["lo_index"])
    assert len(bidx) == ex["n_configs"]
    # every node's basin representative is a local-optimum node
    assert set(bidx) <= lo, f"{lid}: basin reps {set(bidx) - lo} not in local optima"
    # greedy basin sizes sum to the number of configurations
    assert sum(ex["size_basin_greedy"].values()) == ex["n_configs"]
