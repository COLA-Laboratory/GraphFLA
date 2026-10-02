"""Offline Wagner (2023) replay and independent corrected-formula checks.

Run ``python -m validation.ee_mutations``. Author code is inspected, not executed.
The oracle enumerates substitutions and explicitly collects each neighborhood;
production consumes graph pairs and shares node/position moments instead.
"""

import hashlib
import json
import tempfile
from itertools import product
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

from graphfla.analysis import evolvability_enhancing_fraction
from graphfla.analysis._evolvability import _ee_statistics
from graphfla.landscape import Landscape


FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/literature/wagner2023"


def load_dataset(kind):
    meta = json.loads((FIXTURE / "provenance.json").read_text())["datasets"][kind]
    for key, hash_key in [("input_fixture", "input_sha256"),
                          ("author_fixture", "author_fixture_sha256")]:
        path = FIXTURE / meta[key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta[hash_key]
    frame = pd.read_csv(FIXTURE / meta["input_fixture"], float_precision="round_trip")
    with np.load(FIXTURE / meta["author_fixture"], allow_pickle=False) as source:
        author = {key: source[key] for key in source.files}
    assert len(frame) == meta["variants"] and frame.sequence.is_unique
    return frame, author, meta


def enumerate_pairs(sequences, protein=False):
    """Single substitutions; standard genetic code restricts the protein graph."""
    # Standard code in T,C,A,G lexicographic codon order (NCBI table 1).
    amino_acids = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
    codons = dict(zip(map("".join, product("TCAG", repeat=3)), amino_acids))
    allowed = set()
    for c, a in codons.items():
        for d, b in codons.items():
            if a != "*" and b != "*" and a != b:
                if sum(x != y for x, y in zip(c, d)) == 1:
                    allowed.add((a, b))
    lookup = {seq: i for i, seq in enumerate(sequences)}
    alphabets = [sorted({s[p] for s in sequences}) for p in range(len(sequences[0]))]
    pairs = []
    for i, seq in enumerate(sequences):
        for pos, alphabet in enumerate(alphabets):
            for allele in alphabet:
                if protein and (seq[pos], allele) not in allowed:
                    continue
                other = seq[:pos] + allele + seq[pos + 1:]
                j = lookup.get(other)
                if j is not None and i < j:
                    pairs.append((i, j))
    return sorted(pairs)


def reference_statistics(
    configs, fitness, pairs, variance=None, author_bug=False, fdr=0.01
):
    """Literal per-mutation oracle, independent of production helpers."""
    adjacency = [set() for _ in fitness]
    for u, v in pairs:
        adjacency[u].add(v)
        adjacency[v].add(u)
    directed = list(pairs) + [(v, u) for u, v in pairs]
    rows = []
    for u, v in directed:
        positions = [p for p, (a, b) in enumerate(zip(configs[u], configs[v])) if a != b]
        assert len(positions) == 1
        p = positions[0]
        left = sorted(w for w in adjacency[u] if configs[w][p] == configs[u][p])
        right = sorted(w for w in adjacency[v] if configs[w][p] == configs[v][p])
        n = min(len(left), len(right))
        if not left or not right:
            rows.append((u, v, fitness[v]-fitness[u], np.nan, np.nan, n))
            continue
        m1, m2 = np.mean(fitness[left]), np.mean(fitness[right])
        if variance is None:
            v1, v2 = np.var(fitness[left]), np.var(fitness[right])
        else:
            v1 = sum(variance[w] for w in left) / len(left)**2
            v2 = sum(variance[w] for w in right) / len(right)**2
        rows.append((u, v, fitness[v]-fitness[u], m2-m1,
                     2*v2 if author_bug else v1+v2, n))
    rows = np.asarray(rows).reshape(-1, 6)
    dw, dm, variances, n = rows[:, 2:].T
    with np.errstate(divide="ignore", invalid="ignore"):
        p_effect = ttest_ind_from_stats(dm, np.sqrt(variances), n,
                                       dw, 0, 2, equal_var=False).pvalue
        p_zero = ttest_ind_from_stats(dm, np.sqrt(variances), n,
                                     0, 0, 2, equal_var=False).pvalue
    # Specify the limiting and untestable cases independently of SciPy's NaNs.
    for pval, diff in [(p_effect, dm-dw), (p_zero, dm)]:
        pval[n < 2] = np.nan
        mask = (n >= 2) & (variances == 0)
        pval[mask] = np.where(diff[mask] == 0, 1.0, 0.0)

    def decisions(pvalues):
        values = sorted(1.0 if not np.isfinite(p) else p for p in pvalues)
        cutoff = -1.0
        for rank, p in enumerate(values, 1):
            threshold = fdr * rank / len(values)
            passes = p < threshold if author_bug else p <= threshold
            if passes:
                cutoff = p
        return pvalues <= cutoff

    flag_effect = np.where(decisions(p_effect), np.sign(dm-dw), 0).astype(int)
    flag_zero = np.where(decisions(p_zero), np.sign(dm), 0).astype(int)
    return dict(source=rows[:, 0].astype(int), target=rows[:, 1].astype(int),
                effect=dw, delta_mean=dm, p_effect=p_effect, p_zero=p_zero,
                flag_effect=flag_effect, flag_zero=flag_zero)


def counts(effect, flag_effect, flag_zero):
    return {
        "beneficial": int(np.sum((effect > 0) & (flag_effect == 1))),
        "deleterious": int(np.sum((effect < 0) & (flag_zero == 1))),
        "neutral": int(np.sum((effect == 0) & (flag_zero == 1))),
    }


def import_landscape(frame, pairs, path):
    """Public GraphML adapter preserves the study's exact neighborhood model."""
    f = frame.fitness.to_numpy()
    edges = [(u, v) if f[u] < f[v] else (v, u) for u, v in pairs]
    assert all(f[u] != f[v] for u, v in pairs)
    graph = ig.Graph(n=len(frame), edges=edges, directed=True)
    graph.vs["fitness"] = f.tolist()
    columns = [f"site_{p}" for p in range(len(frame.sequence.iloc[0]))]
    for pos, column in enumerate(columns):
        graph.vs[column] = frame.sequence.str[pos].tolist()
    graph["data_types_data"] = repr(dict.fromkeys(columns, "categorical"))
    graph["maximize"] = True
    graph["epsilon"] = "0"
    graph.write_graphml(str(path))
    return Landscape.build_from_graph(str(path), verbose=False)


def reproduce_dataset(kind):
    frame, author, meta = load_dataset(kind)
    case_path = Path(__file__).with_name("cases") / f"wagner.{kind}.ee.author.v1.json"
    expected = json.loads(case_path.read_text())["expected"]
    f = frame.fitness.to_numpy()
    configs = np.asarray([list(s) for s in frame.sequence])
    pairs = enumerate_pairs(frame.sequence.tolist(), protein=kind == "protein")
    assert 2 * len(pairs) == meta["ordered_pairs"] == expected["ordered_pairs"]
    variance = 6*frame.fitness_se.to_numpy()**2 if kind == "rna" else None
    replay = reference_statistics(configs, f, pairs, variance, author_bug=True)
    order = {(u, v): i for i, (u, v) in enumerate(zip(replay["source"], replay["target"]))}
    idx = [order[u, v] for u, v in zip(author["source"], author["target"])]
    for name in ["flag_effect", "flag_zero"]:
        np.testing.assert_array_equal(replay[name][idx], author[name])
    # Recreate the published text output round trip, rather than classify
    # full precision effects as though the notebook had read those values.
    rounded = np.asarray([float(f"{x:.4f}") for x in replay["effect"]])
    rounded_differences = rounded[idx] != author["rounded_effect"]
    # One RNA pair (both orientations) differs at the last printed digit.
    # Recovered delta is 0.14295000039940747, only 4e-10 above the rounding
    # boundary, whereas the author table stores 0.1429. The exact original
    # input-file precision is unavailable. Do not claim byte-for-byte replay;
    # the effect class and every significance flag still agree.
    assert int(rounded_differences.sum()) == (2 if kind == "rna" else 0)
    if rounded_differences.any():
        midpoint = np.abs(replay["effect"][idx][rounded_differences]) * 1e4
        np.testing.assert_allclose(midpoint % 1, .5, atol=4e-6, rtol=0)
        np.testing.assert_allclose(np.abs(replay["effect"][idx][rounded_differences]),
                                   .14295000039940747, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(np.sign(rounded[idx]), np.sign(author["rounded_effect"]))
    historical = counts(rounded, replay["flag_effect"], replay["flag_zero"])
    assert historical["beneficial"] == expected["beneficial"]
    assert historical["deleterious"] == expected["deleterious"]

    oracle = reference_statistics(configs, f, pairs, variance)
    actual = _ee_statistics(configs, f, pairs, fitness_variance=variance)
    for name in ["p_effect", "p_zero"]:
        np.testing.assert_allclose(actual[name], oracle[name], atol=2e-12, rtol=2e-10)
    expected_ee = np.where(oracle["effect"] > 0, oracle["flag_effect"] == 1,
                          oracle["flag_zero"] == 1)
    np.testing.assert_array_equal(actual.ee, expected_ee)
    corrected = counts(oracle["effect"], oracle["flag_effect"], oracle["flag_zero"])

    # The current scalar API has no measurement-variance input. Check it
    # against a separate neighborhood-variation oracle also on the RNA graph.
    generic = oracle if variance is None else reference_statistics(configs, f, pairs)
    generic_counts = counts(generic["effect"], generic["flag_effect"], generic["flag_zero"])
    with tempfile.TemporaryDirectory() as tmp:
        landscape = import_landscape(frame, pairs, Path(tmp)/"landscape.graphml")
        proportion = evolvability_enhancing_fraction(landscape)
    expected_proportion = sum(generic_counts.values())/meta["ordered_pairs"]
    np.testing.assert_allclose(proportion, expected_proportion, rtol=0, atol=1e-15)
    return {
        "variants": len(frame), "ordered_pairs": meta["ordered_pairs"],
        "author_signed_flags_compared": 2*meta["ordered_pairs"],
        "author_flag_mismatches": 0,
        "four_decimal_effect_differences": int(rounded_differences.sum()),
        "paper_replay_rounded_effects": historical,
        "author_procedure_full_precision": counts(replay["effect"], replay["flag_effect"], replay["flag_zero"]),
        "corrected_symmetric_variance": corrected,
        "corrected_kernel_oracle_mismatches": 0,
        "public_neighborhood_variation_counts": generic_counts,
        "public_proportion": proportion,
    }


if __name__ == "__main__":
    print(json.dumps({kind: reproduce_dataset(kind) for kind in ["protein", "rna"]}, indent=2))
