"""Literal Lyons SD-ratio references; no GraphFLA imports or input loading."""

from itertools import permutations

import numpy as np


def mutation_effects(configs, fitness, positions=None):
    """Enumerate substitutions by tuple lookup, independently of grouped codes."""
    configs = [tuple(row) for row in configs]
    lookup = dict(zip(configs, fitness))
    positions = range(len(configs[0])) if positions is None else positions
    effects = {}
    for j in positions:
        for a, b in permutations(sorted({row[j] for row in configs}), 2):
            values = []
            for x, f in zip(configs, fitness):
                y = x[:j] + (b,) + x[j + 1 :]
                if x[j] == a and y in lookup:
                    values.append(lookup[y] - f)
            effects[j, a, b] = np.asarray(values)
    return effects


def sequence_effects(sequences, fitness, positions):
    """Independent sequence substitution, also returning improving Hamming edges."""
    lookup = {s: i for i, s in enumerate(sequences)}
    effects, edges = {}, []
    for pos in positions:
        for a, b in permutations("ACGT", 2):
            values = []
            for i, seq in enumerate(sequences):
                if seq[pos] != a:
                    continue
                j = lookup.get(seq[:pos] + b + seq[pos + 1 :])
                if j is not None:
                    values.append(fitness[j] - fitness[i])
                    if fitness[i] < fitness[j]:
                        edges.append((i, j))
            effects[pos, a, b] = np.asarray(values)
    return effects, edges


def control_ratios(effects, pool, seed=None, author_seeds=False, min_pairs=3):
    """One same-size, with-replacement null per eligible directed mutation."""
    pool = np.asarray(pool)
    rng = np.random.RandomState(seed)
    values = {}
    for key, effect in effects.items():
        n = len(effect)
        if n < min_pairs:
            continue
        if author_seeds:
            rng = np.random.RandomState(n**2 + 3)
        pairs = rng.choice(len(pool), size=(n, 2), replace=True)
        null = [pool[j] - pool[i] for i, j in pairs]
        sd = np.std(null)
        values[key] = float(np.std(effect) / sd) if sd else np.nan
    return values


def landscape_mean(configs, fitness, seed, min_pairs=3):
    effects = mutation_effects(configs, fitness)
    ratios = control_ratios(effects, fitness, seed=seed, min_pairs=min_pairs)
    return float(np.mean(list(ratios.values()))) if ratios else np.nan
