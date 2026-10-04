"""Exact sparse preparation-time traversal for GraphFLA gamma statistics.

This helper groups observed two-variable squares by sparse genotype context,
then reuses GraphFLA's existing pooled moment kernels. It does not alter the
package implementation or the landscape population.
"""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd

from graphfla.analysis.epistasis.gamma import (
    _EMPTY,
    _gamma_grid_moments,
    _merge_gamma_contributions,
)


def _encode_signature(codes: np.ndarray, baseline: np.ndarray) -> bytes:
    """Pack sorted ``(position, allele-code)`` deviations into 3-byte tokens."""
    changed = np.flatnonzero(codes != baseline)
    packed = bytearray(3 * len(changed))
    for j, position in enumerate(changed):
        position = int(position)
        value = int(codes[position])
        offset = 3 * j
        packed[offset] = position & 0xFF
        packed[offset + 1] = (position >> 8) & 0xFF
        packed[offset + 2] = value
    return bytes(packed)


def _iter_signature(signature: bytes):
    for offset in range(0, len(signature), 3):
        position = signature[offset] | (signature[offset + 1] << 8)
        yield position, signature[offset + 2], offset


def _without_token(signature: bytes, offset: int) -> bytes:
    return signature[:offset] + signature[offset + 3 :]


def sparse_gamma_statistics(landscape, *, return_diagnostics=False):
    """Return exact ``gamma`` and ``gamma_star`` for an observed landscape.

    The traversal uses the modal allele at each variable site as an implicit
    reference. It emits only contexts containing observed non-reference
    substitutions, then adds a reference-allele row by exact signature lookup.
    Each resulting sparse grid is the same complete-square population consumed
    by GraphFLA's dense/sparse workers.

    Parameters
    ----------
    landscape : graphfla.landscape.Landscape
        Built landscape with unique observed genotypes and finite fitness.
    return_diagnostics : bool, default=False
        If true, include counts of sparse signatures, position pairs, and
        square contexts visited.

    Returns
    -------
    result : dict
        Keys ``gamma`` and ``gamma_star`` contain floats or NaN when their
        denominators are empty. With diagnostics enabled, a ``diagnostics``
        key contains structural counters.
    """
    landscape._check_built()
    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError("Built landscape requires a graph fitness attribute.")
    if landscape.n_vars < 2:
        result = {"gamma": math.nan, "gamma_star": math.nan}
        if return_diagnostics:
            result["diagnostics"] = {"rows": landscape.graph.vcount(), "variables": landscape.n_vars}
        return result

    frame = landscape.get_data()
    n_rows = len(frame)
    n_vars = len(landscape.data_types)
    if n_rows != landscape.graph.vcount():
        raise ValueError("Landscape data and graph node counts differ.")
    if n_vars > 65535:
        raise ValueError("Sparse signature encoding supports at most 65,535 variables.")

    # Codes and the row signature index are much smaller than materializing a
    # fresh N x (P-2) context matrix for every variable pair.
    Xcodes = np.empty((n_rows, n_vars), dtype=np.uint8)
    for j, column in enumerate(landscape.data_types):
        codes, _ = pd.factorize(frame[column], sort=True)
        if codes.size and (codes.min() < 0 or codes.max() > 255):
            raise ValueError(f"Unsupported missing or high-cardinality values in {column!r}.")
        Xcodes[:, j] = codes.astype(np.uint8, copy=False)

    baseline = np.empty(n_vars, dtype=np.uint8)
    for j in range(n_vars):
        baseline[j] = np.argmax(np.bincount(Xcodes[:, j].astype(np.intp)))

    signatures = []
    signature_to_row = {}
    rows_by_site = [[] for _ in range(n_vars)]
    for row_index in range(n_rows):
        signature = _encode_signature(Xcodes[row_index], baseline)
        if signature in signature_to_row:
            raise ValueError("Sparse traversal requires unique observed genotypes.")
        signature_to_row[signature] = row_index
        signatures.append(signature)
        for position, _allele, _offset in _iter_signature(signature):
            rows_by_site[position].append(row_index)
    del Xcodes

    fitness = frame["fitness"].to_numpy(dtype=np.float64, copy=False)
    total = _EMPTY
    visited_pairs = set()
    square_contexts = 0

    for focal_p in range(n_vars):
        # One group maps each observed p-allele at a fixed all-other-sites
        # context to its source row. Rows with p at the modal allele are added
        # by looking up the context signature after this group is built.
        p_groups = {}
        for row_index in rows_by_site[focal_p]:
            signature = signatures[row_index]
            for position, allele, offset in _iter_signature(signature):
                if position == focal_p:
                    context = _without_token(signature, offset)
                    group = p_groups.setdefault(context, {})
                    if allele in group:
                        raise ValueError("Duplicate genotype in a sparse p-context.")
                    group[allele] = row_index
                    break

        # Add a measured modal-p state where available. Inserting with its
        # actual modal allele code preserves the exact original allele labels.
        for context, alleles in p_groups.items():
            reference_row = signature_to_row.get(context)
            if reference_row is not None:
                alleles[int(baseline[focal_p])] = reference_row

        # A square needs at least two observed p alleles; discard other groups
        # before emitting q-background maps.
        p_groups = {ctx: alleles for ctx, alleles in p_groups.items() if len(alleles) >= 2}
        if not p_groups:
            continue

        # For each q allele and outer context, retain the p-effect group. A
        # measured q-reference group is linked by exact outer-signature lookup.
        q_groups = {}
        for context, p_alleles in p_groups.items():
            for q, q_allele, offset in _iter_signature(context):
                if q <= focal_p:
                    continue
                key = (q, _without_token(context, offset))
                states = q_groups.setdefault(key, {})
                if q_allele in states and states[q_allele] is not p_alleles:
                    raise ValueError("Duplicate q state in a sparse square context.")
                states[q_allele] = p_alleles
                reference_context = key[1]
                reference_group = p_groups.get(reference_context)
                if reference_group is not None:
                    states[int(baseline[q])] = reference_group

        for (focal_q, _outer_context), q_states in q_groups.items():
            q_allele_codes = sorted(q_states)
            if len(q_allele_codes) < 2:
                continue
            p_allele_codes = sorted({
                p_allele
                for p_alleles in q_states.values()
                for p_allele in p_alleles
            })
            if len(p_allele_codes) < 2:
                continue
            grid = np.full((1, len(p_allele_codes), len(q_allele_codes)), np.nan)
            p_index = {allele: i for i, allele in enumerate(p_allele_codes)}
            for j, q_allele in enumerate(q_allele_codes):
                for p_allele, row_index in q_states[q_allele].items():
                    grid[0, p_index[p_allele], j] = fitness[row_index]
            addition = _gamma_grid_moments(grid, statistic=None)
            addition = _merge_gamma_contributions(
                addition,
                _gamma_grid_moments(grid.swapaxes(1, 2), statistic=None),
            )
            total = _merge_gamma_contributions(total, addition)
            visited_pairs.add((focal_p, focal_q))
            square_contexts += 1

    num, den, sign_num, sign_den, _ = total
    result = {
        "gamma": num / den if den else math.nan,
        "gamma_star": sign_num / sign_den if sign_den else math.nan,
    }
    if return_diagnostics:
        result["diagnostics"] = {
            "rows": n_rows,
            "variables": n_vars,
            "sparse_mutations": sum(map(len, rows_by_site)),
            "candidate_position_pairs": len(visited_pairs),
            "complete_square_contexts": square_contexts,
        }
    return result
