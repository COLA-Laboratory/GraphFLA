"""Epistasis motifs: classification and extradimensional bypass.

4-node-motif analysis on the landscape graph -- sign / magnitude /
reciprocal epistasis classification and reciprocal-sign bypass detection.
"""

import contextlib
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import igraph as ig
import pandas as pd

from .._utils import _pythonize
import logging

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _seeded_igraph(seed):
    """Temporarily route igraph's RNG through a locally-seeded generator.

    Makes the sampling in ``motifs_randesu(cut_prob=...)`` reproducible when a
    seed is given. Restores igraph's default generator afterwards so global
    state is left untouched. A no-op when ``seed is None``.
    """
    if seed is None:
        yield
        return
    ig.set_random_number_generator(random.Random(seed))
    try:
        yield
    finally:
        ig.set_random_number_generator(None)


# Module-level workers for process-based parallelism (must be importable so
# joblib's loky backend can pickle them; large code arrays are auto-memmapped).
def _assign_roles_for_epistasis_igraph(graph, squares):
    """Assigns roles within collected square motif instances."""
    squares_with_roles = []
    if "fitness" not in graph.vs.attributes():
        raise ValueError(
            "igraph.Graph must have a 'fitness' vertex attribute for role assignment."
        )

    for square_nodes in squares:
        if len(square_nodes) != 4:
            continue  # defensive: should already be filtered

        try:
            nodes_in_square = list(square_nodes)
            fitness_values_list = graph.vs[nodes_in_square]["fitness"]
            fitness_values = {
                node: fitness
                for node, fitness in zip(nodes_in_square, fitness_values_list)
            }

            double_mutant = max(fitness_values, key=fitness_values.get)
            all_predecessors = graph.predecessors(double_mutant)
            square_set = set(nodes_in_square)
            single_mutants = [p for p in all_predecessors if p in square_set]

            if len(single_mutants) != 2:
                continue  # not the expected square structure

            wild_type_set = square_set - set(single_mutants) - {double_mutant}
            if len(wild_type_set) != 1:
                continue  # WT not uniquely identifiable
            wild_type = list(wild_type_set)[0]

            single_mutants.sort()  # stable ordering

            squares_with_roles.append(
                {
                    "wild_type": wild_type,
                    "single_mutant_1": single_mutants[0],
                    "single_mutant_2": single_mutants[1],
                    "double_mutant": double_mutant,
                    "fitness_values": fitness_values,
                }
            )
        except Exception as e:
            logger.info(
                f"WARN: Could not process square {square_nodes} for role assignment: {e}"
            )
            continue

    return squares_with_roles


def _calculate_pos_neg_epistasis_igraph(squares_with_roles):
    """Calculates positive/negative epistasis from squares with assigned roles."""
    if not squares_with_roles:
        return {"positive epistasis": 0.0, "negative epistasis": 0.0}

    data_for_df = []
    for square_role_info in squares_with_roles:
        fit_vals = square_role_info["fitness_values"]
        try:
            data_for_df.append(
                {
                    "ab": fit_vals[square_role_info["wild_type"]],
                    "aB": fit_vals[square_role_info["single_mutant_1"]],
                    "Ab": fit_vals[square_role_info["single_mutant_2"]],
                    "AB": fit_vals[square_role_info["double_mutant"]],
                }
            )
        except KeyError as e:
            continue  # skip squares with missing role data

    if not data_for_df:
        return {"positive epistasis": 0.0, "negative epistasis": 0.0}

    df_squares = pd.DataFrame(data_for_df)
    effect_mut1_b = df_squares["Ab"] - df_squares["ab"]
    effect_mut2_a = df_squares["aB"] - df_squares["ab"]
    effect_both = df_squares["AB"] - df_squares["ab"]

    positive_count = (effect_both > (effect_mut1_b + effect_mut2_a)).sum()
    total_squares = len(df_squares)

    perc_positive = positive_count / total_squares if total_squares > 0 else 0.0
    perc_negative = 1.0 - perc_positive

    return _pythonize({
        "positive epistasis": perc_positive,
        "negative epistasis": perc_negative,
    })


@dataclass(frozen=True)
class EpistasisClassification:
    """Proportions of the five epistasis types among 4-node motifs.

    ``magnitude`` + ``sign`` + ``reciprocal_sign`` partition the magnitude/sign
    group (they sum to 1 when any motifs are found); ``positive`` and
    ``negative`` are a separate decomposition of the same motifs.
    """

    magnitude: float
    sign: float
    reciprocal_sign: float
    positive: float
    negative: float


def classify_epistasis(landscape, approximate=False, sample_cut_prob=0.2, seed=None):
    """
    Calculates proportions of five epistasis types using 4-node motifs in an igraph graph.

    Determines magnitude, sign, and reciprocal sign epistasis based on counts/estimates
    of motifs 19, 52, 66. Determines positive and negative epistasis by analyzing
    the fitness relationships within instances of these motifs.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object, containing landscape.graph as an igraph.Graph
        with a "fitness" vertex attribute.
    approximate : bool, optional
        If True, estimates motif counts and uses a sample of motif instances
        for positive/negative epistasis calculation. Faster but less accurate.
        Defaults to False (exact counts and all relevant instances).
    sample_cut_prob : float, optional
        The probability used for pruning the search tree at each level during
        sampling when approximate=True. Higher values -> faster, less accurate.
        Defaults to 0.2.

    Returns
    -------
    EpistasisClassification
        A dataclass with proportion fields:
        - ``magnitude``: The magnitude of the combined fitness effect of mutations
        differs from the sum of their individual effects, but the direction relative to
        single mutants or wild-type may not change sign.
        - ``sign``: The sign of the fitness effect of at least one mutation changes depending
        on the presence of other mutations. For example, a mutation beneficial on its own becomes
        deleterious when combined with another specific mutation.
        - ``reciprocal_sign``: A specific form of sign epistasis where the sign of the effect
        of *each* mutation depends on the allele state at the other locus.
        - ``positive``: The combined fitness effect of mutations is greater than the sum of
        their individual effects, often referred to as synergistic epistasis.
        - ``negative``: The combined fitness effect of mutations is less than the sum of their
        individual effects, often referred to as antagonistic epistasis.

        Fields are zero if relevant counts/instances are zero or cannot be processed.

    Raises
    ------
    AttributeError
        If landscape.graph is not an igraph.Graph object or does not exist.
    ValueError
        If sample_cut_prob is not between 0 and 1, or if fitness attribute missing.
    """
    motif_size = 4
    square_indices = {19, 52, 66}  # set for O(1) membership in callback

    if not hasattr(landscape, "graph") or not isinstance(landscape.graph, ig.Graph):
        raise AttributeError(
            "Input 'landscape' must have a 'graph' attribute that is an igraph.Graph object."
        )
    if "fitness" not in landscape.graph.vs.attributes():
        raise ValueError("igraph.Graph must have a 'fitness' vertex attribute.")
    if approximate and not 0.0 <= sample_cut_prob <= 1.0:
        raise ValueError("sample_cut_prob must be between 0.0 and 1.0")

    collected_square_instances = defaultdict(list)
    cut_prob_vector = [sample_cut_prob] * motif_size if approximate else None

    if approximate:
        def motif_collector_callback_approx(graph, vertices, isoclass):
            if isoclass in square_indices:
                collected_square_instances[isoclass].append(tuple(sorted(vertices)))
            return False  # continue search

        # Seed the sampling RNG so approximate results are reproducible.
        with _seeded_igraph(seed):
            # Run 1: estimated counts for mag/sign/recip proportions
            estimated_motif_counts = landscape.graph.motifs_randesu(
                size=motif_size, cut_prob=cut_prob_vector
            )
            # Run 2: collect a sample of square instances
            landscape.graph.motifs_randesu(
                size=motif_size,
                cut_prob=cut_prob_vector,
                callback=motif_collector_callback_approx,
            )

        reci_sign_count = (
            np.nan_to_num(estimated_motif_counts[19])
            if len(estimated_motif_counts) > 19
            else 0
        )
        sign_count = (
            np.nan_to_num(estimated_motif_counts[52])
            if len(estimated_motif_counts) > 52
            else 0
        )
        mag_count = (
            np.nan_to_num(estimated_motif_counts[66])
            if len(estimated_motif_counts) > 66
            else 0
        )

    else:  # exact calculation
        def motif_collector_callback_exact(graph, vertices, isoclass):
            if isoclass in square_indices:
                collected_square_instances[isoclass].append(tuple(sorted(vertices)))
            return False  # continue search

        landscape.graph.motifs_randesu(
            size=motif_size, callback=motif_collector_callback_exact
        )

        reci_sign_count = len(collected_square_instances.get(19, []))
        sign_count = len(collected_square_instances.get(52, []))
        mag_count = len(collected_square_instances.get(66, []))

    # --- Step 2: Calculate Mag/Sign/Recip Proportions ---
    total_mag_sign_recip = reci_sign_count + sign_count + mag_count
    if total_mag_sign_recip == 0:
        mag_sign_recip_props = {
            "magnitude epistasis": 0.0,
            "sign epistasis": 0.0,
            "reciprocal sign epistasis": 0.0,
        }
    else:
        mag_sign_recip_props = {
            "magnitude epistasis": mag_count / total_mag_sign_recip,
            "sign epistasis": sign_count / total_mag_sign_recip,
            "reciprocal sign epistasis": reci_sign_count / total_mag_sign_recip,
        }

    # --- Step 4: Assign Roles within Collected Squares ---
    all_collected_squares = []
    for idx in square_indices:
        all_collected_squares.extend(collected_square_instances.get(idx, []))

    if not all_collected_squares:
        pos_neg_props = {"positive epistasis": 0.0, "negative epistasis": 0.0}
    else:
        squares_with_roles = _assign_roles_for_epistasis_igraph(
            landscape.graph, all_collected_squares
        )

        # --- Step 5: Calculate Positive/Negative Epistasis Proportions ---
        if not squares_with_roles:
            pos_neg_props = {"positive epistasis": 0.0, "negative epistasis": 0.0}
        else:
            pos_neg_props = _calculate_pos_neg_epistasis_igraph(squares_with_roles)

    # --- Step 6: Combine Results ---
    return EpistasisClassification(
        magnitude=float(mag_sign_recip_props["magnitude epistasis"]),
        sign=float(mag_sign_recip_props["sign epistasis"]),
        reciprocal_sign=float(mag_sign_recip_props["reciprocal sign epistasis"]),
        positive=float(pos_neg_props["positive epistasis"]),
        negative=float(pos_neg_props["negative epistasis"]),
    )


@dataclass(frozen=True)
class ExtradimensionalBypass:
    """Summary of extradimensional bypasses around reciprocal-sign-epistasis motifs.

    ``bypass_proportion`` is ``motifs_with_bypass / total_motifs``;
    ``average_bypass_length`` is NaN when no bypass exists.
    """

    bypass_proportion: float
    average_bypass_length: float
    total_motifs: int
    motifs_with_bypass: int


def extradimensional_bypass(landscape, approximate=False, sample_cut_prob=0.2, seed=None):
    """
    Analyzes extradimensional bypasses in reciprocal sign epistasis motifs.

    For each motif representing reciprocal sign epistasis (type 19), this function
    identifies whether accessible evolutionary paths exist that bypass the direct
    path between the double mutant nodes. Such indirect paths are called
    extradimensional bypasses and allow evolution to traverse fitness valleys
    that would otherwise be inaccessible under strong selection.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object, containing landscape.graph as an igraph.Graph
        with a "fitness" vertex attribute.
    approximate : bool, optional
        If True, uses sampling to find motif instances. Faster but less accurate.
        Defaults to False (exact enumeration of all instances).
    sample_cut_prob : float, optional
        The probability used for pruning the search tree at each level during
        sampling when approximate=True. Higher values -> faster, less accurate.
        Defaults to 0.2.

    Returns
    -------
    ExtradimensionalBypass
        A dataclass with attributes:

        - ``bypass_proportion`` : proportion of reciprocal-sign-epistasis motifs
          for which an extradimensional bypass exists (float between 0 and 1).
        - ``average_bypass_length`` : average length of extradimensional bypasses
          for motifs where such bypasses exist; NaN if none exist.
        - ``total_motifs`` : total number of type-19 motifs analyzed.
        - ``motifs_with_bypass`` : number of motifs that have a bypass.

    Raises
    ------
    AttributeError
        If landscape.graph is not an igraph.Graph object or does not exist.
    ValueError
        If sample_cut_prob is not between 0 and 1, or if fitness attribute missing.

    Notes
    -----
    Reciprocal sign epistasis occurs when both the wildtype (ab) and double mutant (AB)
    have higher fitness than both single mutants (aB, Ab). This creates a fitness valley
    that prevents direct evolutionary access between ab and AB. Extradimensional bypasses
    are indirect paths through the broader fitness landscape that circumvent this valley.
    """

    # --- Validate Input ---
    if not hasattr(landscape, "graph") or not isinstance(landscape.graph, ig.Graph):
        raise AttributeError(
            "Input 'landscape' must have a 'graph' attribute that is an igraph.Graph object."
        )
    if "fitness" not in landscape.graph.vs.attributes():
        raise ValueError("igraph.Graph must have a 'fitness' vertex attribute.")
    if approximate and not 0.0 <= sample_cut_prob <= 1.0:
        raise ValueError("sample_cut_prob must be between 0.0 and 1.0")

    # --- Find Type 19 Motifs (Reciprocal Sign Epistasis) ---
    try:
        motif_19_instances = _get_motif_node_indices(
            landscape.graph,
            motif_size=4,
            target_motif_type=19,
            approximate=approximate,
            sample_cut_prob=sample_cut_prob,
            seed=seed,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to find motif instances: {e}")

    if not motif_19_instances:
        return ExtradimensionalBypass(
            bypass_proportion=0.0,
            average_bypass_length=float("nan"),
            total_motifs=0,
            motifs_with_bypass=0,
        )

    # --- Analyze Each Motif for Extradimensional Bypasses ---
    total_motifs = len(motif_19_instances)

    # Identify (ab, AB) per motif from a precomputed fitness array, caching each
    # AB's predecessor set (one high-fitness node anchors many reciprocal-sign
    # squares). AB = fittest node (first on ties, matching the previous dict-max);
    # ab = first remaining node that is not a direct predecessor of AB.
    fit = np.asarray(landscape.graph.vs["fitness"], dtype=float)
    pred_cache = {}
    ABs_by_ab = defaultdict(list)
    for motif_nodes in motif_19_instances:
        nodes = list(motif_nodes)
        AB = nodes[int(np.argmax(fit[nodes]))]
        pred = pred_cache.get(AB)
        if pred is None:
            pred = pred_cache[AB] = set(landscape.graph.predecessors(AB))
        ab = next((n for n in nodes if n != AB and n not in pred), None)
        if ab is None:
            continue
        ABs_by_ab[ab].append(AB)

    # One forward traversal per UNIQUE ab -- the SAME improving (OUT) direction the
    # original used from the wildtype (whose descendant set is typically far smaller
    # than the double-mutant's ancestor set, so going from AB instead would be
    # slower), now covering all of that ab's motifs in a single BFS. igraph forbids
    # duplicate targets, so query distinct ABs once and count every motif occurrence
    # (approximate sampling can repeat a motif; the original processed the list
    # element-wise). A finite distance means an extradimensional bypass exists.
    bypass_lengths = []
    motifs_with_bypass = 0
    for ab, AB_list in ABs_by_ab.items():
        uniq_ABs = list(dict.fromkeys(AB_list))
        row = landscape.graph.distances(source=[ab], target=uniq_ABs, mode="out")[0]
        dist_map = dict(zip(uniq_ABs, row))
        for AB in AB_list:
            distance = dist_map[AB]
            if not np.isinf(distance):
                bypass_lengths.append(distance)
                motifs_with_bypass += 1

    # --- Calculate Results ---
    bypass_proportion = motifs_with_bypass / total_motifs if total_motifs > 0 else 0.0
    average_bypass_length = np.mean(bypass_lengths) if bypass_lengths else np.nan

    return ExtradimensionalBypass(
        bypass_proportion=float(bypass_proportion),
        average_bypass_length=float(average_bypass_length),
        total_motifs=int(total_motifs),
        motifs_with_bypass=int(motifs_with_bypass),
    )


def _get_motif_node_indices(
    graph, motif_size=4, target_motif_type=19, approximate=False, sample_cut_prob=0.2,
    seed=None,
):
    """
    Find all instances of a specific motif type and return their node indices.

    Parameters
    ----------
    graph : igraph.Graph
        The igraph object to search for motifs
    motif_size : int
        Size of motifs to search for (default 4)
    target_motif_type : int
        The specific motif ID to collect (e.g., 19, 52, 66)
    approximate : bool, optional
        If True, uses sampling to find motif instances. Faster but less accurate.
        Defaults to False (exact enumeration of all instances).
    sample_cut_prob : float, optional
        The probability used for pruning the search tree at each level during
        sampling when approximate=True. Higher values -> faster, less accurate.
        Defaults to 0.2.

    Returns
    -------
    list
        List of tuples, where each tuple contains the node indices
        for one instance of the target motif

    Raises
    ------
    ValueError
        If sample_cut_prob is not between 0 and 1.
    """
    if approximate and not 0.0 <= sample_cut_prob <= 1.0:
        raise ValueError("sample_cut_prob must be between 0.0 and 1.0")

    collected_motifs = []
    cut_prob_vector = [sample_cut_prob] * motif_size if approximate else None

    def motif_collector_callback(graph, vertices, isoclass):
        if isoclass == target_motif_type:
            collected_motifs.append(tuple(sorted(vertices)))
        return False  # continue search

    # Find motifs with or without sampling (seeded RNG only matters for the
    # approximate, sampling-based path).
    with _seeded_igraph(seed if approximate else None):
        if approximate:
            graph.motifs_randesu(
                size=motif_size, cut_prob=cut_prob_vector, callback=motif_collector_callback
            )
        else:
            graph.motifs_randesu(size=motif_size, callback=motif_collector_callback)

    return collected_motifs
