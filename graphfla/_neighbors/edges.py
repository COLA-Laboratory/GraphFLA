"""The public edge-construction entry point.

``build_edges`` resolves a neighbourhood strategy and dispatches to the
compute kernels, returning an :class:`EdgeResult`. This is the single
public API of the neighbours subpackage.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, Callable

import numpy as np

from ._arrays import _empty_edges, _empty_deltas
from . import _kernels
from .generators import (
    BooleanNeighborGenerator,
    SequenceNeighborGenerator,
    OrdinalNeighborGenerator,
    DefaultNeighborGenerator,
)
from ._kernels import _build_active, _build_pairwise, _build_broadcast
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeResult:
    """Container for the output of :func:`build_edges`.

    ``edges`` holds directed ``(source, target)`` node-index pairs and
    ``delta_fits`` the aligned ``|Δfitness|`` edge weights (``delta_fits[i]`` is
    the weight of edge ``i``). Both are kept in whichever container the chosen
    producer emits:

    * the ``active`` strategy returns an ``(E, 2)`` ``int64`` ndarray of edges
      and a 1-D ``float64`` ndarray of weights (no per-edge Python objects);
    * the ``pairwise`` / ``broadcast`` strategies return a Python list of
      ``(int, int)`` tuples and a list of floats.

    The empty case is always the canonical ``(0, 2)`` / ``(0,)`` ndarrays.
    igraph 0.11 ingests either container, so :meth:`Landscape._build_graph`
    consumes both without conversion. ``neutral_pairs`` is always a Python list
    of ``(int, int)`` tuples.
    """

    edges: Union[np.ndarray, List[Tuple[int, int]]]
    delta_fits: Union[np.ndarray, List[float]]
    neutral_pairs: List[Tuple[int, int]]


def build_edges(
    *,
    configs,
    config_dict,
    fitness,
    n_configs: int,
    n_vars: int,
    n_edit: int,
    strategy: str,
    epsilon: float,
    maximize: bool,
    verbose: bool,
    neighbor_generator: Callable,
    configs_array=None,
) -> EdgeResult:
    """Build improving edges and neutral pairs from a configuration dataset.

    This is the single public entry point for neighborhood construction.
    It validates inputs, resolves the ``'auto'`` strategy, and dispatches
    to the appropriate implementation.

    Parameters
    ----------
    configs : pandas.Series
        Mapping from node index to configuration tuple.
    config_dict : dict
        Encoding metadata keyed by variable index.
    fitness : numpy.ndarray
        Validated fitness values aligned with the configuration rows.
    n_configs, n_vars, n_edit : int
        Dataset dimensions and edit-distance threshold.
    strategy : str
        One of ``'auto'``, ``'active'``, ``'pairwise'``, ``'broadcast'``.
    epsilon : float
        Neutrality threshold.
    maximize : bool
        Optimization direction.
    verbose : bool
        Whether to print progress.
    neighbor_generator : callable
        Bound ``generate`` method of a :class:`NeighborGenerator`.
    configs_array : numpy.ndarray, optional
        Pre-computed numeric configuration matrix.

    Returns
    -------
    EdgeResult
    """
    # ``configs`` (tuple Series) may be None when deferred; ``configs_array`` is
    # then the source of truth and _active_generic derives tuples on demand.
    if (configs is None and configs_array is None) or config_dict is None:
        raise RuntimeError("Cannot build edges: configs/config_dict missing.")
    if n_configs is None:
        raise RuntimeError("n_configs not set before edge construction.")

    valid_strategies = {"auto", "active", "pairwise", "broadcast"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {sorted(valid_strategies)}."
        )

    resolved = strategy
    if resolved == "auto":
        resolved = _select_strategy(
            n_configs, n_vars, config_dict, n_edit, neighbor_generator
        )
        if verbose:
            logger.info(f" - Auto-selected '{resolved}' neighborhood strategy.")

    kwargs = dict(
        configs=configs,
        config_dict=config_dict,
        fitness=fitness,
        n_edit=n_edit,
        epsilon=epsilon,
        maximize=maximize,
        verbose=verbose,
        neighbor_generator=neighbor_generator,
        configs_array=configs_array,
    )

    if resolved == "pairwise":
        edges, delta_fits, neutral_pairs = _build_pairwise(**kwargs)
    elif resolved == "broadcast":
        edges, delta_fits, neutral_pairs = _build_broadcast(**kwargs)
    else:
        edges, delta_fits, neutral_pairs = _build_active(**kwargs)

    edges, delta_fits = _normalize_edge_output(edges, delta_fits)

    return EdgeResult(
        edges=edges,
        delta_fits=delta_fits,
        neutral_pairs=neutral_pairs,
    )


def _normalize_edge_output(edges, delta_fits):
    """Canonicalise the empty case while preserving the producer's container.

    The ``active`` producer returns ``(E, 2)`` int64 / 1-D float64 ndarrays;
    the ``pairwise``/``broadcast`` producers return Python lists of
    ``(source, target)`` tuples and floats. Both forms are accepted directly by
    igraph 0.11's edge-list / edge-attr ingestion, so they are passed through
    unchanged here. Converting the list producers to arrays is deliberately
    avoided: ``np.asarray`` over a large list of tuples is costly and igraph
    ingests a Python tuple list faster than an ndarray, so forcing arrays would
    regress those (small-cardinality, dense-pairwise) datasets.

    Only the empty case is normalised, to the canonical ``(0, 2)`` / ``(0,)``
    ndarray shapes, so :meth:`Landscape._build_graph` has one empty sentinel.
    """
    edges_out = edges if len(edges) else _empty_edges()
    delta_out = delta_fits if len(delta_fits) else _empty_deltas()
    return edges_out, delta_out


# ===================================================================
# Strategy selection (private)
# ===================================================================


def _select_strategy(n_configs, n_vars, config_dict, n_edit, neighbor_generator):
    """Choose a supported backend using the generator and encoded key space."""
    generator = getattr(neighbor_generator, "__self__", None)
    generator_type = type(generator)
    builtins = (
        BooleanNeighborGenerator,
        SequenceNeighborGenerator,
        OrdinalNeighborGenerator,
        DefaultNeighborGenerator,
    )
    if generator_type not in builtins:
        return "active"
    if n_edit > 1:
        return "pairwise"
    if generator_type is OrdinalNeighborGenerator:
        return "active"
    if generator_type is DefaultNeighborGenerator:
        has_ordinal = any(info["type"] == "ordinal" for info in config_dict.values())
        return "active" if has_ordinal else "pairwise"

    base = 2 if generator_type is BooleanNeighborGenerator else generator.alphabet_size
    if base > 256:
        return "pairwise"
    if n_vars >= 64 or base**n_vars > (1 << 63):
        return "active"  # The active overflow path uses exact masked grouping.
    affordable = min(_kernels._LUT_MAX_CELLS, max(1 << 20, 8 * n_configs))
    return "active" if base**n_vars <= affordable else "pairwise"
