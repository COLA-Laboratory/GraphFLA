"""Neighbor-generation strategy classes.

Each generator enumerates the adjacent configurations of a genotype in a
given encoding space (boolean, sequence, ordinal, or mixed/default). The
edge-construction layer receives a bound ``generate`` method.
"""

import warnings
from typing import Protocol, Tuple, Dict, List, runtime_checkable

import numpy as np

from ._arrays import _neutral_abs_threshold


@runtime_checkable
class NeighborGenerator(Protocol):
    """Protocol defining the interface for neighbor generation."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """
        Generate neighbors for a given configuration.

        Parameters
        ----------
        config : tuple
            The configuration for which to find neighbors.
        config_dict : dict
            Dictionary describing the encoding.
        n_edit : int
            Edit distance for neighborhood definition.

        Returns
        -------
        list[tuple]
            List of neighboring configurations.
        """
        ...


class BooleanNeighborGenerator:
    """Neighbor generator for boolean spaces (single bit flips)."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors by flipping bits."""
        if n_edit != 1:
            raise ValueError(
                f"BooleanNeighborGenerator only supports n_edit=1 "
                f"(single-bit flips). Received n_edit={n_edit}. "
                f"Use neighborhood_strategy='pairwise' or 'broadcast' if you need "
                f"Hamming neighborhoods with n_edit>1."
            )

        return [
            config[:i] + (1 - config[i],) + config[i + 1 :]
            for i in range(len(config))
        ]


class SequenceNeighborGenerator:
    """Neighbor generator for discrete-alphabet sequences (substitutions)."""

    def __init__(self, alphabet_size: int):
        """
        Initialize with the size of the alphabet.

        Parameters
        ----------
        alphabet_size : int
            Number of possible values at each position.
        """
        self.alphabet_size = alphabet_size

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors by substituting at each position."""
        if n_edit != 1:
            raise ValueError(
                f"SequenceNeighborGenerator only supports n_edit=1 "
                f"(single-position substitutions). Received n_edit={n_edit}. "
                f"Use neighborhood_strategy='pairwise' or 'broadcast' for "
                f"Hamming neighborhoods with n_edit>1."
            )

        neighbors = []
        for i, original in enumerate(config):
            prefix, suffix = config[:i], config[i + 1 :]
            for val in range(self.alphabet_size):
                if val != original:
                    neighbors.append(prefix + (val,) + suffix)
        return neighbors


class OrdinalNeighborGenerator:
    """Neighbor generator for ordinal spaces (±1 step on the ordinal scale).

    For an ordinal variable with allowed encoded values ``0..max``, each
    configuration has at most two neighbors per position: one ``+1`` step
    (if ``current < max``) and one ``-1`` step (if ``current > 0``).
    This corresponds to a Manhattan-distance-1 neighborhood on each axis,
    which is the standard definition of an ordinal-landscape neighborhood
    in the literature.
    """

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate ±1-step neighbors at each ordinal position."""
        if n_edit != 1:
            raise ValueError(
                f"OrdinalNeighborGenerator only supports n_edit=1 "
                f"(single ±1 step on the ordinal scale). Received n_edit={n_edit}. "
                f"Use neighborhood_strategy='pairwise' or 'broadcast' for "
                f"larger Hamming-style neighborhoods (note: those use Hamming, "
                f"not Manhattan, distance)."
            )

        neighbors = []
        for i in range(len(config)):
            info = config_dict[i]
            current = int(config[i])
            max_val = int(info["max"])
            for delta in (-1, 1):
                new_val = current + delta
                if 0 <= new_val <= max_val:
                    neighbor = list(config)
                    neighbor[i] = new_val
                    neighbors.append(tuple(neighbor))
        return neighbors


class DefaultNeighborGenerator:
    """Neighbor generator for mixed data types (boolean / categorical / ordinal)."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors based on data types in config_dict."""
        if n_edit != 1:
            raise ValueError(
                f"DefaultNeighborGenerator only supports n_edit=1. "
                f"Received n_edit={n_edit}. "
                f"Use neighborhood_strategy='pairwise' or 'broadcast' for "
                f"Hamming neighborhoods with n_edit>1."
            )

        neighbors = []
        for i in range(len(config)):
            info = config_dict[i]
            current = config[i]
            dtype = info["type"]

            if dtype == "boolean":
                new_vals = [1 - current]
            elif dtype == "categorical":
                new_vals = [v for v in range(info["max"] + 1) if v != current]
            elif dtype == "ordinal":
                # ±1 step on the ordinal scale, as in OrdinalNeighborGenerator.
                cur = int(current)
                max_val = int(info["max"])
                new_vals = [v for v in (cur - 1, cur + 1) if 0 <= v <= max_val]
            else:
                warnings.warn(
                    f"Unsupported dtype '{dtype}', skipping variable {i}.",
                    RuntimeWarning,
                )
                continue

            for val in new_vals:
                neighbor = list(config)
                neighbor[i] = val
                neighbors.append(tuple(neighbor))
        return neighbors


# ===================================================================
# Edge construction — public API
# ===================================================================
