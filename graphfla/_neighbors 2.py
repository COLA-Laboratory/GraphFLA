from typing import Protocol, Tuple, Dict, List, Any, runtime_checkable
import warnings


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
            The configuration for which to find neighbors
        config_dict : dict
            Dictionary describing the encoding
        n_edit : int
            Edit distance for neighborhood definition

        Returns
        -------
        list[tuple]
            List of neighboring configurations
        """
        ...


class BooleanNeighborGenerator:
    """Generator for boolean neighbors (bit flips)."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors by flipping bits."""
        if n_edit != 1:
            warnings.warn(
                f"BooleanNeighborGenerator only supports n_edit=1 for single bit flips. "
                f"Received n_edit={n_edit}. Returning no neighbors.",
                UserWarning,
            )
            return []

        neighbors = []
        current_config_list = list(config)
        num_bits = len(current_config_list)

        for i in range(num_bits):
            neighbor_list = current_config_list.copy()
            neighbor_list[i] = 1 - neighbor_list[i]  # Flip bit
            neighbors.append(tuple(neighbor_list))

        return neighbors


class SequenceNeighborGenerator:
    """Generator for sequence neighbors (substitutions)."""

    def __init__(self, alphabet_size: int):
        """
        Initialize with the size of the alphabet.

        Parameters
        ----------
        alphabet_size : int
            Number of possible values at each position
        """
        self.alphabet_size = alphabet_size

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors by substituting at each position."""
        if n_edit != 1:
            warnings.warn(
                f"SequenceNeighborGenerator only supports n_edit=1 for single position substitutions. "
                f"Received n_edit={n_edit}. Returning no neighbors.",
                UserWarning,
            )
            return []

        neighbors = []
        current_config_list = list(config)
        num_positions = len(current_config_list)

        for i in range(num_positions):
            original_val = current_config_list[i]
            # Try each possible substitution at this position
            for new_val in range(self.alphabet_size):
                if new_val != original_val:
                    neighbor_list = current_config_list.copy()
                    neighbor_list[i] = new_val
                    neighbors.append(tuple(neighbor_list))

        return neighbors


class DefaultNeighborGenerator:
    """Default generator for mixed data types."""

    def generate(
        self, config: Tuple, config_dict: Dict, n_edit: int = 1
    ) -> List[Tuple]:
        """Generate neighbors based on data types in config_dict."""
        if n_edit != 1:
            warnings.warn(
                f"DefaultNeighborGenerator only fully supports n_edit=1. "
                f"Received n_edit={n_edit}.",
                UserWarning,
            )

        neighbors = []
        num_vars = len(config)

        for i in range(num_vars):
            info = config_dict[i]
            current_val = config[i]
            dtype = info["type"]

            if dtype == "boolean":
                # Flip the bit (0 to 1, 1 to 0)
                new_vals = [1 - current_val]
            elif dtype in ["categorical", "ordinal"]:
                # Iterate through all possible values
                max_val = info["max"]
                new_vals = [v for v in range(max_val + 1) if v != current_val]
            else:
                warnings.warn(
                    f"Unsupported dtype '{dtype}' in generate_neighbors, skipping var {i}",
                    RuntimeWarning,
                )
                continue

            # Create neighbor tuples
            for new_val in new_vals:
                neighbor_list = list(config)
                neighbor_list[i] = new_val
                neighbors.append(tuple(neighbor_list))

        return neighbors
