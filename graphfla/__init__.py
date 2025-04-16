# graphfla/__init__.py

"""
graphfla: A Python package for Graph-based Fitness Landscape Analysis.
========================================================

graphfla provides tools for generating, analyzing, simulating evolution on,
and visualizing fitness landscapes, commonly encountered in evolutionary
computation, biology, optimization, and machine learning model training dynamics.

It aims to offer a modular and user-friendly interface for researchers and
practitioners working with sequence spaces, combinatorial spaces, and
their associated fitness functions.
"""

# Authors: [Mingyu Huang, COLALab@UoE]

import importlib
import logging
import os
import random

__version__ = "0.1.dev0"

logger = logging.getLogger(__name__)

_exported_config_functions = []

_exported_core_objects = ["Landscape"]


# List of submodules and top-level utility modules to be accessible
# via lazy loading (e.g., graphfla.analysis, graphfla.utils)
_submodules = [
    "analysis",
    "algorithms",
    "distances",
    "landscape",
    "lon",
    "plotting",
    "problems",
    "sampling",
    "filters" "utils",
]

__all__ = _submodules + _exported_config_functions + _exported_core_objects


def __dir__():
    """Provides controlled module listing for autocompletion."""
    return __all__


def __getattr__(name):
    """
    Lazily imports submodules and top-level modules upon first access.

    Example:
        >>> import graphfla
        >>> graphfla.analysis.fdc # analysis submodule is imported here
    """
    if name in _submodules:
        return importlib.import_module(f".{name}", __name__)
    elif name in _exported_core_objects or name in _exported_config_functions:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs."""
    import numpy as np

    _random_seed = os.environ.get("GRAPHFLA_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)

    logger.info("I: Seeding RNGs with %r", _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
