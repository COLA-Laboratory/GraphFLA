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

__version__ = "0.1.dev0"

logger = logging.getLogger(__name__)

_exported_core_objects = ["Landscape"]


# Lazily loaded submodules (e.g. graphfla.analysis, graphfla.utils)
_submodules = [
    "analysis",
    "algorithms",
    "distances",
    "landscape",
    "lon",
    "plotting",
    "problems",
    "sampling",
    "filters",
    "utils",
]

__all__ = _submodules + _exported_core_objects


def __dir__():
    """Provides controlled module listing for autocompletion."""
    return __all__


def __getattr__(name):
    """Lazily import submodules and the top-level ``Landscape`` on first access.

    Example
    -------
        >>> import graphfla
        >>> graphfla.analysis        # the analysis submodule is imported here
        >>> graphfla.Landscape       # the core class, imported lazily
    """
    if name in _submodules:
        return importlib.import_module(f".{name}", __name__)
    if name == "Landscape":
        from .landscape import Landscape

        return Landscape
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
