"""Epistasis analysis for fitness landscapes.

Split across motifs / gamma / idiosyncrasy / higher_order / walsh_hadamard
submodules; the public functions are re-exported here.
"""

from .motifs import (
    classify_epistasis,
    EpistasisClassification,
    extradimensional_bypass,
    ExtradimensionalBypass,
)
from .gamma import gamma, gamma_star
from .idiosyncrasy import (
    idiosyncratic_index,
    global_idiosyncratic_index,
    diminishing_returns_index,
    increasing_costs_index,
)
from .higher_order import higher_order_epistasis
from .walsh_hadamard import walsh_hadamard

__all__ = [
    "classify_epistasis",
    "EpistasisClassification",
    "extradimensional_bypass",
    "ExtradimensionalBypass",
    "gamma",
    "gamma_star",
    "idiosyncratic_index",
    "global_idiosyncratic_index",
    "diminishing_returns_index",
    "increasing_costs_index",
    "higher_order_epistasis",
    "walsh_hadamard",
]
