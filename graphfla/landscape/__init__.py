# graphfla/landscape/__init__.py
"""
Classes for landscape representations.
"""

from .landscape import Landscape
from .boolean import BooleanLandscape
from .sequence import (
    SequenceLandscape,
    DNALandscape,
    RNALandscape,
    ProteinLandscape,
)

__all__ = [
    "Landscape",
    "BooleanLandscape",
    "SequenceLandscape",
    "DNALandscape",
    "RNALandscape",
    "ProteinLandscape",
]
