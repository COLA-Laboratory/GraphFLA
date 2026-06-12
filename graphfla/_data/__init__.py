"""Data preparation and encoding for fitness landscape construction.

Input handlers, the filter/prepare/clean/encode pipeline, and the
PreparedData container. Split across _validation / handlers / pipeline
submodules; the public surface is re-exported here.
"""

from ._validation import (
    ALLOWED_DATA_TYPES,
    DNA_ALPHABET,
    RNA_ALPHABET,
    PROTEIN_ALPHABET,
    PreparedData,
    configs_series_from_array,
)
from .handlers import (
    InputHandler,
    BooleanHandler,
    OrdinalHandler,
    SequenceHandler,
    DefaultHandler,
)
from .pipeline import filter_data, prepare_data, clean_data, encode_data

__all__ = [
    "ALLOWED_DATA_TYPES",
    "DNA_ALPHABET",
    "RNA_ALPHABET",
    "PROTEIN_ALPHABET",
    "PreparedData",
    "configs_series_from_array",
    "InputHandler",
    "BooleanHandler",
    "OrdinalHandler",
    "SequenceHandler",
    "DefaultHandler",
    "filter_data",
    "prepare_data",
    "clean_data",
    "encode_data",
]
