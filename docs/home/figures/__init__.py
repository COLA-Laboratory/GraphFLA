"""Landscape illustrations of the GraphFLA landing page, drawn as faceted SVG surfaces."""
from .palette import Palette, load_tokens
from .scenes import SCENES, render

__all__ = ["SCENES", "Palette", "load_tokens", "render"]
