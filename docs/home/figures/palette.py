"""Figure colours, read from the site's design tokens."""
import re
from dataclasses import dataclass
from pathlib import Path

TOKENS = Path(__file__).resolve().parents[1] / "styles" / "tokens.css"

_COMMENT = re.compile(r"/\*(.*?)\*/", re.DOTALL)
_DECLARATION = re.compile(r"(--[\w-]+)\s*:\s*([^;]+);")
_REFERENCE = re.compile(r"var\((--[\w-]+)\)")


def token_groups(path=TOKENS):
    """Return ``[(group title, [(name, value), ...]), ...]`` as written in the tokens file.

    A group starts at each comment inside the ``:root`` block.
    """
    body = Path(path).read_text().split(":root", 1)[1]
    groups = []
    pieces = _COMMENT.split(body)
    for title, block in zip(pieces[1::2], pieces[2::2]):
        declarations = [(name, value.strip()) for name, value in _DECLARATION.findall(block)]
        if declarations:
            groups.append((" ".join(title.split()), declarations))
    return groups


def load_tokens(path=TOKENS):
    """Return ``{name: value}`` for every token, with ``var()`` references resolved."""
    tokens = {name: value for _, group in token_groups(path) for name, value in group}

    def resolve(value):
        while (match := _REFERENCE.search(value)):
            value = value.replace(match.group(0), tokens[match.group(1)])
        return value

    return {name: resolve(value) for name, value in tokens.items()}


def _rgb(color):
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError(f"figure colours must be six-digit hex, got {color!r}")
    return tuple(int(color[k:k + 2], 16) for k in (1, 3, 5))


@dataclass(frozen=True)
class Palette:
    """Colours of one figure.

    Parameters
    ----------
    surface : str
        Colour the figure sits on. Casings, rings and masks are painted in it.
    line : str
        Colour of terrain lines at full strength.
    mark : str
        Colour of markers and walks that are not accented.
    accent : str
        Colour of the highlighted marker or walk.
    muted : str
        Colour of axes.
    """

    surface: str
    line: str
    mark: str
    accent: str
    muted: str

    @classmethod
    def from_tokens(cls, surface, tokens=None):
        """Build the palette for a figure placed on the ``surface`` token (``page`` or ``card``)."""
        tokens = load_tokens() if tokens is None else tokens
        return cls(
            surface=tokens[f"--gfl-color-{surface}"],
            line=tokens["--gfl-color-data"],
            mark=tokens["--gfl-color-mark"],
            accent=tokens["--gfl-color-accent"],
            muted=tokens["--gfl-color-text-muted"],
        )

    def tone(self, strength):
        """Line colour at ``strength`` in [0, 1]: 0 is the surface, 1 the full line colour."""
        strength = max(0.0, min(1.0, strength))
        surface, line = _rgb(self.surface), _rgb(self.line)
        mixed = (surface[k] + (line[k] - surface[k]) * strength for k in range(3))
        return "#" + "".join(f"{max(0, min(255, round(c))):02x}" for c in mixed)
