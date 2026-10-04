"""Benchmark line charts as inline SVG.

The marks carry only class names; their colours come from ``components.css``.
"""
import math
from dataclasses import dataclass
from html import escape

WIDTH, HEIGHT = 400, 290
LEFT, RIGHT, TOP, BOTTOM = 54, 372, 30, 240
COLUMN_INSET = 38
MERGE_DISTANCE = 6


@dataclass(frozen=True)
class Series:
    """One line of the chart.

    Parameters
    ----------
    name : str
        Name used in the text description.
    values : sequence of float
        One value per x position.
    emphasis : bool, default=False
        Draw in the accent colour, with labels below the points instead of above.
    """

    name: str
    values: tuple
    emphasis: bool = False


def format_duration(seconds):
    """Format seconds as ``3.3 ms``, ``303 ms`` or ``5.18 s``."""
    if seconds >= 1:
        return f"{seconds:.2f} s"
    milliseconds = seconds * 1000
    return f"{milliseconds:.1f} ms" if milliseconds < 10 else f"{milliseconds:.0f} ms"


def line_chart(columns, series, ticks, format_value, *, log=False, x_title, title, both="both"):
    """Return an SVG line chart with every point labelled.

    Parameters
    ----------
    columns : sequence of str
        Labels of the x positions, evenly spaced.
    series : sequence of Series
        Lines in paint order; the emphasised one should come last.
    ticks : sequence of (float, str)
        Value and label of each horizontal grid line. The first and last set the y range.
    format_value : callable
        Maps a value to its point label.
    log : bool, default=False
        Use a logarithmic y axis.
    x_title : str
        Title under the x axis.
    title : str
        Opening of the text description for screen readers.
    both : str, default="both"
        Prefix of the single label used where two points coincide.
    """
    scale = math.log10 if log else (lambda value: value)
    low, high = scale(ticks[0][0]), scale(ticks[-1][0])
    y_of = lambda value: BOTTOM - (scale(value) - low) / (high - low) * (BOTTOM - TOP)
    span = RIGHT - LEFT - 2 * COLUMN_INSET
    x_of = lambda k: LEFT + COLUMN_INSET + span * k / (len(columns) - 1)

    description = escape(_describe(title, columns, series, format_value))
    out = [f'<svg class="gfl-chart" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{description}">']
    for value, label in ticks:
        y = y_of(value)
        kind = "gfl-chart__axis" if value == ticks[0][0] else "gfl-chart__grid"
        out.append(f'<path class="{kind}" d="M{LEFT} {y:.1f}H{RIGHT}"/>')
        out.append(f'<text class="gfl-chart__tick" x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{escape(label)}</text>')
    for k, column in enumerate(columns):
        out.append(f'<text class="gfl-chart__tick" x="{x_of(k):.1f}" y="{BOTTOM + 20}" text-anchor="middle">{escape(column)}</text>')
    out.append(f'<text class="gfl-chart__tick" x="{(LEFT + RIGHT) / 2:.1f}" y="{BOTTOM + 41}" text-anchor="middle">{escape(x_title)}</text>')

    for item in series:
        points = " L".join(f"{x_of(k):.1f} {y_of(value):.1f}" for k, value in enumerate(item.values))
        out.append(f'<path class="{_classes("gfl-chart__line", item)}" d="M{points}"/>')
    labels = []
    for k in range(len(columns)):
        ys = [y_of(item.values[k]) for item in series]
        texts = [format_value(item.values[k]) for item in series]
        # Points that overlap and read the same share one marker and one label.
        merged = len(series) == 2 and texts[0] == texts[1] and abs(ys[0] - ys[1]) < MERGE_DISTANCE
        strong = " gfl-chart__label--strong" if k == len(columns) - 1 else ""
        for item, y, text in zip(series, ys, texts):
            if merged and not item.emphasis:
                continue
            out.append(f'<circle class="{_classes("gfl-chart__point", item)}" cx="{x_of(k):.1f}" cy="{y:.1f}" r="4.5"/>')
            above = merged or not item.emphasis
            labels.append(f'<text class="gfl-chart__label{strong}" x="{x_of(k):.1f}" y="{y - 13 if above else y + 19:.1f}" '
                          f'text-anchor="middle">{escape(f"{both} {text}" if merged else text)}</text>')
    return "\n".join(out + labels + ["</svg>"])


def _classes(base, item):
    return f"{base} {base}--accent" if item.emphasis else base


def _describe(title, columns, series, format_value):
    parts = [f"{item.name}: " + ", ".join(f"{format_value(value)} at {column}"
                                          for column, value in zip(columns, item.values)) for item in series]
    return f"{title}. " + ". ".join(parts) + "."
