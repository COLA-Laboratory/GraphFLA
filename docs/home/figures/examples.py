"""Authored workflow illustrations, using the original smooth three-peak surface."""
import math
import random

from .camera import Camera
from .palette import Palette
from .scenes import Figure
from .svg import disc, path_data
from .surfaces import workflow
from .terrain import terrain

SCENARIOS = ("protein", "chemistry", "materials", "software", "hpo")


def draw(key, tokens):
    palette = Palette.from_tokens("card", tokens)
    figure = Figure("example-" + key, 560, 520)
    variation = SCENARIOS.index(key)
    shift = ((0, 0), (0.035, -0.018), (-0.025, 0.015), (0.015, 0.025), (-0.018, -0.025))[variation]
    height = lambda u, v: workflow(u + shift[0], v + shift[1])
    upper = Camera(280, 205, 365, 38, 26, 178)
    lower = Camera(280, 388, 365, 38, 26, 178)

    # The original layered composition, with a sparse, gently irregular mesh below.
    # Adjacent rows are linked without crossings; no data-derived force layout.
    rng = random.Random(31)
    rows, points = [], []
    for row, count in enumerate((4, 5, 6, 7, 6, 4)):
        indices = []
        span = 0.14 * (count - 1)
        for col in range(count):
            u = 0.5 - span / 2 + col * 0.14 + rng.uniform(-0.012, 0.012)
            v = 0.14 + row * 0.144 + rng.uniform(-0.01, 0.01)
            indices.append(len(points))
            points.append((u, v))
        rows.append(indices)
    links = set()
    for row in rows:
        links.update(zip(row, row[1:]))
    for first, second in zip(rows, rows[1:]):
        for a in first:
            b = min(second, key=lambda b: abs(points[a][0] - points[b][0]))
            links.add((a, b))
        for b in second:
            a = min(first, key=lambda a: abs(points[a][0] - points[b][0]))
            links.add((a, b))

    values = [height(*p) for p in points]
    best = max(range(len(points)), key=lambda i: values[i])
    # Place the highlighted configuration exactly at the main summit.
    points[best] = (0.36 - shift[0], 0.40 - shift[1])
    values[best] = height(*points[best])
    floor = [lower.project(*p) for p in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))]
    figure.body.append(f'<path d="{path_data(floor)}" fill="none" stroke="{palette.tone(0.28)}" stroke-width="0.7"/>')
    # Light projection guides preserve the graph-to-surface reading of the original.
    for i in (1, 6, best, 20, 28):
        a = lower.project(*points[i])
        b = upper.project(*points[i], height(*points[i]))
        figure.body.append(f'<path d="{path_data([a, b])}" fill="none" stroke="{palette.tone(0.3)}" '
                           'stroke-width="0.8" stroke-dasharray="3 5"/>')
    for a, b in sorted(links):
        x1, y1 = lower.project(*points[a])
        x2, y2 = lower.project(*points[b])
        bend = 2.4 * (1 if (a + b) % 2 else -1)
        length = math.hypot(x2 - x1, y2 - y1)
        cx = (x1 + x2) / 2 - (y2 - y1) / length * bend
        cy = (y1 + y2) / 2 + (x2 - x1) / length * bend
        figure.body.append(f'<path d="M{x1:.2f} {y1:.2f}Q{cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" '
                           f'fill="none" stroke="{palette.tone(0.56)}" stroke-width="1.1"/>')
    for i, p in enumerate(points):
        color = palette.accent if i == best else palette.tone(0.40 + 0.5 * values[i] / max(values))
        figure.body.append(disc(lower.project(*p), 6.2 if i == best else 5, color, palette.surface, 1.2))
    figure.body += terrain(upper, height, palette, step=0.04, every=4, density=0.8)
    for i in (1, 6, best, 20, 28):
        u, v = points[i]
        if upper.visible(height, u, v):
            figure.body.append(disc(upper.project(u, v, height(u, v) + .008),
                                   5 if i == best else 2.8,
                                   palette.accent if i == best else palette.mark, palette.surface, 1.2))
    figure.label("surface", (24, 20), "title")
    figure.label("graph", (344, 467), "title")
    return figure


def render_examples(out, tokens):
    metadata = {}
    for key in SCENARIOS:
        fig = draw(key, tokens)
        (out / (fig.name + ".svg")).write_text(fig.svg())
        metadata[fig.name] = fig.metadata()
    return metadata
