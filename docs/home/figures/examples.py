"""Workflow illustrations: each scenario's neighbor graph and the landscape above it.

Every scenario has the graph of its own kind of search space, laid out by stress
majorization, and its own surface. Node fitness is the surface height at the node,
so the local optima and fitness-distance correlation of the drawn graph are
computed from the drawing, not asserted.
"""
import math
import random

from .camera import Camera
from .graphs import distances, fit_disc, product, simplex, stress_layout
from .palette import Palette
from .scenes import Figure
from .svg import disc, path_data
from .terrain import terrain

BINARY, CATEGORY = (2, "categorical"), "categorical"

# Search space, layout seed, and (amplitude, width) of each surface peak, highest first.
SCENARIOS = {
    "protein": (product(*[BINARY] * 5), 3, [(0.80, 0.075), (0.52, 0.068), (0.42, 0.064), (0.34, 0.06)]),
    "chemistry": (product((4, CATEGORY), (3, CATEGORY), (3, CATEGORY)), 5, [(0.80, 0.11), (0.58, 0.1), (0.40, 0.09)]),
    "materials": (simplex(6), 1, [(0.78, 0.2), (0.34, 0.1)]),
    "software": (product(*[BINARY] * 4, (3, "ordinal")), 7, [(0.80, 0.1), (0.66, 0.1), (0.36, 0.08)]),
    "hpo": (product((4, "ordinal"), (3, "ordinal"), (3, CATEGORY)), 11, [(0.76, 0.16), (0.40, 0.09)]),
}
# Where the main summit sits: left of centre and toward the back, so the surface does not hide it.
SUMMIT = (0.40, 0.40)
PEAK_HEIGHT = 0.72
WIDTH, HEIGHT = 560, 520
UPPER = dict(cx=280, cy=205, scale=365, azimuth=38, tilt=26, zscale=178)
LOWER = dict(UPPER, cy=388)
# Each edge bends to a random side by up to this fraction of its length, so the arcs
# weave in every direction instead of all bowing the same way.
BEND = (0.06, 0.26)


def _peaks(points, hops, widths):
    """Return one node per hill width, each as far as possible from the hills already placed.

    Hills are two or more steps apart in the graph and far enough apart on the floor to
    stay distinct. The first is the node nearest ``SUMMIT``.
    """
    chosen = [min(range(len(points)), key=lambda i: math.dist(points[i], SUMMIT))]
    for width in widths[1:]:
        allowed = [i for i in range(len(points)) if math.dist(points[i], (0.5, 0.5)) < 0.34
                   and all(hops[i][c] >= 2 and math.dist(points[i], points[c]) > 1.2 * (width + widths[k])
                           for k, c in enumerate(chosen))]
        if not allowed:
            break
        chosen.append(max(allowed, key=lambda i: min(math.dist(points[i], points[c]) for c in chosen)))
    return chosen


def _surface(centres, peaks):
    """Return a height function with one hill above each peak node, scaled to ``PEAK_HEIGHT``."""
    def raw(u, v):
        z = 0.12 * math.exp(-((u - 0.5) ** 2 + (v - 0.5) ** 2) / (2 * 0.3 ** 2))
        for (cu, cv), (amplitude, width) in zip(centres, peaks):
            z += amplitude * math.exp(-((u - cu) ** 2 + (v - cv) ** 2) / (2 * width ** 2))
        return z
    # Every scenario reaches the same summit height, so the figures share one frame.
    top = max(raw(*c) for c in centres)
    return lambda u, v: PEAK_HEIGHT * raw(u, v) / top


def _ranks(values):
    """Return average ranks, with ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for k in range(start, end + 1):
            ranks[order[k]] = (start + end) / 2
        start = end + 1
    return ranks


def spearman(x, y):
    """Return the Spearman rank correlation of two equally long sequences."""
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def _arc(camera, a, b, bend):
    """Return the ``d`` of a quadratic arc from ``a`` to ``b`` on the floor plane, and its length.

    ``bend`` is the signed offset of the control point, as a fraction of the edge length.
    """
    (u1, v1), (u2, v2) = a, b
    control = ((u1 + u2) / 2 - (v2 - v1) * bend, (v1 + v2) / 2 + (u2 - u1) * bend)
    (x1, y1), (cx, cy), (x2, y2) = (camera.project(*p) for p in (a, control, b))
    return f"M{x1:.2f} {y1:.2f}Q{cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}", math.hypot(u2 - u1, v2 - v1)


def draw(key, tokens):
    """Return the figure of one scenario and the statistics of its drawn graph."""
    palette = Palette.from_tokens("card", tokens)
    figure = Figure("example-" + key, WIDTH, HEIGHT)
    (nodes, edges), seed, peaks = SCENARIOS[key]
    points = fit_disc(stress_layout(len(nodes), edges, seed=seed), radius=0.4)
    hops = distances(len(points), edges)
    centres = _peaks(points, hops, [width for _, width in peaks])
    peaks = peaks[:len(centres)]
    height = _surface([points[c] for c in centres], peaks)
    upper, lower = Camera(**UPPER), Camera(**LOWER)

    # Fitness decays with graph distance to the strongest nearby peak, so exactly the
    # peak nodes are local optima; the surface puts a hill directly above each of them.
    fitness = [max(amplitude * math.exp(-hops[i][c]) for c, (amplitude, _) in zip(centres, peaks))
               for i in range(len(points))]
    neighbors = [set() for _ in points]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    optima = [i for i in range(len(points)) if all(fitness[i] > fitness[j] for j in neighbors[i])]
    assert sorted(optima) == sorted(centres), key
    best = centres[0]
    low, top = min(fitness), fitness[best]
    to_best = hops[best]
    others = [i for i in range(len(points)) if i != best]
    fdc = spearman([fitness[i] for i in others], [to_best[i] for i in others])

    floor = [lower.project(*p) for p in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))]
    figure.body.append(f'<path d="{path_data(floor)}" fill="none" stroke="{palette.tone(0.28)}" stroke-width="0.7"/>')
    # Projection guides tie each local optimum in the graph to its point on the surface.
    for i in optima:
        guide = [lower.project(*points[i]), upper.project(*points[i], height(*points[i]))]
        figure.body.append(f'<path d="{path_data(guide)}" fill="none" stroke="{palette.tone(0.3)}" '
                           'stroke-width="0.8" stroke-dasharray="3 5"/>')
    rng = random.Random(seed)
    arcs = [_arc(lower, points[a], points[b], rng.choice((-1, 1)) * rng.uniform(*BEND)) for a, b in edges]
    longest = max(length for _, length in arcs)
    # Long links fade slightly so the local structure stays readable.
    for d, length in sorted(arcs, key=lambda arc: -arc[1]):
        strength = 0.62 - 0.22 * length / longest
        figure.body.append(f'<path d="{d}" fill="none" stroke="{palette.tone(strength)}" stroke-width="1"/>')
    radius = 4.2 if len(points) > 40 else 4.8
    for i in sorted(range(len(points)), key=lambda i: lower.rotate(*points[i])[1]):
        quality = (fitness[i] - low) / (top - low)
        if i == best:
            color, r = palette.accent, radius + 1.6
        elif i in optima:
            color, r = palette.mark, radius + 0.6
        else:
            color, r = palette.tone(0.3 + 0.45 * quality), radius
        figure.body.append(disc(lower.project(*points[i]), r, color, palette.surface, 1.2))

    figure.body += terrain(upper, height, palette, step=0.04, every=4, density=0.8)
    for i in optima:
        u, v = points[i]
        if upper.visible(height, u, v):
            figure.body.append(disc(upper.project(u, v, height(u, v) + 0.008), 5 if i == best else 2.8,
                                    palette.accent if i == best else palette.mark, palette.surface, 1.2))
    figure.label("surface", (24, 20), "title")
    figure.label("graph", (344, 467), "title")
    return figure, {"nodes": len(points), "local_optima": len(optima), "fdc": fdc}


def render_examples(out, tokens):
    """Write each scenario's figure and return metadata, including its graph's statistics."""
    metadata = {}
    for key in SCENARIOS:
        figure, stats = draw(key, tokens)
        (out / (figure.name + ".svg")).write_text(figure.svg())
        metadata[figure.name] = {**figure.metadata(), **stats}
    return metadata
