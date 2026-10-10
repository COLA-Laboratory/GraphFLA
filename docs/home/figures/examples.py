"""Workflow illustrations: each scenario's neighbor graph and the landscape above it.

Each scenario has its own search-space graph and authored surface. Fitness
decays with graph distance to the selected optima, whose coordinates also locate
the surface peaks. Both layers share one camera and the same candidate positions.
"""
import math

from .camera import Camera
from .graphs import distances, fit_disc, product, simplex, stress_layout
from .palette import Palette
from .scenes import Figure
from .svg import disc, path_data
from .facets import terrain

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
WIDTH, HEIGHT = 560, 560
UPPER = dict(cx=280, cy=205, scale=365, azimuth=38, tilt=26, zscale=178)
LOWER = dict(UPPER, cy=413)
# Both layers share the same camera; only the vertical screen position differs.
BEND = 0.06


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
    to_best = hops[best]
    others = [i for i in range(len(points)) if i != best]
    fdc = spearman([fitness[i] for i in others], [to_best[i] for i in others])

    floor = [lower.project(*point) for point in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))]
    figure.body.append(f'<path d="{path_data(floor)}" fill="none" stroke="{palette.tone(0.12)}" stroke-width=".65"/>')
    for i in centres:
        guide = [lower.project(*points[i]), upper.project(*points[i], height(*points[i]))]
        figure.body.append(f'<path d="{path_data(guide)}" fill="none" stroke="{palette.tone(0.18)}" '
                           'stroke-width=".65" stroke-dasharray="2 6"/>')
    figure.body += terrain(upper, height, palette)
    for i in centres:
        u, v = points[i]
        if upper.visible(height, u, v):
            figure.body.append(disc(upper.project(u, v, height(u, v) + 0.008), 5 if i == best else 2.8,
                                    palette.accent if i == best else palette.mark, palette.surface, 1.2))
    # Construct shallow arcs on the same floor plane, then project vertices and controls together.
    for a, b in edges:
        left, right = points[a], points[b]
        du, dv = right[0] - left[0], right[1] - left[1]
        midpoint = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        side = 1 if (midpoint[0] - 0.5) * (-dv) + (midpoint[1] - 0.5) * du > 0 else -1
        path, _ = _arc(lower, left, right, BEND * side)
        figure.body.append(f'<path data-edge="{a}-{b}" d="{path}" fill="none" '
                           f'stroke="{palette.tone(0.30)}" stroke-width=".85" stroke-linecap="round"/>')
    for i, point in enumerate(points):
        color, radius = ((palette.accent, 6.1) if i == best else
                         (palette.mark, 5.2) if i in centres else (palette.node, 4.8))
        mark = disc(lower.project(*point), radius, color, palette.surface, 1.4)
        figure.body.append(mark.replace('<circle ', f'<circle data-node="{i}" '))
    figure.label("surface", (24, 14), "title")
    figure.label("graph", (24, 326), "title")
    return figure, {"nodes": len(points), "local_optima": len(optima), "fdc": fdc}


def render_examples(out, tokens):
    """Write each scenario's figure and return metadata, including its graph's statistics."""
    metadata = {}
    for key in SCENARIOS:
        figure, stats = draw(key, tokens)
        (out / (figure.name + ".svg")).write_text(figure.svg())
        metadata[figure.name] = {**figure.metadata(), **stats}
    return metadata
