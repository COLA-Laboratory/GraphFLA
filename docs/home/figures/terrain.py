"""Line-art terrain: contour lines plus fall lines running down the slopes."""
import math

from .geometry import Occupancy, descend, find_peaks, trace_level, visible_runs
from .svg import path_data

CONTOUR_GRID = 120
EDGE_SAMPLES = 140
EDGES = (((0, 0), (0, 1)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)))


def mask(camera, height, fill, cells=30):
    """Return the surface painted in ``fill``, hiding whatever was drawn behind it."""
    out = ['<g stroke-width="0.8" stroke-linejoin="round">']
    order = sorted((camera.rotate((i + 0.5) / cells, (j + 0.5) / cells)[1], i, j)
                   for i in range(cells) for j in range(cells))
    for _, i, j in order:
        corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
        quad = [camera.project(a / cells, b / cells, height(a / cells, b / cells)) for a, b in corners]
        out.append(f'<path d="{path_data(quad)}Z" fill="{fill}" stroke="{fill}"/>')
    out.append("</g>")
    return out


def rim(camera, height, color):
    """Return the visible parts of the four edges of the surface."""
    out = []
    for a, b in EDGES:
        points = [(a[0] + (b[0] - a[0]) * k / EDGE_SAMPLES, a[1] + (b[1] - a[1]) * k / EDGE_SAMPLES)
                  for k in range(EDGE_SAMPLES + 1)]
        for run in visible_runs(camera, height, points, height):
            out.append(f'<path d="{path_data(run)}" stroke="{color}" stroke-width="0.8"/>')
    return out


def contours(camera, height, palette, step, every):
    """Return contour lines every ``step`` of height; each ``every``-th line is drawn bold.

    Lines strengthen with height so the peaks read as the brightest part.
    """
    n = CONTOUR_GRID
    grid = [[height(i / n, j / n) for j in range(n + 1)] for i in range(n + 1)]
    top = max(max(row) for row in grid)
    out = []
    k = 1
    while k * step < top:
        level = k * step
        bold = k % every == 0
        strength = min(1.0, 0.30 + (0.22 if bold else 0.0) + 0.46 * min(1.0, level) ** 0.7)
        d = "".join(path_data(run) for chain in trace_level(grid, n, level)
                    for run in visible_runs(camera, height, chain, lambda u, v, level=level: level))
        if d:
            out.append(f'<path d="{d}" stroke="{palette.tone(strength)}" stroke-width="{1.2 if bold else 0.65}"/>')
        k += 1
    return out


def fall_lines(camera, height, color, density=1.0, starts=()):
    """Return steepest-descent lines fanning out from every summit.

    A line ends where it would run closer than a fixed screen distance to another one.

    Parameters
    ----------
    density : float, default=1.0
        Multiplier on the number of lines leaving each summit.
    starts : iterable of (u, v), default=()
        Extra start points, for slopes that no summit line reaches.
    """
    gap = 5.5 / camera.scale
    seeds = [(u, v, 4) for u, v in starts]
    for pu, pv in sorted(find_peaks(height), key=lambda q: -height(*q)):
        count = max(8, round((8 + 22 * min(1.0, height(pu, pv))) * density))
        skip = math.ceil((gap * count / (2 * math.pi) - 0.02) / 0.004) + 2
        for a in range(count):
            angle = 2 * math.pi * (a + 0.5) / count
            seeds.append((pu + 0.02 * math.cos(angle), pv + 0.02 * math.sin(angle), max(4, skip)))
    taken = Occupancy(gap)
    d = ""
    for u, v, skip in seeds:
        route = descend(height, u, v, taken, skip)
        taken.add(route[skip:])
        if len(route) > 16:
            d += "".join(path_data(run) for run in visible_runs(camera, height, route[::2], height))
    return [f'<path d="{d}" stroke="{color}" stroke-width="0.6"/>'] if d else []


def terrain(camera, height, palette, step=0.035, every=4, density=1.0, masked=True, starts=()):
    """Return the full terrain: optional mask, rim, fall lines and contours.

    Parameters
    ----------
    masked : bool, default=True
        Paint the surface in the palette's surface colour first. Needed only when
        something is drawn behind the terrain.
    """
    body = mask(camera, height, palette.surface) if masked else []
    body.append('<g fill="none" stroke-linecap="round" stroke-linejoin="round">')
    body += rim(camera, height, palette.tone(0.42))
    body += fall_lines(camera, height, palette.tone(0.34), density, starts)
    body += contours(camera, height, palette, step, every)
    body.append("</g>")
    return body
