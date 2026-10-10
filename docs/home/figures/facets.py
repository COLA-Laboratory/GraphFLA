"""Deterministic triangular terrain in the landing page's graphite-gold palette."""
import math
import random

from .geometry import find_peaks
from .palette import _rgb
from .svg import path_data


def mix(left, right, amount):
    """Interpolate two six-digit RGB colours."""
    amount = max(0.0, min(1.0, amount))
    return "#" + "".join(f"{round(a + (b - a) * amount):02x}"
                         for a, b in zip(_rgb(left), _rgb(right)))


def terrain(camera, height, palette, **_line_options):
    """Sample the original height function without moving its summits or camera."""
    cells = 18
    peaks = find_peaks(height)
    rng = random.Random(882)
    grid = {}
    for i in range(cells + 1):
        for j in range(cells + 1):
            grid[i, j] = (
                i / cells + (rng.uniform(-0.18, 0.18) / cells if 0 < i < cells else 0),
                j / cells + (rng.uniform(-0.18, 0.18) / cells if 0 < j < cells else 0),
            )
    faces = []
    for i in range(cells):
        for j in range(cells):
            quad = [grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]]
            summit = next((point for point in peaks
                           if i / cells <= point[0] < (i + 1) / cells
                           and j / cells <= point[1] < (j + 1) / cells), None)
            triangles = ([(quad[k], quad[(k + 1) % 4], summit) for k in range(4)] if summit
                         else [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])])
            for triangle in triangles:
                points = [(u, v, height(u, v)) for u, v in triangle]
                u = sum(point[0] for point in triangle) / 3
                v = sum(point[1] for point in triangle) / 3
                a, b, c = points
                ab = [b[k] - a[k] for k in range(3)]
                ac = [c[k] - a[k] for k in range(3)]
                normal = (ab[1] * ac[2] - ab[2] * ac[1],
                          ab[2] * ac[0] - ab[0] * ac[2],
                          ab[0] * ac[1] - ab[1] * ac[0])
                norm = math.sqrt(sum(value * value for value in normal))
                light = 0.72 + 0.28 * max(0, (normal[0] * -0.6 + normal[1] * -0.35 + normal[2])
                                         / (norm * 1.218))
                z = sum(point[2] for point in points) / 3
                pigment = (mix(palette.facet_low, palette.facet_mid, z / 0.45) if z < 0.45
                           else mix(palette.facet_mid, palette.facet_high, (z - 0.45) / 0.55))
                fill = mix(palette.facet_edge, pigment, light)
                outline = path_data([camera.project(*point) for point in points])
                face = (f'<path d="{outline}Z" fill="{fill}" stroke="{palette.facet_edge}" '
                        'stroke-width="0.75" stroke-linejoin="round"/>')
                faces.append((camera.rotate(u, v)[1], face))
    return [face for _, face in sorted(faces)]
