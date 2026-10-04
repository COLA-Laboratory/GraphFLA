"""Contours, visibility and gradient walks on a height function over the unit square."""
import math


def trace_level(grid, n, level):
    """Trace one contour level of an (n + 1) x (n + 1) grid with marching squares.

    Returns
    -------
    list of list of (u, v)
        Contour polylines, each chained from the per-cell segments.
    """
    segments = []
    for i in range(n):
        for j in range(n):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            vals = [grid[a][b] for a, b in corners]
            if min(vals) >= level or max(vals) < level:
                continue
            hits = []
            for a in range(4):
                (i0, j0), (i1, j1) = corners[a], corners[(a + 1) % 4]
                z0, z1 = vals[a], vals[(a + 1) % 4]
                if (z0 < level) != (z1 < level):
                    t = (level - z0) / (z1 - z0)
                    hits.append(((i0 + (i1 - i0) * t) / n, (j0 + (j1 - j0) * t) / n))
            if len(hits) == 2:
                segments.append((hits[0], hits[1]))
            elif len(hits) == 4:
                # Saddle cell: the cell mean decides which pairs of crossings connect.
                if (sum(vals) / 4 < level) == (vals[0] < level):
                    segments += [(hits[0], hits[1]), (hits[2], hits[3])]
                else:
                    segments += [(hits[3], hits[0]), (hits[1], hits[2])]
    key = lambda q: (round(q[0], 7), round(q[1], 7))
    ends = {}
    for index, (a, b) in enumerate(segments):
        ends.setdefault(key(a), []).append(index)
        ends.setdefault(key(b), []).append(index)
    used = [False] * len(segments)
    lines = []
    for index in range(len(segments)):
        if used[index]:
            continue
        used[index] = True
        chain = list(segments[index])
        for forward in (True, False):
            while True:
                tip = chain[-1] if forward else chain[0]
                options = [k for k in ends.get(key(tip), []) if not used[k]]
                if not options:
                    break
                used[options[0]] = True
                a, b = segments[options[0]]
                other = b if key(a) == key(tip) else a
                if forward:
                    chain.append(other)
                else:
                    chain.insert(0, other)
        lines.append(chain)
    return lines


def visible_runs(camera, height, points, z_of):
    """Split a (u, v) polyline into the stretches the camera can see, projected to the screen."""
    runs, current = [], []
    for u, v in points:
        if camera.visible(height, u, v):
            current.append(camera.project(u, v, z_of(u, v)))
        else:
            if len(current) > 1:
                runs.append(current)
            current = []
    if len(current) > 1:
        runs.append(current)
    return runs


def _gradient(height, u, v, eps=1e-3):
    return ((height(u + eps, v) - height(u - eps, v)) / (2 * eps),
            (height(u, v + eps) - height(u, v - eps)) / (2 * eps))


def ascend(height, u, v, step=0.004, limit=3000):
    """Return the steepest-ascent route from (u, v) to the peak of its basin."""
    route = [(u, v)]
    for _ in range(limit):
        gu, gv = _gradient(height, u, v)
        norm = math.hypot(gu, gv)
        if norm < 0.02:
            break
        nu, nv = u + step * gu / norm, v + step * gv / norm
        if not (0 <= nu <= 1 and 0 <= nv <= 1) or height(nu, nv) <= height(u, v):
            break
        u, v = nu, nv
        route.append((u, v))
    return route


def winding_ascent(height, u, v, step=0.004, limit=3000):
    """Follow improving steps with smooth lateral turns instead of steepest ascent."""
    route = [(u, v)]
    for k in range(limit):
        gu, gv = _gradient(height, u, v)
        norm = math.hypot(gu, gv)
        if norm < 0.02:
            break
        angle = 0.75 * math.sin(k * 0.055)
        du = (gu * math.cos(angle) - gv * math.sin(angle)) / norm
        dv = (gu * math.sin(angle) + gv * math.cos(angle)) / norm
        nu, nv = u + step * du, v + step * dv
        if not (0 <= nu <= 1 and 0 <= nv <= 1) or height(nu, nv) <= height(u, v):
            break
        u, v = nu, nv
        route.append((u, v))
    return route


def descend(height, u, v, taken, skip, flat=0.9):
    """Return the steepest-descent route from (u, v), stopped where it would crowd ``taken``.

    The first ``skip`` steps ignore ``taken`` so that lines can leave a shared summit.
    """
    route = [(u, v)]
    for step in range(700):
        gu, gv = _gradient(height, u, v)
        norm = math.hypot(gu, gv)
        if (step > 14 and norm < flat) or norm < 1e-6:
            break
        u, v = u - 0.004 * gu / norm, v - 0.004 * gv / norm
        if not (0 <= u <= 1 and 0 <= v <= 1):
            break
        if step > skip and taken.near(u, v):
            break
        route.append((u, v))
    return route


def pick_start(camera, height, target, region, min_visible=0.92, walk=ascend):
    """Return the longest mostly-visible ascent route that starts in ``region`` and ends on ``target``."""
    best = None
    for i in range(41):
        for j in range(41):
            start = (0.02 + 0.96 * i / 40, 0.02 + 0.96 * j / 40)
            if not region(*start):
                continue
            route = walk(height, *start)
            end = route[-1]
            if math.hypot(end[0] - target[0], end[1] - target[1]) > 0.04:
                continue
            probe = route[::8]
            if sum(camera.visible(height, *q) for q in probe) / len(probe) < min_visible:
                continue
            if best is None or len(route) > len(best):
                best = route
    return best


def local_maxima(grid, floor):
    """Return (u, v, z) of the grid nodes higher than ``floor`` and all eight neighbours."""
    n = len(grid) - 1
    peaks = []
    for i in range(1, n):
        for j in range(1, n):
            z = grid[i][j]
            if z > floor and all(z > grid[i + a][j + b] for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b):
                peaks.append((i / n, j / n, z))
    return peaks


def find_peaks(height, floor=0.12, n=80):
    """Return the summits of ``height``, refined by ascent from the grid maxima."""
    grid = [[height(i / n, j / n) for j in range(n + 1)] for i in range(n + 1)]
    peaks = []
    for u, v, _ in local_maxima(grid, floor):
        top = ascend(height, u, v)[-1]
        if all(math.hypot(top[0] - p[0], top[1] - p[1]) > 0.03 for p in peaks):
            peaks.append(top)
    return peaks


class Occupancy:
    """Spatial hash of points already used by fall lines.

    Parameters
    ----------
    gap : float
        Closest distance, in (u, v) units, allowed between two lines.
    """

    def __init__(self, gap):
        self.gap = gap
        self.cells = {}

    def near(self, u, v):
        """Return True when a recorded point lies within ``gap`` of (u, v)."""
        ci, cj = int(u / self.gap), int(v / self.gap)
        for i in (ci - 1, ci, ci + 1):
            for j in (cj - 1, cj, cj + 1):
                for pu, pv in self.cells.get((i, j), ()):
                    if math.hypot(u - pu, v - pv) < self.gap:
                        return True
        return False

    def add(self, points):
        """Record the points of a finished line."""
        for u, v in points:
            self.cells.setdefault((int(u / self.gap), int(v / self.gap)), []).append((u, v))
