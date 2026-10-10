"""The six figures of the landing page, and where their text labels go."""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import surfaces
from .camera import Camera
from .geometry import ascend, find_peaks, pick_start, winding_ascent
from .palette import Palette, load_tokens
from .svg import arrow_head, disc, document, path_data, stroke
from .facets import terrain

CARD_WIDTH, CARD_HEIGHT = 560, 300


@dataclass
class Figure:
    """One rendered figure.

    Attributes
    ----------
    name : str
        File stem and key in the metadata.
    width, height : int
        Size of the SVG view box.
    body : list of str
        SVG elements in paint order.
    labels : dict
        Position (percent of the figure size) and kind of each text label. The page
        lays the label text over the image, so the wording stays out of the SVG.
    """

    name: str
    width: int
    height: int
    body: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)

    def label(self, key, point, kind, **extra):
        """Register a text label whose top-left corner sits at the screen ``point``."""
        self.labels[key] = {"x": round(100 * point[0] / self.width, 1),
                            "y": round(100 * point[1] / self.height, 1), "kind": kind, **extra}

    def svg(self):
        return document(self.width, self.height, self.body)

    def metadata(self):
        return {"file": f"{self.name}.svg", "width": self.width, "height": self.height, "labels": self.labels}


def _card_camera(cx=280, cy=200, scale=300, zscale=150):
    return Camera(cx, cy, scale, 38, 26, zscale)


def _walk(camera, height, route, color, palette, width=2.0, steps=True, end_radius=5.2):
    """Return an adaptive walk drawn on the surface, and its screen points.

    The walk starts at a hollow marker and ends at a filled one; ``steps`` adds a bead
    every few mutations.
    """
    points = [camera.project(u, v, height(u, v) + 0.006) for u, v in route[::3] + [route[-1]]]
    out = [stroke(points, color, width, casing=palette.surface)]
    if steps:
        for k in range(26, len(route) - 12, 26):
            bead = camera.project(*route[k], height(*route[k]) + 0.006)
            out.append(disc(bead, 3.0, color, palette.surface, 1.4))
    out.append(disc(points[0], 4.8, palette.surface, color, 2))
    out.append(disc(points[-1], end_radius, color, palette.surface, 2))
    return out, points


def _height_axis(camera, top, color, origin=0.0):
    """Return a vertical arrow at the left corner of the floor, and the position of its tip."""
    x, y0 = camera.project(0.0, 1.0, origin)
    _, y1 = camera.project(0.0, 1.0, top)
    return (f'<path d="M{x:.1f} {y0:.1f}V{y1:.1f}" stroke="{color}" stroke-width="1.2" fill="none"/>'
            f'<path d="M{x:.1f} {y1 - 7:.1f}l-4 8h8Z" fill="{color}"/>'), (x, y1)


def hero(palette):
    """Three adaptive walks on a landscape with one global and two local peaks."""
    figure = Figure("hero", 760, 520)
    camera = Camera(385, 338, 505, 38, 27, 265)
    height = surfaces.hero
    figure.body = terrain(camera, height, palette)

    local_ends = []
    for start in ((0.97, 0.80), (0.06, 0.93)):
        marks, points = _walk(camera, height, winding_ascent(height, *start), palette.mark, palette, 1.9)
        figure.body += marks
        local_ends.append(points[-1])
    in_front = lambda u, v: camera.rotate(u, v)[1] > 0.12
    route = pick_start(camera, height, (0.40, 0.36), in_front, walk=winding_ascent)
    marks, points = _walk(camera, height, route, palette.accent, palette, 2.5, end_radius=6.6)
    figure.body += marks
    axis, axis_tip = _height_axis(camera, 0.62, palette.muted, origin=height(0, 1))
    figure.body.append(axis)

    summit = points[-1]
    figure.label("global_peak", (summit[0] + 23, summit[1] - 14), "accent")
    figure.label("local_peak_east", (local_ends[0][0] + 23, local_ends[0][1] - 23), "strong")
    figure.label("local_peak_west", (local_ends[1][0] - 118, local_ends[1][1] - 36), "strong")
    figure.label("height_axis", (axis_tip[0] + 12, axis_tip[1] - 4), "axis")
    left, front = camera.project(0, 1), camera.project(1, 1)
    slope = math.degrees(math.atan2(front[1] - left[1], front[0] - left[0]))
    along = 0.36
    figure.label("floor_axis", (left[0] + along * (front[0] - left[0]) - 30,
                                left[1] + along * (front[1] - left[1]) + 10), "axis", rotate=round(slope, 1))
    return figure


def ruggedness(palette):
    """A single-peaked landscape beside one with many local peaks."""
    figure = Figure("ruggedness", CARD_WIDTH, CARD_HEIGHT)
    for key, raw, cx in (("smooth", surfaces.smooth, 146), ("rugged", surfaces.rugged, 414)):
        camera = _card_camera(cx, 196, 186, 118)
        top = max(raw(i / 60, j / 60) for i in range(61) for j in range(61))
        height = lambda u, v, raw=raw, top=top: raw(u, v) / top
        figure.body += terrain(camera, height, palette, step=0.0625, every=4, density=0.6, masked=False)
        peaks = sorted(find_peaks(height), key=lambda p: height(*p))
        for u, v in peaks:
            if camera.visible(height, u, v):
                highest = (u, v) == peaks[-1]
                figure.body.append(disc(camera.project(u, v, height(u, v) + 0.01), 4.6 if highest else 3.4,
                                        palette.accent if highest else palette.mark, palette.surface, 1.5))
        figure.label(key, (cx, 0.88 * CARD_HEIGHT), "caption")
    return figure


def epistasis(palette):
    """The same step climbs toward a peak in one place and descends from a peak in another."""
    figure = Figure("epistasis", CARD_WIDTH, CARD_HEIGHT)
    camera = _card_camera()
    height = surfaces.two_peaks
    figure.body = terrain(camera, height, palette, step=0.05, every=4, density=0.75, masked=False)
    for v, u0, u1, color in ((0.70, 0.335, 0.59, palette.mark), (0.30, 0.36, 0.615, palette.accent)):
        steps = [u0 + (u1 - u0) * k / 24 for k in range(25)]
        points = [camera.project(u, v, height(u, v) + 0.012) for u in steps]
        figure.body.append(stroke(points, color, 2.4, casing=palette.surface))
        figure.body.append(arrow_head(points, color, 12, palette.surface))
        figure.body.append(disc(points[0], 4.4, color, palette.surface, 1.6))
    return figure


def navigability(palette):
    """Five adaptive walks: two reach the global peak, three stop on local peaks."""
    figure = Figure("navigability", CARD_WIDTH, CARD_HEIGHT)
    camera = _card_camera()
    height = surfaces.three_peaks
    figure.body = terrain(camera, height, palette, step=0.05, every=4, density=0.75, masked=False)
    for start in ((0.95, 0.84), (0.05, 0.95), (0.62, 0.97)):
        figure.body += _walk(camera, height, ascend(height, *start), palette.mark, palette, 1.8,
                             steps=False, end_radius=4.8)[0]
    for region in (lambda u, v: camera.rotate(u, v)[1] > 0.10, lambda u, v: camera.rotate(u, v)[0] < -0.25):
        route = pick_start(camera, height, (0.42, 0.36), region)
        figure.body += _walk(camera, height, route, palette.accent, palette, 2.2, steps=False, end_radius=6)[0]
    return figure


def neutrality(palette):
    """A neutral network on a plateau, and the one step that leaves it for a higher peak."""
    figure = Figure("neutrality", CARD_WIDTH, CARD_HEIGHT)
    camera = _card_camera()
    height = surfaces.plateau
    # The plateau has no summit to seed fall lines from, so they start on a ring around its rim.
    ring = [(0.40 + 0.27 * 0.8 * math.cos(2 * math.pi * (a + 0.5) / 26),
             0.60 + 0.22 * 0.8 * math.sin(2 * math.pi * (a + 0.5) / 26)) for a in range(26)]
    body = terrain(camera, height, palette, step=0.05, every=4, density=0.75, masked=False, starts=ring)
    nodes = [(0.28, 0.62), (0.36, 0.52), (0.34, 0.70), (0.44, 0.62), (0.42, 0.73), (0.50, 0.58),
             (0.26, 0.53), (0.46, 0.49)]
    links = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (3, 5), (1, 6), (0, 6), (1, 7), (5, 7), (0, 3), (2, 3)]
    on_surface = lambda u, v: camera.project(u, v, height(u, v) + 0.012)
    placed = [on_surface(u, v) for u, v in nodes]
    for a, b in links:
        body.append(stroke([placed[a], placed[b]], palette.mark, 1.5, casing=palette.surface,
                           casing_width=1.6, dash="1 4.5"))
    exit_node = nodes[7]
    peak = ascend(height, 0.64, 0.36)[-1]
    climb = [on_surface(exit_node[0] + (peak[0] - exit_node[0]) * k / 24,
                        exit_node[1] + (peak[1] - exit_node[1]) * k / 24) for k in range(22)]
    body.append(stroke(climb, palette.accent, 2.3, casing=palette.surface))
    body.append(arrow_head(climb, palette.accent, 11, palette.surface))
    for point in placed:
        body.append(disc(point, 4.4, palette.mark, palette.surface, 1.6))
    body.append(disc(on_surface(*peak), 6, palette.accent, palette.surface, 1.8))
    figure.body = body
    return figure


# Scene and the surface token its figure is displayed on.
SCENES = {
    "hero": (hero, "page"),
    "ruggedness": (ruggedness, "card"),
    "epistasis": (epistasis, "card"),
    "navigability": (navigability, "card"),
    "neutrality": (neutrality, "card"),
}


def render(out_dir, tokens=None, names=None):
    """Write the figures as SVG files next to a ``figures.json`` describing them.

    Parameters
    ----------
    out_dir : path-like
        Directory the files are written to.
    tokens : dict, optional
        Design tokens; read from ``styles/tokens.css`` when omitted.
    names : iterable of str, optional
        Subset of ``SCENES`` to render. All figures by default.

    Returns
    -------
    dict
        ``{name: {"file", "width", "height", "labels"}}`` for the rendered figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens() if tokens is None else tokens
    metadata = {}
    for name in names or SCENES:
        scene, surface = SCENES[name]
        figure = scene(Palette.from_tokens(surface, tokens))
        (out_dir / f"{name}.svg").write_text(figure.svg())
        metadata[name] = figure.metadata()
    (out_dir / "figures.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
