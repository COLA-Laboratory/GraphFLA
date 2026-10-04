"""SVG primitives shared by the figures."""
import math


def path_data(points):
    """Return the ``d`` attribute of a polyline through screen points."""
    return "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in points)


def stroke(points, color, width, casing=None, casing_width=2.4, dash=None):
    """Return a round-capped polyline, optionally on a wider casing that separates it from what lies below."""
    d = path_data(points)
    out = []
    if casing:
        out.append(f'<path d="{d}" fill="none" stroke="{casing}" stroke-width="{width + casing_width}" '
                   'stroke-linecap="round" stroke-linejoin="round"/>')
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
               f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')
    return "\n".join(out)


def disc(center, radius, fill, ring, ring_width=1.6):
    """Return a filled circle with a ring stroke."""
    return (f'<circle cx="{center[0]:.1f}" cy="{center[1]:.1f}" r="{radius}" fill="{fill}" '
            f'stroke="{ring}" stroke-width="{ring_width}"/>')


def arrow_head(points, color, size, casing):
    """Return a triangular head continuing the last segment of ``points``."""
    (x0, y0), (x1, y1) = points[-2], points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    tip = (x1 + dx * size * 0.55, y1 + dy * size * 0.55)
    left = (x1 - dx * size * 0.45 - dy * size * 0.5, y1 - dy * size * 0.45 + dx * size * 0.5)
    right = (x1 - dx * size * 0.45 + dy * size * 0.5, y1 - dy * size * 0.45 - dx * size * 0.5)
    return (f'<path d="{path_data([tip, left, right])}Z" fill="{color}" stroke="{casing}" stroke-opacity="0.75" '
            'stroke-width="1.2" stroke-linejoin="round"/>')


def document(width, height, body):
    """Return a complete SVG document. The page supplies the description as the image's alt text."""
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n' + "\n".join(body) + "\n</svg>\n"
