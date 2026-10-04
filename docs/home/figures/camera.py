"""Oblique camera that projects the unit square and its heights onto the screen."""
import math


class Camera:
    """Oblique projection of a surface ``z = f(u, v)`` over the unit square.

    Parameters
    ----------
    cx, cy : float
        Screen position of the centre of the unit square at height zero.
    scale : float
        Screen length of one unit along ``u`` or ``v``.
    azimuth, tilt : float
        Rotation about the vertical axis and viewing elevation, in degrees.
    zscale : float
        Screen length of one unit of height.
    lift : float, default=0.0
        Upward screen offset applied to everything the camera projects.
    """

    def __init__(self, cx, cy, scale, azimuth, tilt, zscale, lift=0.0):
        self.cx, self.cy, self.scale, self.zscale, self.lift = cx, cy, scale, zscale, lift
        self.cos_az, self.sin_az = math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))
        self.sin_tilt = math.sin(math.radians(tilt))

    def rotate(self, u, v):
        """Return (across, depth) of a point; depth grows toward the viewer."""
        x, y = u - 0.5, v - 0.5
        return x * self.cos_az - y * self.sin_az, x * self.sin_az + y * self.cos_az

    def project(self, u, v, z=0.0):
        """Return the screen position of the point (u, v) at height z."""
        across, depth = self.rotate(u, v)
        return (self.cx + self.scale * across,
                self.cy + self.scale * depth * self.sin_tilt - self.zscale * z - self.lift)

    def visible(self, height, u, v):
        """Return True when no nearer part of the surface hides the point (u, v)."""
        z = height(u, v)
        rise = self.scale * self.sin_tilt / self.zscale
        t = 0.01
        while t < 1.5:
            uu, vv = u + t * self.sin_az, v + t * self.cos_az
            if not (0 <= uu <= 1 and 0 <= vv <= 1):
                break
            if height(uu, vv) > z + t * rise + 0.004:
                return False
            t += 0.01
        return True
