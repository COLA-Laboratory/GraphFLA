"""Height functions of the illustrated landscapes, each defined on the unit square."""
import math
import random


def _bump(u, v, cu, cv, su, sv, amplitude):
    return amplitude * math.exp(-0.5 * (((u - cu) / su) ** 2 + ((v - cv) / sv) ** 2))


def hero(u, v):
    """One global peak, two local peaks and a low shoulder."""
    z = _bump(u, v, 0.47, 0.47, 0.36, 0.36, 0.16)
    z += _bump(u, v, 0.40, 0.36, 0.085, 0.10, 0.84)
    z += _bump(u, v, 0.78, 0.54, 0.10, 0.09, 0.56)
    z += _bump(u, v, 0.22, 0.70, 0.085, 0.09, 0.46)
    z += _bump(u, v, 0.72, 0.17, 0.07, 0.07, 0.22)
    z += 0.016 * (math.sin(9 * u + 1.3) * math.cos(8 * v + 0.4) + math.sin(13 * v + 3 * u))
    return max(0.0, z)


def workflow(u, v):
    """Three peaks of decreasing height."""
    z = _bump(u, v, 0.5, 0.5, 0.38, 0.38, 0.14)
    z += _bump(u, v, 0.36, 0.40, 0.10, 0.11, 0.80)
    z += _bump(u, v, 0.72, 0.62, 0.11, 0.10, 0.50)
    z += _bump(u, v, 0.25, 0.78, 0.07, 0.07, 0.30)
    return z


def smooth(u, v):
    """A single broad peak."""
    return _bump(u, v, 0.5, 0.5, 0.36, 0.36, 0.12) + _bump(u, v, 0.48, 0.46, 0.15, 0.16, 0.74)


_rng = random.Random(5)
_RUGGED_BUMPS = [(_rng.uniform(0.12, 0.88), _rng.uniform(0.12, 0.88), _rng.uniform(0.05, 0.066),
                  _rng.uniform(0.16, 0.40)) for _ in range(15)]


def rugged(u, v):
    """The smooth peak broken up by many narrow local peaks."""
    z = 0.62 * smooth(u, v)
    for cu, cv, s, amplitude in _RUGGED_BUMPS:
        z += _bump(u, v, cu, cv, s, s, amplitude * (0.35 + smooth(cu, cv)))
    return z


def two_peaks(u, v):
    """Two separated peaks of similar height."""
    return (_bump(u, v, 0.5, 0.5, 0.4, 0.4, 0.10) + _bump(u, v, 0.68, 0.30, 0.11, 0.11, 0.80)
            + _bump(u, v, 0.30, 0.70, 0.11, 0.11, 0.66))


def three_peaks(u, v):
    """One global peak and two lower local peaks."""
    return (_bump(u, v, 0.5, 0.5, 0.38, 0.38, 0.15) + _bump(u, v, 0.42, 0.36, 0.10, 0.11, 0.82)
            + _bump(u, v, 0.77, 0.60, 0.10, 0.09, 0.50) + _bump(u, v, 0.22, 0.74, 0.08, 0.08, 0.40))


def plateau(u, v):
    """A flat-topped plateau next to a higher peak."""
    d = math.hypot((u - 0.40) / 0.27, (v - 0.60) / 0.22)
    return 0.03 + 0.40 / (1 + math.exp((d - 1) / 0.11)) + _bump(u, v, 0.66, 0.34, 0.085, 0.085, 0.56)
