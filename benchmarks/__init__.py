"""airspeed-velocity (asv) benchmark suite for GraphFLA.

Three benchmark groups, run primarily on the real empirical landscapes committed
in ``data/`` (with synthetic NK fallbacks); see benchmarks/README.md:

* ``construction`` -- ``build_from_data`` time and peak memory, per landscape kind/size
* ``analysis``     -- the landscape-analysis metrics
* ``algorithms``   -- the trajectory walkers (HillClimb / RandomWalk)

Run with ``asv``. These are NOT pytest tests.
"""
