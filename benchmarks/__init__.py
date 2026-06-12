"""airspeed-velocity (asv) benchmark suite for GraphFLA.

Three benchmark groups, each on synthetic self-contained landscapes:

* ``construction`` -- ``build_from_data`` time and peak memory, per landscape kind/size
* ``analysis``     -- the landscape-analysis metrics on a fixed NK landscape
* ``algorithms``   -- the trajectory walkers (HillClimb / RandomWalk)

Run with ``asv`` (see benchmarks/README.md). These are NOT pytest tests.
"""
