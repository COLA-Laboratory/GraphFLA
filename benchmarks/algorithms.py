"""asv benchmarks: trajectory walkers (HillClimb / RandomWalk).

A fixed NK boolean landscape and its ``SearchCache`` are built once (``setup``);
each ``time_*`` runs a batch of walks from a fixed, seeded set of start nodes.
"""

import random

import numpy as np

from graphfla.landscape import BooleanLandscape
from graphfla.algorithms import HillClimb, RandomWalk, SearchCache

from . import _datasets

N_STARTS = 2000
WALK_LENGTH = 100
SEED = 99


class Trajectories:
    def setup(self):
        X, f = _datasets.nk_boolean(n=12, k=2, seed=0)
        landscape = BooleanLandscape(maximize=True)
        landscape.build_from_data(X, f, verbose=False)
        self.cache = SearchCache(landscape.graph)
        rng = np.random.default_rng(123)
        self.starts = rng.integers(
            0, landscape.n_configs, size=N_STARTS, dtype=np.int64
        ).tolist()

    def time_hillclimb_best(self):
        climber = HillClimb(self.cache, strategy="best-improvement")
        for s in self.starts:
            climber.descend(s)

    def time_hillclimb_first(self):
        climber = HillClimb(self.cache, strategy="first-improvement", seed=SEED)
        for s in self.starts:
            climber.descend(s)

    def time_randomwalk(self):
        master = random.Random(SEED)
        for s in self.starts:
            RandomWalk(self.cache, length=WALK_LENGTH, seed=master.getrandbits(32)).run(s)
