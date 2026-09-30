"""asv benchmarks: trajectory walkers (HillClimb / RandomWalk).

Run on real empirical landscapes large enough for meaningful trajectory timing
(a boolean antibody landscape and a protein landscape). The landscape and its
``SearchCache`` are built once in ``setup``; each ``time_*`` runs a batch of
walks from a fixed, seeded set of start nodes.
"""

import random

import numpy as np

from graphfla.algorithms import HillClimb, RandomWalk, SearchCache

from . import _datasets

N_STARTS = 2000
WALK_LENGTH = 100
SEED = 99


class Trajectories:
    # Large real landscapes: CR9114 (~65k boolean), GB1 (~150k protein).
    params = ["CR9114", "GB1"]
    param_names = ["dataset"]

    def setup(self, dataset):
        cls, X, f = _datasets.load_real(dataset)  # skipped if data absent
        landscape = cls(maximize=True)
        landscape.build_from_data(X, f, verbose=False)
        self.graph = landscape.graph
        self.cache = SearchCache(self.graph)
        rng = np.random.default_rng(123)
        self.starts = rng.integers(
            0, landscape.n_configs, size=N_STARTS, dtype=np.int64
        ).tolist()

    def time_hillclimb_best(self, dataset):
        climber = HillClimb(self.cache, strategy="best-improvement")
        for s in self.starts:
            climber.descend(s)

    def time_hillclimb_first(self, dataset):
        climber = HillClimb(self.cache, strategy="first-improvement", seed=SEED)
        for s in self.starts:
            climber.descend(s)

    def time_randomwalk(self, dataset):
        master = random.Random(SEED)
        for s in self.starts:
            RandomWalk(self.cache, length=WALK_LENGTH, seed=master.getrandbits(32)).run(
                s
            )

    def time_search_cache(self, dataset):
        SearchCache(self.graph)

    def peakmem_search_cache(self, dataset):
        SearchCache(self.graph)

    def peakmem_hillclimb_best(self, dataset):
        self.time_hillclimb_best(dataset)

    def peakmem_hillclimb_first(self, dataset):
        self.time_hillclimb_first(dataset)

    def peakmem_randomwalk(self, dataset):
        self.time_randomwalk(dataset)
