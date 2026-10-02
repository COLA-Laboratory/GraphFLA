"""End-to-end construction; input loading is excluded from measurement."""

from .._datasets import DATASETS, load_dataset


class Construction:
    params = DATASETS
    param_names = ["dataset"]
    timeout = 180
    number = 1
    repeat = 5
    processes = 3
    warmup_time = 0.2

    def setup(self, dataset):
        self.cls, self.X, self.f, self.kwargs = load_dataset(dataset)

    def _build(self):
        return self.cls().build_from_data(self.X, self.f, verbose=False, **self.kwargs)

    def time_build(self, dataset):
        self._build()

    def peakmem_build(self, dataset):
        self._build()

    def track_n_edges(self, dataset):
        return self._build().n_edges

    def track_n_variants(self, dataset):
        return self._build().n_configs

    def track_variable_sites(self, dataset):
        return self._build().n_vars


class NeighborhoodStrategies:
    params = (
        ["dense", "sparse", "wide"],
        [
            ("active", 1),
            ("pairwise", 1),
            ("broadcast", 1),
            ("pairwise", 2),
            ("broadcast", 2),
        ],
    )
    param_names = ["geometry", "strategy_and_radius"]
    timeout = 120

    def setup(self, geometry, strategy_and_radius):
        import numpy as np
        from graphfla.landscape import BooleanLandscape
        from .._datasets import nk_boolean

        self.cls = BooleanLandscape
        if geometry == "wide":
            self.X = np.concatenate(
                [np.zeros((1, 96), dtype=int), np.eye(96, dtype=int)]
            )
        else:
            X, _ = nk_boolean(10 if geometry == "dense" else 14)
            self.X = (
                X
                if geometry == "dense"
                else X.sample(n=1000, random_state=0).reset_index(drop=True)
            )
        self.fitness = np.random.default_rng(1).normal(size=len(self.X))
        self.strategy, self.radius = strategy_and_radius

    def _build(self):
        return self.cls().build_from_data(
            self.X,
            self.fitness,
            neighborhood_strategy=self.strategy,
            n_edit=self.radius,
            verbose=False,
        )

    def time_build(self, geometry, strategy_and_radius):
        self._build()

    def peakmem_build(self, geometry, strategy_and_radius):
        self._build()
