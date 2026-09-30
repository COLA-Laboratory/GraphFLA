"""End-to-end construction; input loading is excluded from measurement."""

from ._datasets import DATASETS, load_dataset


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
