"""asv benchmarks: landscape construction (``build_from_data``).

Build time, peak memory and edge count over the curated *real* empirical
landscapes (protein / DNA / boolean / ordinal, ~500 to ~260k configurations),
plus a few synthetic cubes for quick portable runs. The data is loaded in
``setup`` (excluded from the timing); real datasets are skipped automatically
when their data file is not present.
"""

from . import _datasets

# Real empirical landscapes (small -> large) + synthetic fallbacks.
REAL = list(_datasets.REAL)  # WReOs, CR6261, TrpB3I, Westmann, CR9114, GB1, Papkou
SYNTHETIC = ["synthetic-boolean", "synthetic-ordinal", "synthetic-dna"]


def _synthetic(name):
    if name == "synthetic-boolean":
        from graphfla.landscape import BooleanLandscape
        return BooleanLandscape, *_datasets.nk_boolean(12)
    if name == "synthetic-ordinal":
        from graphfla.landscape import OrdinalLandscape
        return OrdinalLandscape, *_datasets.random_ordinal(6, 3)
    if name == "synthetic-dna":
        from graphfla.landscape import DNALandscape
        return DNALandscape, *_datasets.nk_dna(6)
    raise ValueError(name)


class Construction:
    params = REAL + SYNTHETIC
    param_names = ["dataset"]

    def setup(self, dataset):
        if dataset in _datasets.REAL:
            self.cls, self.X, self.f = _datasets.load_real(dataset)  # may skip
        else:
            self.cls, self.X, self.f = _synthetic(dataset)

    def _build(self):
        landscape = self.cls(maximize=True)
        landscape.build_from_data(self.X, self.f, verbose=False)
        return landscape

    def time_build(self, dataset):
        self._build()

    def peakmem_build(self, dataset):
        self._build()

    def track_n_edges(self, dataset):
        return self._build().n_edges
