"""asv benchmarks: landscape construction (``build_from_data``).

Times the full build and measures peak memory, parametrised over landscape kind
and size. The synthetic data is generated in ``setup`` (excluded from the timing).
"""

from graphfla.landscape import BooleanLandscape, OrdinalLandscape, DNALandscape

from . import _datasets


def _make(dataset):
    """Return ``(landscape_cls, X, f)`` for a dataset label."""
    if dataset == "boolean-n10":
        return BooleanLandscape, *_datasets.nk_boolean(10)
    if dataset == "boolean-n12":
        return BooleanLandscape, *_datasets.nk_boolean(12)
    if dataset == "boolean-n14":
        return BooleanLandscape, *_datasets.nk_boolean(14)
    if dataset == "ordinal-6x3":
        return OrdinalLandscape, *_datasets.random_ordinal(6, 3)
    if dataset == "dna-l6":
        return DNALandscape, *_datasets.nk_dna(6)
    raise ValueError(dataset)


class Construction:
    """Build time, peak memory and edge count per landscape kind/size."""

    params = ["boolean-n10", "boolean-n12", "boolean-n14", "ordinal-6x3", "dna-l6"]
    param_names = ["dataset"]

    def setup(self, dataset):
        self.cls, self.X, self.f = _make(dataset)

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
