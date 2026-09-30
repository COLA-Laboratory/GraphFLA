"""Lazy analysis, data export and graph serialization workloads."""

from pathlib import Path
import tempfile

from graphfla.landscape import Landscape

from ._datasets import build_dataset

FLAGS = {
    "basins": "_basin_calculated",
    "accessible_paths": "_path_calculated",
    "dist_to_go": "_distance_calculated",
    "neighbor_fitness": "_neighbor_fit_calculated",
    "pagerank": "_pagerank_calculated",
}


class CachedProperties:
    params = (["CR6261", "synthetic-rna"], list(FLAGS), ["cold", "warm"])
    param_names = ["dataset", "property", "cache"]
    number = 1
    repeat = 5

    def setup(self, dataset, property_name, cache):
        self.landscape = build_dataset(dataset)
        if cache == "warm":
            getattr(self.landscape, property_name)

    def _read(self, property_name, cache):
        if cache == "cold":
            setattr(self.landscape, FLAGS[property_name], False)
            if (
                property_name == "pagerank"
                and "pagerank" in self.landscape.graph.vs.attributes()
            ):
                del self.landscape.graph.vs["pagerank"]
        return getattr(self.landscape, property_name)

    def time_property(self, dataset, property_name, cache):
        self._read(property_name, cache)

    def peakmem_property(self, dataset, property_name, cache):
        self._read(property_name, cache)


class LandscapeOperations:
    params = ["CR6261", "synthetic-rna"]
    param_names = ["dataset"]
    number = 1

    def setup(self, dataset):
        self.landscape = build_dataset(dataset)
        self.landscape.basins
        self.landscape.configs
        self.directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self.directory.name) / "landscape.graphml")
        self.landscape.to_graph(self.path)

    def teardown(self, dataset):
        self.directory.cleanup()

    def time_get_data(self, dataset):
        self.landscape.get_data()

    def time_configs(self, dataset):
        self.landscape._configs = None
        return self.landscape.configs

    def time_get_lon(self, dataset):
        self.landscape.get_lon(verbose=False)

    def time_to_graph(self, dataset):
        self.landscape.to_graph(self.path)

    def time_build_from_graph(self, dataset):
        Landscape.build_from_graph(self.path, verbose=False)

    def peakmem_get_data(self, dataset):
        self.landscape.get_data()

    def peakmem_get_lon(self, dataset):
        self.landscape.get_lon(verbose=False)

    def peakmem_to_graph(self, dataset):
        self.landscape.to_graph(self.path)

    def peakmem_build_from_graph(self, dataset):
        Landscape.build_from_graph(self.path, verbose=False)
