"""asv benchmarks: landscape-analysis metrics.

One fixed NK boolean landscape is built once (``setup``) and its lazy caches
(basins, distance-to-optimum, neighbour fitness, accessible paths) are warmed,
so each ``time_*`` measures only that metric's own work.
"""

from graphfla.landscape import BooleanLandscape
from graphfla import analysis as A

from . import _datasets

SEED = 0


class Analysis:
    # asv runs setup() before each benchmark; building+warming once here keeps
    # the per-metric timings clean. Sized at 2**12 = 4096 configurations.
    def setup(self):
        X, f = _datasets.nk_boolean(n=12, k=2, seed=0)
        landscape = BooleanLandscape(maximize=True)
        landscape.build_from_data(X, f, verbose=False)
        # warm the lazy, cached prerequisites the metrics read
        landscape.basins
        landscape.dist_to_go
        landscape.neighbor_fitness
        landscape.accessible_paths
        self.ls = landscape

    # --- ruggedness / structure ---
    def time_local_optima_ratio(self):
        A.local_optima_ratio(self.ls)

    def time_gradient_intensity(self):
        A.gradient_intensity(self.ls)

    def time_autocorrelation(self):
        A.autocorrelation(self.ls, seed=SEED)

    def time_r_s_ratio(self):
        A.r_s_ratio(self.ls)

    def time_neutrality(self):
        A.neutrality(self.ls)

    # --- correlations ---
    def time_fdc(self):
        A.fdc(self.ls)

    def time_basin_fitness_correlation(self):
        A.basin_fitness_correlation(self.ls)

    def time_neighbor_fitness_correlation(self):
        A.neighbor_fitness_correlation(self.ls)

    def time_fitness_flattening_index(self):
        A.fitness_flattening_index(self.ls)

    # --- navigability ---
    def time_global_optima_accessibility(self):
        A.global_optima_accessibility(self.ls)

    def time_mean_path_length_to_global_optimum(self):
        A.mean_path_length_to_global_optimum(self.ls)

    def time_mean_distance_to_global_optimum(self):
        A.mean_distance_to_global_optimum(self.ls)

    # --- robustness ---
    def time_evolvability_enhancing_mutations(self):
        A.evolvability_enhancing_mutations(self.ls)

    def time_all_mutation_effects(self):
        A.all_mutation_effects(self.ls, n_jobs=1)

    # --- epistasis ---
    def time_gamma(self):
        A.gamma(self.ls, n_jobs=1)

    def time_higher_order_epistasis(self):
        A.higher_order_epistasis(self.ls, order=2)

    def time_walsh_hadamard(self):
        A.walsh_hadamard(self.ls, max_order=2)

    def time_classify_epistasis(self):
        A.classify_epistasis(self.ls, approximate=True, sample_cut_prob=0.5, seed=SEED)

    def time_extradimensional_bypass(self):
        A.extradimensional_bypass(
            self.ls, approximate=True, sample_cut_prob=0.5, seed=SEED
        )

    def time_global_idiosyncratic_index(self):
        A.global_idiosyncratic_index(self.ls, n_jobs=1)

    def time_diminishing_returns_index(self):
        A.diminishing_returns_index(self.ls)

    def time_increasing_costs_index(self):
        A.increasing_costs_index(self.ls)
