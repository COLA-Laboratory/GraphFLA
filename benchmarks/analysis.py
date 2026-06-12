"""asv benchmarks: landscape-analysis metrics.

Run on real empirical landscapes of two encodings (a boolean antibody landscape
and a protein landscape), each kept moderate so the heavy metrics stay tractable.
The landscape is built and its lazy caches warmed in ``setup`` (excluded from the
timing) so each ``time_*`` measures only that metric's own work.
"""

from graphfla import analysis as A

from . import _datasets

SEED = 0


class Analysis:
    # Moderate real landscapes: CR6261 (~1.9k boolean), TrpB3I (~7.8k protein).
    params = ["CR6261", "TrpB3I"]
    param_names = ["dataset"]

    def setup(self, dataset):
        cls, X, f = _datasets.load_real(dataset)  # skipped if data absent
        landscape = cls(maximize=True)
        landscape.build_from_data(X, f, verbose=False)
        landscape.basins
        landscape.dist_to_go
        landscape.neighbor_fitness
        landscape.accessible_paths
        self.ls = landscape

    # --- ruggedness / structure ---
    def time_local_optima_ratio(self, dataset):
        A.local_optima_ratio(self.ls)

    def time_gradient_intensity(self, dataset):
        A.gradient_intensity(self.ls)

    def time_autocorrelation(self, dataset):
        A.autocorrelation(self.ls, seed=SEED)

    def time_r_s_ratio(self, dataset):
        A.r_s_ratio(self.ls)

    def time_neutrality(self, dataset):
        A.neutrality(self.ls)

    # --- correlations ---
    def time_fdc(self, dataset):
        A.fdc(self.ls)

    def time_basin_fitness_correlation(self, dataset):
        A.basin_fitness_correlation(self.ls)

    def time_neighbor_fitness_correlation(self, dataset):
        A.neighbor_fitness_correlation(self.ls)

    def time_fitness_flattening_index(self, dataset):
        A.fitness_flattening_index(self.ls)

    # --- navigability ---
    def time_global_optima_accessibility(self, dataset):
        A.global_optima_accessibility(self.ls)

    def time_mean_path_length_to_global_optimum(self, dataset):
        A.mean_path_length_to_global_optimum(self.ls)

    def time_mean_distance_to_global_optimum(self, dataset):
        A.mean_distance_to_global_optimum(self.ls)

    # --- robustness ---
    def time_evolvability_enhancing_mutations(self, dataset):
        A.evolvability_enhancing_mutations(self.ls)

    def time_all_mutation_effects(self, dataset):
        A.all_mutation_effects(self.ls, n_jobs=1)

    # --- epistasis ---
    def time_gamma(self, dataset):
        A.gamma(self.ls, n_jobs=1)

    def time_higher_order_epistasis(self, dataset):
        A.higher_order_epistasis(self.ls, order=2)

    def time_walsh_hadamard(self, dataset):
        A.walsh_hadamard(self.ls, max_order=2)

    def time_classify_epistasis(self, dataset):
        A.classify_epistasis(self.ls, sample_cut_prob=0.5, seed=SEED)

    def time_extradimensional_bypass(self, dataset):
        A.extradimensional_bypass(self.ls, sample_cut_prob=0.5, seed=SEED)

    def time_global_idiosyncratic_index(self, dataset):
        A.global_idiosyncratic_index(self.ls, n_jobs=1)

    def time_diminishing_returns_index(self, dataset):
        A.diminishing_returns_index(self.ls)

    def time_increasing_costs_index(self, dataset):
        A.increasing_costs_index(self.ls)
