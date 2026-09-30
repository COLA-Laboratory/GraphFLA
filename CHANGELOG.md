# Changelog

All notable changes to GraphFLA are recorded here. Entries that change a
returned statistic are marked, because downstream analyses depend on them.

## Unreleased

### Changed — affects returned values

- `single_mutation_effects` and `all_mutation_effects` now report `mean_effect`
  as `mutation_to` minus `mutation_from`, matching the row's own label and the
  convention already used by `fitness_effect_distribution`. Up to and including
  0.3.0 the sign was reversed, and `test_type="positive"` correspondingly tested
  for a fitness *decrease*. Results computed with an earlier release should be
  re-derived, or the sign of `mean_effect` flipped and `test_type` swapped.
  `median_abs_effect` is unaffected.
- `r_s_ratio`'s degeneracy test is now invariant under an affine rescaling of
  fitness, `f -> a*f + b`, as its definition implies: roughness and slope both
  scale by `|a|` and the intercept absorbs `b`. It previously used an absolute
  tolerance and reported `inf` for fitness in very small units. Note that the
  residual root-mean-square is still computed by squaring, so the *returned
  value* remains subject to float64 overflow beyond roughly 1e150 and underflow
  below roughly 1e-150. That arithmetic limit is unchanged by this release.
- `r_s_ratio` returns NaN rather than `inf` for constant fitness, where
  roughness and slope are both zero and the ratio is undefined. A purely
  epistatic landscape, with positive roughness and zero slope, still gives
  `inf`. Constant fitness is detected by exact equality, so values whose
  standard deviation is a non-zero round-off (such as a repeated 0.1) are
  recognised. The degeneracy test compares the slope against the fitness
  *range*, so it stays correct at scales where squaring the deviations would
  overflow or underflow.

### Fixed

- `fitness_effect_distribution` no longer raises on a single-position landscape,
  where every genotype shares one trivial genetic background.
- Landscape construction no longer rejects a fully neutral landscape as
  edgeless. Such a landscape has neighbours but no directed (improving) edges;
  construction now fails only when neither directed edges nor neutral pairs are
  found.
- `diminishing_returns_index` and `increasing_costs_index` return NaN with a
  warning on an edgeless landscape instead of raising from the sparse-adjacency
  fallback.
- Largest-component filtering (`tau` with `filter_mode`) now counts neutral
  pairs toward connectivity. Because a neutral pair carries no directed edge, a
  plateau previously looked like a set of singleton components, and filtering a
  neutral landscape could silently reduce it to a single genotype. The
  functional filter severs neutral pairs on the same rule it applies to directed
  edges, so a below-threshold plateau can no longer win component selection over
  the functional region.

### Documentation

- The `gamma` and `gamma_star` return-value descriptions contradicted their own
  Notes and Ferretti et al. (2016): an additive landscape gives `gamma = 1` and
  a House-of-Cards landscape gives `gamma` near 0, not the reverse.

## 0.3.0

- See the repository history for releases up to 0.3.0.
