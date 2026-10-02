# Metric-provenance findings — 30 September 2026 (second tranche)

Three independent definition audits were run against the papers GraphFLA cites
for its weakest-evidenced metrics. Each claim below was re-verified by the lead
against the GraphFLA source or by direct arithmetic; agent reports alone were
not accepted.

## 1. `evolvability_enhancing_mutations` does not implement Wagner's definition

**Citation is to a non-existent paper.** The docstring cites
`10.1038/s41576-023-00559-0` ("The role of evolvability in the evolution of
complex traits", Nat Rev Genet 24, 1-16). Verified by the lead: that DOI returns
HTTP 404 and Crossref reports "Resource not found"; no Nature Reviews Genetics
article of that title is registered. The intended source is Wagner, A.,
"Evolvability-enhancing mutations in the fitness landscapes of an RNA and a
protein", Nat. Commun. 14, 3624 (2023), `10.1038/s41467-023-39321-8`, confirmed
by the lead via Crossref.

**The implemented inequality is the one Wagner explicitly rejects.** Wagner's SI
eq. (S2): a mutation is evolvability-enhancing if
`mean(w, n_m) - mean(w, n_wt) > max(0, dw)`. For a beneficial mutation the paper
requires strictly more than the mutation's own benefit, because
`... > 0` would be, in the paper's words, "trivially met" under additivity.
GraphFLA tests `delta_mean_neighbor_fit > epsilon` on improving edges, i.e. the
`> 0` form.

Lead verification on an additive OneMax(4) landscape:
`delta_mean_neighbor_fit` is 0.5 on every edge and `dw` is 1.0 on every edge.
GraphFLA counts 32 of 32 edges as EE and returns 1.0; Wagner's criterion counts
0 of 32. The existing test `test_evol_enhance_additive_is_one` pins the incorrect
value. (Under Wagner's own position-exclusion the additive identity is exact,
`mean(w,n_m) - mean(w,n_wt) == dw`, so additive landscapes give exactly 0.)

Three further divergences reported by the audit, not yet independently verified
by the lead:
- Neighbourhood membership: Wagner excludes every allele at the mutated
  position from both neighbourhoods; GraphFLA excludes nothing. Wagner's two
  phrasings ("exclude m itself" and "exclude the mutated position") coincide for
  a biallelic landscape, but neither coincides with GraphFLA, which excludes
  nothing at any arity. The additive OneMax(4) figures above show the gap on a
  binary landscape: 0.5 under GraphFLA's neighbourhood against the exact dw = 1
  that Wagner's position-exclusion yields. Correcting only the inequality would
  therefore leave this divergence in place.
- Significance: Wagner requires a one-sample t test with Benjamini-Hochberg
  FDR 0.01. GraphFLA's `epsilon` has no statistical semantics, so its output is
  not comparable to any fraction printed in the paper.
- Denominator: Wagner counts every ordered neighbour pair (both orientations,
  including deleterious and tied); GraphFLA divides by the improving-edge count.

Reported targets, for any future reproduction: protein (ParD3, Lite 2020)
beneficial EE 0.39% (681) and deleterious EE 0.13% (221); RNA (tRNA, Domingo
2018) beneficial EE 5.7% (2983 of 52672) and deleterious EE 7.0% (3702/52672).
Author code and precomputed per-edge outputs:
`https://github.com/andreas-wagner-uzh/EE_mutations`.

## 2. `global_idiosyncratic_index` uses an analytic baseline, not Lyons' control

**2026-10-02 follow-up:** the historical discrepancy described below is now
addressed by matched-size sampling and directed-mutation aggregation. The
published tRNA mean and SEM reproduce with the author's seeds; the public
seeded function matches a separate same-stream oracle. Input-population and
Fig. 1a source differences remain explicit. See
[the four-paper review](IDIOSYNCRASY_REVIEW.md). The following audit is retained
as historical evidence rather than rewritten as if the old behavior never existed.

Lyons et al. 2020 (Nat Ecol Evol, `10.1038/s41559-020-01286-y`) define the
per-mutation index as the SD of a mutation's effects across its backgrounds
divided by the SD of fitness differences of **an equal number of randomly
sampled genotype pairs**, and the landscape index as the unweighted mean of the
per-mutation values. The landscape-level aggregate IS defined by the paper, so
`global_idiosyncratic_index` is legitimately citable to it — but only under that
construction.

Lead verification in source: `graphfla/analysis/epistasis/idiosyncrasy.py:186`
and `:237` use `np.sqrt(2.0) * np.std(all_fitness_values)`. That is the
large-sample limit of the paper's control, not the paper's procedure: Lyons
draws exactly `n` pairs matched to that mutation's background count, so the two
denominators differ materially for sparsely-sampled mutations. GraphFLA's
`min_pairs=3` has no counterpart in the paper's method. The `np.nanmean`
aggregation does match.

Reported target: yeast tRNA landscape (Li et al. 2016) I_id = 0.612 +/- 0.005
over 828 directed single mutations. Data and the authors' notebook:
`https://github.com/lyonsdm/idiosyncrasy`.

`diminishing_returns_index` and `increasing_costs_index` have no basis in Lyons
2020 — the paper reports distributions of per-mutation Pearson correlations and
defines no scalar index. Citing Lyons for those scalars would be wrong.

## 3. Ferretti gamma / r-s: definitions confirmed, one target partially reproduced

Appendix A specifies regressors `A_i` in {0,1}, not +/-1. GraphFLA uses 0/1 and
therefore follows the paper; the factor-two gap against Skwara 2023 is a
coordinate-convention difference, not a defect.

Eq. (1)'s denominator sums over all loci with an `(L-1)` factor, which is
algebraically identical to summing over the ordered (background, focal) pairs.
Lead verification: the independent oracle `gamma_reference` in
`tests/test_analysis_oracles.py` sums over exactly those triples, so GraphFLA
carries no denominator slip.

Figure 4(c) prints values for two landscapes. Lead ran GraphFLA on the repo's
existing 32-genotype beta-lactamase CSV
(`data/BioSequence/Weinreich2006Tan2011_Weinreich2006.csv`):

| statistic | GraphFLA | Ferretti Fig 4c |
| --- | --- | --- |
| peaks | 1 | 1 |
| sinks | 1 | 1 |
| gamma | 0.8254 | 0.85 |
| gamma* | 0.6136 | 0.59 |
| r/s | 0.4455 | 0.43 |

Structure matches exactly; every statistic is 2-4% off. Since gamma is anchored
to an independent implementation of eq. (1), the likely cause is that the repo
CSV is not the fitness vector Ferretti used. Recovering the original Weinreich
2006 input and its log transform is the open action.

## Status

None of these findings has been acted on in production code. Items 1 and 2
change what a published metric means and require a scope decision before any
edit.
