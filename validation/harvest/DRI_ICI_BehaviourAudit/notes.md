# Diminishing returns / increasing costs: numerical behaviour audit

## Scope and source reuse

This is the empirical Part B audit. The requested papers/_dri_ici/ directory was absent, so there were no prior files there to reuse. I reused the existing papers/Papkou2023/ paper PDF, author notebook, and R helper without copying or changing them. I downloaded no source; therefore no _codex source file or acquisition_codex.jsonl entry was created.

Bibliography was verified by title and DOI against the PubMed citation record. The verified entry is “A rugged yet easily navigable fitness landscape,” DOI “10.1126/science.adh3860,” by “Andrei Papkou, Lucia Garcia-Pastor, José Antonio Escudero, Andreas Wagner” (PubMed citation block, [PubMed](https://pubmed.ncbi.nlm.nih.gov/37995212/)). The supplied selected_references.bib entry instead says “Andrei Papkou and Aviv Regev and Joanna Masel” (lines 12–19); that author list is wrong.

## Verbatim paper and code evidence

Papkou et al. describe diminishing returns in the main text as:

> “a reduction in fitness gains from beneficial mutations as evolving populations approach the maximum fitness (fig. S22).”
> — Discussion, printed p. 5; existing main-paper PDF in papers/Papkou2023/sources/Papkou2023_main_umich.pdf.

The article defines one-mutant neighbors as:

> “Any two DHFR variants that differ at one nucleotide position are immediate (one-mutant) neighbors connected by an edge”
> — Results, “The fitness landscape is rugged,” printed p. 2.

The two printed graph counts are supported by these verbatim snippets:

> “Fifty-two percent of DHFR variants (135,178)”
> — Results, “The fitness landscape is rugged,” printed p. 2.

> “with as many functional-to-functional edges (50%, 161,015/324,044) as nonfunctional-to-functional edges.”
> — Results, “The fitness landscape is rugged,” printed p. 2.

> “In general, we only considered fitness-increasing edges accessible to Darwinian evolution.”
> — Materials and methods, printed p. 6.

GraphFLA DRI’s docstring says:

> “the average fitness improvement provided by its direct successors (fitter one-mutant neighbors). A significant negative correlation indicates diminishing returns.”
> — graphfla/analysis/epistasis/idiosyncrasy.py:268–270.

The implementation comment confirms the per-node aggregation:

> “# Mean improvement toward the optimum across each node's improving out-edges.”
> — graphfla/analysis/epistasis/idiosyncrasy.py:306.

ICI’s docstring says:

> “A significant positive correlation indicates increasing cost.”
> — graphfla/analysis/epistasis/idiosyncrasy.py:402–408.

Its implementation comment says:

> “# Mirror of diminishing_returns_index over IN-edges: mean cost across each node's improving predecessors.”
> — graphfla/analysis/epistasis/idiosyncrasy.py:444–448.

Both functions assign node_fitnesses = fitness (DRI lines 312–313; ICI lines 449–450), then correlate raw node fitness against per-node means. The default method is Pearson; method=regression returns a slope over the same node-level means. The correlation code discards the p-value (correlation, _ = corr_func(...) at DRI line 389), so the functions return no significance test. These implementation claims are verbatim-source-based.

## Papkou input handling and numerical target boundary

The existing author R helper states:

> get_threshold <- function() {
>     return(-0.507774)
> }
> — papers/Papkou2023/sources/zenodo_8228920/lib/import_data.R:18–20.

The author graph notebook floors endpoint fitness below the threshold and drops zero-difference pairs (cell 4):

> ## set fitness below threshold to threshold
> distDF_fitness$Fitness_to [distDF_fitness$Fitness_to < threshold] = threshold
> distDF_fitness$Fitness_from [distDF_fitness$Fitness_from < threshold] = threshold
> ...
> ## remove pairs with threshold fitness in both variants
> distDF_fitness_0 = distDF_fitness %>%
>         mutate(Fitness_diff = Fitness_to - Fitness_from) %>%
>         filter(Fitness_diff!=0)
> — Existing author repository notebook 00.make_graph_object/01.script.ipynb, cell 4.

The exact floor -0.507774 is in author code; exact threshold: NOT STATED IN PAPER. Both the raw and author-floor-clipped calculations are retained; the floor was not selected by searching for a DRI match. The author notebook’s saved aligned-edge output is “324321” (cell 5), while the article prints “324,044” and the local floor-clipped reconstruction gives 324,044. This unresolved source-state difference is recorded rather than hidden.

Fig. S22 fitted values are plotted-only. Exact slope/correlation values: NOT STATED IN PAPER. No plotted point or fitted coefficient was read as a target. The paper’s printed n_edges=324,044 and n_configs=135,178 were separate printed targets and were reproduced exactly under the author-code floor preprocessing.

## Independent reconstruction

The requested field-standard comparison was implemented independently as edge-level regressions: one observation per observed one-site mutation effect, pooled across mutation types and grouped by position/directed allele change for per-mutation slopes. Beneficial response is positive gain against the lower-fitness background. Reverse deleterious cost is the same positive magnitude against the higher-fitness endpoint. GraphFLA instead averages improving edges per genotype, then correlates that node mean with fitness. This distinction explains differences in weighting.

The computed values below are verbatim excerpts from record.json. Each locator points to the corresponding machine-readable audit object. These are calculations on the local inputs, not numbers stated by the paper.

### Synthetic landscapes

#### strictly_additive

A priori expectation: Benefit slope 0 and reverse cost-magnitude slope 0 for every mutation; Pearson is undefined because each response is constant.

> “pooled_beneficial_slope=0.0; pooled_beneficial_pearson=NaN; mean_per_mutation_beneficial_slope=0.0; median_per_mutation_beneficial_slope=0.0; pooled_reverse_cost_slope=0.0; pooled_reverse_cost_pearson=NaN; GraphFLA_DRI_pearson=NaN; GraphFLA_DRI_regression=-6.593557537388059e-17; GraphFLA_ICI_pearson=NaN; GraphFLA_ICI_regression=-6.181460191301306e-17”
> — record.json JSONPath record.json#/behavior_audit/synthetic_cases/strictly_additive.

#### explicit_diminishing_returns

A priori expectation: Beneficial gains decline with background fitness (negative slope/correlation). Reverse deleterious cost magnitudes also decline with higher-fitness source (negative ICI; not increasing costs).

> “pooled_beneficial_slope=-0.3061464179533985; pooled_beneficial_pearson=-0.983582779138306; mean_per_mutation_beneficial_slope=-0.30614641795340125; median_per_mutation_beneficial_slope=-0.30614641795340125; pooled_reverse_cost_slope=-0.43184315699848147; pooled_reverse_cost_pearson=-0.9658150516848337; GraphFLA_DRI_pearson=-0.9812529237559038; GraphFLA_DRI_regression=-0.28215859389608455; GraphFLA_ICI_pearson=-0.9725997186935738; GraphFLA_ICI_regression=-0.49315453482272475”
> — record.json JSONPath record.json#/behavior_audit/synthetic_cases/explicit_diminishing_returns.

#### explicit_increasing_costs

A priori expectation: Beneficial gains rise with source fitness (positive DRI, hence no diminishing returns); reverse deleterious cost magnitudes rise with higher-fitness source (positive ICI).

> “pooled_beneficial_slope=0.45714285714285713; pooled_beneficial_pearson=0.9561828874675149; mean_per_mutation_beneficial_slope=0.45714285714285713; median_per_mutation_beneficial_slope=0.45714285714285713; pooled_reverse_cost_slope=0.32; pooled_reverse_cost_pearson=0.9797958971132712; GraphFLA_DRI_pearson=0.9672388203287415; GraphFLA_DRI_regression=0.4054054054054054; GraphFLA_ICI_pearson=0.9756156783416059; GraphFLA_ICI_regression=0.35353535353535354”
> — record.json JSONPath record.json#/behavior_audit/synthetic_cases/explicit_increasing_costs.

#### pure_house_of_cards_fixed_seed

A priori expectation: The construction has no shared g(sum(x)) rule. Zero structural trend is the intended null, but raw delta-vs-parent-fitness regression is not expected to be zero: delta contains minus parent fitness, and benefit-only analysis conditions on positive delta. See the 256-landscape sweep.

> “pooled_beneficial_slope=-0.5644369872368947; pooled_beneficial_pearson=-0.5282189352168338; mean_per_mutation_beneficial_slope=-0.5443537680287915; median_per_mutation_beneficial_slope=-0.5708499587132403; pooled_reverse_cost_slope=0.5700097047862809; pooled_reverse_cost_pearson=0.536879621847493; GraphFLA_DRI_pearson=-0.8662471449709506; GraphFLA_DRI_regression=-0.5084612894819515; GraphFLA_ICI_pearson=0.7899604715030412; GraphFLA_ICI_regression=0.5796299490966677”
> — record.json JSONPath record.json#/behavior_audit/synthetic_cases/pure_house_of_cards_fixed_seed.

#### beneficial_trend_only_reverse_costs_flat

A priori expectation: Beneficial gain slope is negative; reverse cost magnitude is exactly uncorrelated with higher-fitness background (slope and Pearson 0).

> “pooled_beneficial_slope=-0.5555555555555556; pooled_beneficial_pearson=-0.7453559924999299; mean_per_mutation_beneficial_slope=-0.5555555555555556; median_per_mutation_beneficial_slope=-0.5555555555555556; pooled_reverse_cost_slope=0.0; pooled_reverse_cost_pearson=0.0; GraphFLA_DRI_pearson=-0.7453559924999299; GraphFLA_DRI_regression=-0.5555555555555555; GraphFLA_ICI_pearson=0.0; GraphFLA_ICI_regression=3.885780586188049e-16”
> — record.json JSONPath record.json#/behavior_audit/synthetic_cases/beneficial_trend_only_reverse_costs_flat.

House-of-Cards needs a caveat. The 32 genotype fitness values are independent by construction, but raw mutation gain is fitness(mutant) minus fitness(parent), so the parent value appears in both the predictor and response. Beneficial-only analysis additionally conditions on positive gain. The 256-landscape sweep therefore shows systematic negative gain slopes and negative DRI even without a shared global-epistasis rule. That is an inference from the construction and the sweep, not a paper result.

> “all_signed_slope mean=-1.0255998070812087; all_signed_corr mean=-0.7151217715148849; benefit_edge_slope mean=-0.5230639180709886; benefit_edge_corr mean=-0.505031797274599; dri mean=-0.7059353696442531; ici mean=0.7042065979393379”
> — record.json JSONPath #/behavior_audit/house_of_cards_ensemble/summary; 256 five-locus iid Normal landscapes, DRI negative in 256/256 and ICI positive in 255/256.

### Repository landscapes

Each quoted result gives pooled beneficial edge slope/Pearson r, the per-mutation mean and median slope, pooled reverse-cost slope/r, then GraphFLA DRI Pearson/regression and ICI Pearson/regression. GraphFLA agrees with an independent reconstruction of the node means to floating-point precision. Differences from edge pooling arise because an edge-pool gives extra weight to genotypes with more beneficial neighbors, while GraphFLA contributes one mean per genotype.

#### Papkou2023_DHFR

> “pooled_beneficial_slope=0.070009252095349; pooled_beneficial_pearson=0.07604039429524366; mean_per_mutation_beneficial_slope=0.08667612453367196; median_per_mutation_beneficial_slope=0.08372335217825017; pooled_reverse_cost_slope=0.4616791598337813; pooled_reverse_cost_pearson=0.7069722429393024; GraphFLA_DRI_pearson=0.06480988633081898; GraphFLA_DRI_regression=0.03216503608187424; GraphFLA_ICI_pearson=0.97133369840374; GraphFLA_ICI_regression=0.4690210344391862”
> — record.json JSONPath record.json#/behavior_audit/real_landscape_comparisons/Papkou2023_DHFR.

#### Papkou2023_DHFR_author_floor_clipped

> “pooled_beneficial_slope=-0.31403398776700997; pooled_beneficial_pearson=-0.32902899010485287; mean_per_mutation_beneficial_slope=0.07449802758838332; median_per_mutation_beneficial_slope=-0.019579525601914907; pooled_reverse_cost_slope=0.46528463070615134; pooled_reverse_cost_pearson=0.5521615778243071; GraphFLA_DRI_pearson=-0.2578492271366022; GraphFLA_DRI_regression=-0.40650132525027216; GraphFLA_ICI_pearson=0.9459258529461695; GraphFLA_ICI_regression=0.4760974581378907”
> — record.json JSONPath record.json#/behavior_audit/real_landscape_comparisons/Papkou2023_DHFR_author_floor_clipped.

#### Bendixsen2019_hdv

> “pooled_beneficial_slope=0.1826463916726223; pooled_beneficial_pearson=0.08554058330048839; mean_per_mutation_beneficial_slope=-0.04318724862004364; median_per_mutation_beneficial_slope=0.03463074167345044; pooled_reverse_cost_slope=0.8003763015589137; pooled_reverse_cost_pearson=0.9123820625130835; GraphFLA_DRI_pearson=0.19171038315656858; GraphFLA_DRI_regression=0.25306674273100344; GraphFLA_ICI_pearson=0.9872228869751002; GraphFLA_ICI_regression=0.8022571372242667”
> — record.json JSONPath record.json#/behavior_audit/real_landscape_comparisons/Bendixsen2019_hdv.

#### Domingo2018

> “pooled_beneficial_slope=-0.3532356851138551; pooled_beneficial_pearson=-0.5423329229224351; mean_per_mutation_beneficial_slope=-0.32543084313931425; median_per_mutation_beneficial_slope=-0.32820968256785443; pooled_reverse_cost_slope=0.0989060082220452; pooled_reverse_cost_pearson=0.12865066954609625; GraphFLA_DRI_pearson=-0.7565430915517805; GraphFLA_DRI_regression=-0.32694487875357037; GraphFLA_ICI_pearson=0.26614110424559495; GraphFLA_ICI_regression=0.09620180853555874”
> — record.json JSONPath record.json#/behavior_audit/real_landscape_comparisons/Domingo2018.

#### Lite2020_ParD3

> “pooled_beneficial_slope=-0.13059950646110471; pooled_beneficial_pearson=-0.17885719096714608; mean_per_mutation_beneficial_slope=-0.03978229153806597; median_per_mutation_beneficial_slope=-0.05333361680530598; pooled_reverse_cost_slope=0.3164964610139238; pooled_reverse_cost_pearson=0.4888473728668145; GraphFLA_DRI_pearson=-0.5598937702717106; GraphFLA_DRI_regression=-0.15735061155130375; GraphFLA_ICI_pearson=0.8899320982783874; GraphFLA_ICI_regression=0.3356687303945935”
> — record.json JSONPath record.json#/behavior_audit/real_landscape_comparisons/Lite2020_ParD3.

For the Papkou floor-clipped row, the independent reconstruction has 324,044 beneficial edges and GraphFLA has 324,044 graph edges, matching the printed article count. The raw no-floor run remains a separate variant and has 1,490,486 improving edges plus a slightly positive DRI relationship. The clipped pooled gain regression is negative while its unweighted mean per-mutation slope is positive, so aggregation matters.

Bendixsen’s DRI Pearson result is positive while its Spearman result is negative; linear and rank association point in opposite directions on that input. I did not infer a figure value or select preprocessing to favor either statistic.

## Robustness probes

### Reference genotype

Subtracting fitness at genotype index 0 or 31 is a constant offset. Pearson correlations and OLS slopes remain unchanged to numerical precision; neither function accepts a reference-genotype argument.

> “reference fitness 0.0: DRI Pearson=-0.9812529237559038, ICI Pearson=-0.9725997186935738; reference fitness 1.791759469228055: DRI Pearson=-0.9812529237559037, ICI Pearson=-0.9725997186935736”
> — record.json JSONPath #/behavior_audit/robustness/reference_offsets.

### Maximize flag

Encoding the same ordering as maximize f or minimize -f reverses reported signs. The implementation correlates raw node fitness rather than an orientation-normalized fitness utility.

> “maximize DRI Pearson=-0.9812529237559038, ICI Pearson=-0.9725997186935738; minimize on -f DRI Pearson=0.9812529237559038, ICI Pearson=0.9725997186935738”
> — record.json JSONPath #/behavior_audit/robustness/maximize_true_fitness_vs_minimize_negative_fitness.

### Monotone rescaling

The tests apply strictly increasing transformations to the same concave surface f=log(1+k). Every tested transform and coefficient is retained in record.json. Both indices can change sign while genotype ranking is unchanged.

- identity: DRI Pearson -0.9812529237559038, regression -0.28215859389608455; ICI Pearson -0.9725997186935738, regression -0.49315453482272475. Locator: record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness.
- square_monotone_on_nonnegative: DRI Pearson -0.3617951220014535, regression -0.025878855071060334; ICI Pearson 0.4577075189225712, regression 0.05479124293916651. Locator: record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness.
- exp_monotone: DRI Pearson 0.027064987134842955, regression -4.665252974567023e-17; ICI Pearson -0.5037643095584788, regression -2.22755322209056e-16. Locator: record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness.
- exp_5x: DRI Pearson 0.9911287580307535, regression 1.400448502365419; ICI Pearson 0.9972303439796026, regression 0.6154081626899196. Locator: record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness.

> “identity DRI=-0.9812529237559038 / ICI=-0.9725997186935738; exp_5x DRI=0.9911287580307535 / ICI=0.9972303439796026”
> — record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness.

### Beneficial mutation selection

DRI counts improving outgoing edges only. ICI uses those same improving edges at their fitter endpoints as reverse-cost magnitudes; it does not independently enumerate deleterious mutation effects. The four-edge synthetic control had a negative beneficial-gain relationship and exactly zero reverse-cost relationship; GraphFLA returned DRI Pearson -0.7453559924999299 and ICI 0.0 (see the synthetic case locator in record.json).

## Verdict and limitations

**Overall verdict: wrong as a general, representation-robust measure.** For maximize=True on a fixed raw fitness scale, the implementation accurately calculates its documented signed node-average correlation. But more negative DRI means stronger diminishing returns; a larger numeric value means less diminishing returns (or more increasing returns). Positive ICI means increasing costs. The names invite a magnitude interpretation; the implementations are not the field’s pooled/per-mutation analysis; both can change sign under nonlinear monotone fitness rescaling; and both signs reverse under an equivalent minimize encoding. The House-of-Cards control also shows nonzero trends from selection and algebraic coupling.

The exact Fig. S22 source edge rows and fitted coefficients remain blocked by the missing author distDF_fitness.rds input and were intentionally not pursued because a sibling is reproducing that figure. No GraphFLA repository files were modified.

## Additional audit quotes and locators

The independent analysis protocol is recorded verbatim as:

> “For each observed one-site pair, independently create one beneficial record oriented from lower to higher fitness, and regress positive delta on lower-endpoint fitness. For reverse deleterious costs, use the same positive magnitude against higher-endpoint fitness. Pool mutation rows and separately group by position and directed allele change for per-mutation slopes. Pooled analysis weights each mutation row; GraphFLA node means weight each genotype once.”
> — record.json JSONPath #/behavior_audit/independent_analysis_method_quote.

The synthetic a priori expectations are retained verbatim:

> “Benefit slope 0 and reverse cost-magnitude slope 0 for every mutation; Pearson is undefined because each response is constant.”
> — record.json JSONPath #/behavior_audit/synthetic_cases/strictly_additive/expected_under_standard_analysis.

> “Beneficial gains decline with background fitness (negative slope/correlation). Reverse deleterious cost magnitudes also decline with higher-fitness source (negative ICI; not increasing costs).”
> — record.json JSONPath #/behavior_audit/synthetic_cases/explicit_diminishing_returns/expected_under_standard_analysis.

> “Beneficial gains rise with source fitness (positive DRI, hence no diminishing returns); reverse deleterious cost magnitudes rise with higher-fitness source (positive ICI).”
> — record.json JSONPath #/behavior_audit/synthetic_cases/explicit_increasing_costs/expected_under_standard_analysis.

> “The construction has no shared g(sum(x)) rule. Zero structural trend is the intended null, but raw delta-vs-parent-fitness regression is not expected to be zero: delta contains minus parent fitness, and benefit-only analysis conditions on positive delta. See the 256-landscape sweep.”
> — record.json JSONPath #/behavior_audit/synthetic_cases/pure_house_of_cards_fixed_seed/expected_under_standard_analysis.

> “Beneficial gain slope is negative; reverse cost magnitude is exactly uncorrelated with higher-fitness background (slope and Pearson 0).”
> — record.json JSONPath #/behavior_audit/synthetic_cases/beneficial_trend_only_reverse_costs_flat/expected_under_standard_analysis.

The House-of-Cards control uses 32 genotypes and a fixed seed, and the separate sweep covers 256 further landscapes:

> “Independent N(0,1) fitness assigned to each of 32 genotypes; numpy default_rng seed 20261001.”
> — record.json JSONPath #/behavior_audit/synthetic_cases/pure_house_of_cards_fixed_seed/fitness_rule.

> “all_signed_slope_mean=-1.0255998070812087; all_signed_corr_mean=-0.7151217715148849; benefit_edge_slope_mean=-0.5230639180709886; benefit_edge_corr_mean=-0.505031797274599; dri_mean=-0.7059353696442531; dri_negative=256/256; ici_mean=0.7042065979393379; ici_positive=255/256”
> — record.json JSONPath #/behavior_audit/house_of_cards_ensemble/result_quote.

The Papkou raw and floor-clipped edge counts and results are retained separately:

> “pooled_beneficial_slope=0.070009252095349; pooled_beneficial_pearson=0.07604039429524366; mean_per_mutation_beneficial_slope=0.08667612453367196; median_per_mutation_beneficial_slope=0.08372335217825017; pooled_reverse_cost_slope=0.4616791598337813; pooled_reverse_cost_pearson=0.7069722429393024; GraphFLA_DRI_pearson=0.06480988633081898; GraphFLA_DRI_regression=0.03216503608187424; GraphFLA_ICI_pearson=0.97133369840374; GraphFLA_ICI_regression=0.4690210344391862; beneficial_edges=1490486; GraphFLA_graph_edges=1490486”
> — record.json JSONPath #/behavior_audit/real_landscape_comparisons/Papkou2023_DHFR/result_quote.

> “pooled_beneficial_slope=-0.31403398776700997; pooled_beneficial_pearson=-0.32902899010485287; mean_per_mutation_beneficial_slope=0.07449802758838332; median_per_mutation_beneficial_slope=-0.019579525601914907; pooled_reverse_cost_slope=0.46528463070615134; pooled_reverse_cost_pearson=0.5521615778243071; GraphFLA_DRI_pearson=-0.2578492271366022; GraphFLA_DRI_regression=-0.40650132525027216; GraphFLA_ICI_pearson=0.9459258529461695; GraphFLA_ICI_regression=0.4760974581378907; beneficial_edges=324044; GraphFLA_graph_edges=324044”
> — record.json JSONPath #/behavior_audit/real_landscape_comparisons/Papkou2023_DHFR_author_floor_clipped/result_quote.

Bendixsen’s linear and rank correlations point in opposite directions:

> “GraphFLA_DRI_pearson=0.19171038315656858; GraphFLA_DRI_spearman=-0.5537058442614368”
> — record.json JSONPath #/behavior_audit/real_landscape_comparisons/Bendixsen2019_hdv/result_quote.

Both function signatures omit a reference-genotype parameter:

> “def diminishing_returns_index(
>     landscape,
>     method: Literal["pearson", "spearman", "regression"] = "pearson",
> ) -> float:”
> — GraphFLA graphfla/analysis/epistasis/idiosyncrasy.py:259–262.

> “def increasing_costs_index(
>     landscape,
>     method: Literal["pearson", "spearman", "regression"] = "pearson",
> ) -> float:”
> — GraphFLA graphfla/analysis/epistasis/idiosyncrasy.py:396–399.

The four monotone-scale outputs are quoted from the stored records:

> “transform=identity; DRI_pearson=-0.9812529237559038; DRI_regression=-0.28215859389608455; ICI_pearson=-0.9725997186935738; ICI_regression=-0.49315453482272475”
> — record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness/identity.

> “transform=square_monotone_on_nonnegative; DRI_pearson=-0.3617951220014535; DRI_regression=-0.025878855071060334; ICI_pearson=0.4577075189225712; ICI_regression=0.05479124293916651”
> — record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness/square_monotone_on_nonnegative.

> “transform=exp_monotone; DRI_pearson=0.027064987134842955; DRI_regression=-4.665252974567023e-17; ICI_pearson=-0.5037643095584788; ICI_regression=-2.22755322209056e-16”
> — record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness/exp_monotone.

> “transform=exp_5x; DRI_pearson=0.9911287580307535; DRI_regression=1.400448502365419; ICI_pearson=0.9972303439796026; ICI_regression=0.6154081626899196”
> — record.json JSONPath #/behavior_audit/robustness/monotone_rescaling_of_concave_fitness/exp_5x.

The split beneficial/deleterious synthetic control directly checks mutation selection:

> “pooled_beneficial_slope=-0.5555555555555556; pooled_beneficial_pearson=-0.7453559924999299; mean_per_mutation_beneficial_slope=-0.5555555555555556; median_per_mutation_beneficial_slope=-0.5555555555555556; pooled_reverse_cost_slope=0.0; pooled_reverse_cost_pearson=0.0; GraphFLA_DRI_pearson=-0.7453559924999299; GraphFLA_DRI_regression=-0.5555555555555555; GraphFLA_ICI_pearson=0.0; GraphFLA_ICI_regression=3.885780586188049e-16”
> — record.json JSONPath #/behavior_audit/synthetic_cases/beneficial_trend_only_reverse_costs_flat/result_quote.

The implementation uses raw node fitness as its predictor and returns the coefficient only:

> “node_fitnesses = fitness”
> — GraphFLA graphfla/analysis/epistasis/idiosyncrasy.py:312–313 and 449–450.

> “correlation, _ = corr_func(node_fitnesses, avg_improvement)”
> — GraphFLA graphfla/analysis/epistasis/idiosyncrasy.py:388–390; ICI equivalent at 512–514.
