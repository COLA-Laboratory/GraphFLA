# Diminishing returns / increasing costs: literature survey

Study key: `DRI_ICI_LiteratureSurvey`  
Dossier: `papers/_dri_ici`  
Survey date: 2026-10-01  
Scope: literature-method survey only. No reproduction, input rebuilding, or GraphFLA metric run was attempted.

## Verdict

The maintainer’s expectation is broadly right: the cited literature usually studies mutation effects against the fitness of their backgrounds and reports a per-mutation trend, a distribution of per-mutation Pearson coefficients or slopes, or a descriptive trend. It does not converge on one field-wide landscape-level “diminishing returns index” or “increasing costs index.” There are exceptions: Wiser et al. fit a model parameter for diminishing returns; Reddy & Desai formalize a slope coefficient per locus; Bakerlee et al. report per-locus fitness-correlated-trend slopes; and Papkou et al. fit a regression to a pool of beneficial one-step transitions. Huang, Zhou & Li (2025) explicitly define pooled, landscape-level Pearson coefficients for beneficial and deleterious transitions.

GraphFLA’s functions are a **defensible but non-standard extension** as a landscape summary. They correlate node fitness with a node-level mean of improving edge effects. That differs from the per-mutation analyses used in most older papers and from Huang et al.’s pooled edge-level Pearson definitions. Calling the current implementation a reproduction of Huang et al.’s scalar would be inaccurate; attributing GraphFLA’s node-average definition to those papers would misrepresent their methods.

## Bibliography verification

I looked up each work by title and DOI and checked metadata against Crossref; publisher or PubMed records were also consulted where available. Supplied author data were not used to find any paper.

| Work | Verified authors and citation | DOI / metadata check |
|---|---|---|
| Chou et al. 2011 | Hsin-Hung Chou, Hsuan-Chao Chiu, Nigel F. Delaney, Daniel Segrè, Christopher J. Marx. *Science* 332(6034):1190–1192. | [10.1126/science.1203799](https://api.crossref.org/works/10.1126/science.1203799) |
| Khan et al. 2011 | Aisha I. Khan, Duy M. Dinh, Dominique Schneider, Richard E. Lenski, Tim F. Cooper. *Science* 332(6034):1193–1196. | [10.1126/science.1203801](https://api.crossref.org/works/10.1126/science.1203801) |
| Kryazhimskiy et al. 2014 | Sergey Kryazhimskiy, Daniel P. Rice, Elizabeth R. Jerison, Michael M. Desai. *Science* 344(6191):1519–1522. | [10.1126/science.1250939](https://api.crossref.org/works/10.1126/science.1250939) |
| Wiser et al. 2013 | Michael J. Wiser, Noah Ribeck, Richard E. Lenski. *Science* 342(6164):1364–1367. | [10.1126/science.1243357](https://api.crossref.org/works/10.1126/science.1243357) |
| Johnson et al. 2019 | Milo S. Johnson, Alena Martsul, Sergey Kryazhimskiy, Michael M. Desai. *Science* 366(6464):490–493. | [10.1126/science.aay4199](https://api.crossref.org/works/10.1126/science.aay4199) |
| Reddy & Desai 2021 | Gautam Reddy, Michael M. Desai. *eLife* 10:e64740. | [10.7554/eLife.64740](https://api.crossref.org/works/10.7554/eLife.64740) |
| Bakerlee et al. 2022 | Christopher W. Bakerlee, Alex N. Nguyen Ba, Yekaterina Shulgina, Jose I. Rojas Echenique, Michael M. Desai. *Science* 376(6593):630–635. | [10.1126/science.abm4774](https://api.crossref.org/works/10.1126/science.abm4774) |
| Papkou et al. 2023 | Andrei Papkou, Lucia Garcia-Pastor, José Antonio Escudero, Andreas Wagner. *Science* 382(6673), article eadh3860. | [10.1126/science.adh3860](https://api.crossref.org/works/10.1126/science.adh3860) |
| Lyons et al. 2020 | Daniel M. Lyons, Zhengting Zou, Haiqing Xu, Jianzhi Zhang. *Nature Ecology & Evolution* 4(12):1685–1693. | [10.1038/s41559-020-01286-y](https://api.crossref.org/works/10.1038/s41559-020-01286-y) |
| Huang, Zhou & Li 2025 | Mingyu Huang, Shasha Zhou, Ke Li. *Advances in Neural Information Processing Systems* 38, pp. 39477–39532. | [10.52202/085713-1180](https://api.crossref.org/works/10.52202/085713-1180) |

The validation workspace’s `corpus_registry.json` contains an incorrect PapkouRM23 `provided_author` field (“Andrei Papkou and Aviv Regev and Joanna Masel”). Crossref for the DOI above gives the four-author list shown in the table. The assignment’s Johnson description is also imprecise: DOI 10.1126/science.aay4199 is titled *Higher-fitness yeast genotypes are less robust to deleterious mutations* and centers on deleterious insertion mutations/increasing costs, not a beneficial-mutation diminishing-returns study.

## Paper-by-paper operationalizations

The locators below refer to printed article pages unless marked SI. Short quoted fragments are verbatim. Numerical values are transcribed from printed text, table cells, or figure annotations; no value was inferred from a plotted point. Full per-paper values and machine-readable locators are also in `record.json`.

### Chou, Chiu, Delaney, Segrè & Marx (2011)

1. **Axes and scale.** The caption says “Variation in relative selective effect (normalized to maximum sᵢ) of each allele as a function of the fitness of the background it was introduced into” (Fig. 1, printed p.1190). For the Methylobacterium panel, “background fitness [is] normalized from EM = 0 to maximum = 1”; the β-lactamase comparison uses “log scale for visualization” (same caption).
2. **Mutations/reference.** “All identified alleles, when present individually in the ancestral background conferred fitness benefits ranging from 10 to 51%”; strains “with each allelic combination (2⁴ = 16) were constructed” (Results, printed p.1191). The article says “each allele was universally beneficial across genetic backgrounds (i.e., showed no sign epistasis)” (same locator). Thus the effect is assessed against the genotype background where the allele is introduced, with ancestral-background benefit used to identify these focal alleles.
3. **Statistic.** The Abstract says “the proportional selective benefit for three of the four loci consistently decreased when they were introduced onto more fit genetic backgrounds” (printed p.1190); the fourth, pntAB, is the stated exception (Results, p.1191). Pearson/Spearman coefficients, regression slopes, and per-allele p values: `NOT STATED IN PAPER`. The plotted effects were not read as numbers.
4. **Index.** NO INDEX DEFINED. The paper plots effect-versus-background relationships for focal alleles and describes them as trends.
5. **Printed numeric coefficient.** `NOT STATED IN PAPER` for a DRI/ICI slope, r, rho, or p value.

### Khan, Dinh, Schneider, Lenski & Cooper (2011)

1. **Axes and scale.** Figure 4 caption: “Relation between the marginal fitness effect of adding a particular mutation and the fitness of the progenitor background to which it was added” (Fig. 4 caption, printed p.1195). Genotype fitness was estimated “in direct competition against a marked variant of the ancestor” (Results, printed p.1194). A log/linear transformation for the marginal effect is `NOT STATED IN PAPER`.
2. **Mutations/reference.** The five trajectory mutations are used: “each mutational step that was followed in the evolving population produced an increase in fitness relative to the immediate progenitor” (Results, printed p.1193). Thus, beneficial status is relative to each mutation’s immediate progenitor, not the shared ancestor. The article also says “The open symbols show the effects of adding each focal mutation to the ancestral strain” (Fig. 4 caption, p.1195); pykF is neutral on that ancestor (Results, p.1195).
3. **Statistic.** Figure 4 caption: “Each panel includes the Pearson correlation coefficient and its significance” (printed p.1195). It is one r and P per focal mutation, not a pooled landscape index. The five printed r/P annotations are: rbs r=−0.256, P=0.339; topA r=−0.586, P=0.017; spoT r=−0.502, P=0.048; glmUS r=−0.499, P=0.049; pykF r=0.652, P=0.006 (Fig. 4, printed p.1195; figure text labels, not point readings).
4. **Index.** NO INDEX DEFINED. A separate analysis asks how epistatic deviation changes with expected genotype fitness: “We observed an overall negative relation” with “absolute epistasis: correlation coefficient (r)=−0.578; relative epistasis: r=−0.586” (Results, printed p.1194). Those responses are epistatic deviation versus expected genotype fitness, not mutation effect versus background fitness.
5. **Printed numeric coefficient.** See the five Fig. 4 r/P annotations in `record.json`, locator Fig. 4, printed p.1195. No single landscape-level DRI/ICI number is defined.

### Kryazhimskiy, Rice, Jerison & Desai (2014)

1. **Axes and scale.** Fig. 3 labels its axes “Fitness effect of knock-out, %” and “Fitness of background strain, %” (printed p.1521); its caption says the effect “declines with the fitness of the background strain” (p.1521). A log transformation is `NOT STATED IN PAPER`.
2. **Mutations/reference.** The paper says “knockouts of these genes are beneficial in our system” and reports “the fitness effects of each knockout in each background” (Results, printed p.1521). The three focal deletions are gat2, whi2 and sfl1; “The ho knockout is a negative control” (Fig. 3 caption, p.1521). The tested background is the comparison genotype for each effect; the paper does not say it filters the measured effects to positive values only.
3. **Statistic.** The Results describe a negative correlation but do not identify Pearson, Spearman, or a regression slope. Coefficient type and numerical r/slope/p are `NOT STATED IN PAPER`.
4. **Index.** NO INDEX DEFINED. The direct analysis is selected mutation/background relationships; the main article also develops global epistasis as a biological interpretation.
5. **Printed numeric coefficient.** `NOT STATED IN PAPER`. The Figure 3 trend and error bars are plotted; no value was read from them.

### Wiser, Ribeck & Lenski (2013)

1. **Axes and scale.** The empirical outcome is fitness through time, not individual mutation effect against background fitness. The Methods say fitness is “the dimensionless ratio of the competitors’ realized growth rates” (printed p.1364); the Fig. 1 caption describes “fitness changes in nine E. coli populations between 40,000 and 50,000 generations” (printed p.1366).
2. **Mutations/reference.** In the model, “Beneficial mutations of advantage s are exponentially distributed” and “the distribution of available benefits declines after a mutation with advantage s fixes” (model description, printed p.1364). It is a sequence of fixed beneficial effects in a population model, not individual mutation/background pairs classified against a shared wild type. Here g is related to the model parameter by “g = 1/2a” (printed p.1365).
3. **Statistic.** The scalar g is a model parameter; its exact printed designation and estimate are in `record.json`. The article does not report a DRI/ICI correlation over local mutations.
4. **Index.** No landscape-level DRI/ICI index is defined. The scalar model parameter g is not the same estimand as a genotype-edge or per-mutation correlation.
5. **Printed numeric value.** “The value of g estimated for the six populations that retained the low ancestral mutation rate throughout 50,000 generations is 6.0 (95% confidence interval 5.3–6.9)” (Results, printed p.1365). Separately, the Abstract states “the correlation coefficients for the grand means and model trajectories are 0.969 and 0.986 for the hyperbolic and power-law models, respectively” (printed p.1364); those are trajectory-fit correlations, not DRI/ICI values.

### Johnson, Martsul, Kryazhimskiy & Desai (2019)

1. **Axes and scale.** Fig. 4 caption: “Histogram of regression slopes between fitness effect and background fitness for each mutation” (printed p.493). Effects were estimated from “barcode-frequency trajectories” for “each mutation in each strain” (Results, printed p.491). A log/linear transformation for these fitted effects is `NOT STATED IN PAPER` in the main text retrieved.
2. **Mutations/reference.** The analysis uses transposon insertion mutations across F1 yeast segregants. The paper says some mutations “are beneficial in the least-fit segregants, neutral in intermediate-fitness segregants, and deleterious in higher-fitness segregants” (Results, printed p.492), so their sign can change with the tested background rather than being assigned against one shared wild type.
3. **Statistic.** Fig. 4 is a distribution of per-mutation linear-regression slopes. It is neither a single pooled Pearson coefficient nor a landscape-wide slope. The printed count of increasing-cost cases and opposite-direction mutations is in `record.json` (Results, p.492).
4. **Index.** NO INDEX DEFINED. The paper reports DFE changes across segregants and a distribution/count of per-mutation regression directions/slopes.
5. **Printed numeric statistics.** Fig. 2B gives “P = 0.03, two-sided t test” for the mean DFE trend (printed p.491). For small-library DFE mean, variance and skew, Fig. 2E–G gives “P = 4.13 × 10−31, P = 7.83 × 10−14, P = 4.08 × 10−12, respectively, two-sided t test” (printed p.492). These are DFE-level trend tests, not per-mutation slopes. Fig. 4 says “48 cases” for increasing-cost epistasis and “six mutations” exhibit the opposite pattern (Results, printed p.492). Numerical per-mutation slopes are `NOT STATED IN PAPER`.

### Reddy & Desai (2021)

1. **Axes and scale.** Equation 1 states “sᵢ = s_additive,i + s_genotype,i − cᵢ y” (printed p.2); the accompanying text says “cᵢ quantifies the magnitude of global epistasis for locus i” (p.2). This is a per-locus linear slope against background fitness y.
2. **Mutations/reference.** The equation “applies independently of whether its additive effect is deleterious (increasing-costs) or beneficial (diminishing-returns)” (printed p.2). Fig. 1 caption describes “Diminishing returns of specific beneficial mutations” for the knockouts and “Increasing costs of specific deleterious mutations” for the Johnson insertions; it says “the mean over 91 mutations (in red) and 5 of the 91 mutations” (printed p.2). The immediate mutation background is y; a single shared wild-type reference is `NOT STATED IN PAPER` for this fit.
3. **Statistic.** The empirical object is a regression slope per mutation/locus, not Pearson/Spearman r. The per-locus equation coefficient is −cᵢ; fitted lines and residuals are analyzed. The numerical genotype coding and directed substitution are described in the paper’s Eq. 2/results (printed pp.2–3).
4. **Index.** No one landscape-wide DRI/ICI scalar is defined. cᵢ is a locus-specific slope parameter, not an aggregate index.
5. **Printed numeric coefficient.** No empirical per-mutation cᵢ or slope values are printed in the main figures; the regression lines are plotted. `NOT STATED IN PAPER` for a single pooled DRI/ICI coefficient.

### Bakerlee, Nguyen Ba, Shulgina, Rojas Echenique & Desai (2022)

1. **Axes and scale.** For paired genotypes, the paper plots “the fitness of a genotype with the mutated allele (ϕMut) against the fitness of the same genotype with the WT allele (ϕWT)” and says “A regression slope (b) different from 1 in these plots signifies an FCT” (Results, Fig. 3A, printed p.633). This differs from direct per-edge gain-versus-background regression.
2. **Mutations/reference.** For each locus, ploidy and environment, focal alleles are compared across otherwise paired genotypes. The paper says its landscapes “have no natural polarization (i.e., neither allele is the assumed WT)” (Discussion, printed p.635), so the slope analysis is not a beneficial-only or deleterious-only partition.
3. **Statistic.** The reported quantity is a per-locus regression slope b, not Pearson/Spearman. “Across all ploidies, environments, and loci, ~44% of regression slopes deviated substantially from 1” (Results, printed p.633). Printed example slopes and their locators are also in `record.json`.
4. **Index.** NO DRI/ICI index defined. It reports distributions and examples of FCT slopes across locus × ploidy × environment conditions; direction is not split by beneficial vs deleterious mutation classes.
5. **Printed numeric slopes.** The RHO5 G10S and AKL1 S176P examples are transcribed in `record.json` (Fig. 3C/D, printed p.633). No pooled DRI/ICI coefficient is reported.

### Papkou, Garcia-Pastor, Escudero & Wagner (2023)

1. **Axes and scale.** SI Fig. S22 says “The fitness gain of beneficial mutations (vertical axis) decreases in genetic backgrounds with higher fitness (horizontal axis)” and “fitness gain is the response variable and the fitness of the genetic background is the predictor variable” (printed SI p.42). The SI Methods prefer “the difference (r_variant – r_ref)” because “this difference ... corresponds to a selection coefficient” (Calculating fitness, SI p.8). The caption identifies a “linear regression model”; whether the measured growth rates are log-transformed before this calculation is `NOT STATED IN PAPER` in these passages.
2. **Mutations/reference.** The SI caption says “N=324,044 mutations” (Fig. S22, printed SI p.42). The analysis is labelled as beneficial mutations, so sign is assigned to each mutation transition; whether a separate shared reference genotype or statistical threshold is used is `NOT STATED IN PAPER` in that caption.
3. **Statistic.** This is a pooled linear regression slope over beneficial transitions, but no numerical slope, Pearson/Spearman coefficient, or p value is printed in the caption/main text. No plotted values were read.
4. **Index.** NO INDEX DEFINED. The supplement shows an edge-pooled trend/regression, without a named landscape-level DRI value.
5. **Printed numeric coefficient.** `NOT STATED IN PAPER` for slope/r/p. The transition count is printed in SI Fig. S22, p.42, not a coefficient.

### Lyons, Zou, Xu & Zhang (2020), contrast only

1. **Axes and scale.** The Methods say “Pearson's correlation coefficient was calculated between mutational effect on a particular trait and background trait value for each single mutation” and “also calculated between all mutational effects and background trait values for each landscape” (author manuscript, Methods, “Examining correlation between background fitness and mutational effect”). Thus effects are the responses and background fitness/trait values the predictors; no separate regression slope is reported.
2. **Mutations/reference.** For dividing mutations into classes, “mutations were deemed beneficial or detrimental depending on their effect on the wild-type genotype in GFP and tRNA, or a random arbitrary genotype in RNA-stability, or on a genotype with fitness value closest to the average fitness in the n-order landscapes” (same Methods section). Beneficial and detrimental sets are both analyzed separately.
3. **Statistic.** The paper reports per-mutation Pearson-r distributions and a separately pooled Pearson r (same Methods section). Its numerical summary says “87.8% of mutations from the yeast tRNA fitness landscape show a negative correlation between fitness effect and background fitness” (Results, Fig. 2b). That is a fraction of per-mutation r signs, not a DRI/ICI index.
4. **Index.** NO DRI/ICI INDEX DEFINED. The paper separately defines an idiosyncrasy index I_id, which is a different metric and must not be cited as DRI/ICI.
5. **Printed numeric summary.** The paper’s tRNA percentage and figure locator are in `record.json`. The chart’s exact r/P annotations are plot-rendered; they are not transcribed here and are not reproduction targets.

### Huang, Zhou & Li (2025), the published pooled-scalar definition

1. **Axes and scale.** Appendix C.3.2 says DRI correlates “the fitness of the background genotype, f(g), and the corresponding positive selection coefficient, s(g → g′)” using the “Pearson correlation coefficient” (printed p.36). Definition C.7 gives “s(g → g′) = f(g′) − f(g)” (printed p.27). This is a difference of the fitness values f; whether a particular input landscape's f values have been log-transformed is `NOT STATED IN PAPER` in this general definition.
2. **Mutations/reference.** DRI includes “all beneficial single-step mutations (where the selection coefficient f(g′) − f(g) is positive)” (Appendix C.3.2, p.36). ICI includes “all deleterious single-step mutations (where the selection coefficient s(g → g′) is negative)” and correlates background fitness with “the magnitude (absolute value) of the corresponding negative selection coefficient, |s(g → g′)|” (same locator). Sign is relative to each starting genotype g, not one shared ancestor.
   Appendix C.3.2 calls the DRI response the “positive selection coefficient”; its quoted Pearson/background-fitness definition is in `record.json` with the same locator.
3. **Statistic.** The paper’s DRI/ICI are single landscape-level Pearson correlations pooled across eligible mutation transitions (edges). They are neither a per-mutation average of Pearson coefficients nor a per-node-mean correlation.
4. **Index.** YES: this paper explicitly defines scalar global epistasis coefficients for diminishing returns and increasing costs. This is the strongest direct published scalar precedent found in this survey.
5. **Printed values.** Table A2/A3 values and locators are in `record.json`. They are printed feature-table entries computed by GraphFLA, so they are not independent validation data.

## GraphFLA implementation (read-only)

Read `/Users/arwen/Documents/GitHub/GraphFLA/graphfla/analysis/epistasis/idiosyncrasy.py`, lines 259–393 and 396–510. No repository file was changed and no GraphFLA call was run.

Verbatim implementation fragments:

```python
method: Literal["pearson", "spearman", "regression"] = "pearson"

# Mean improvement toward the optimum across each node's improving out-edges.
per_node = np.asarray(landscape.graph.strength(mode="out", weights="delta_fit"), dtype=float)
avg_successor_improvement = np.where(outdeg > 0, per_node / outdeg, np.nan)

correlation, _ = corr_func(node_fitnesses, avg_improvement)
return _pythonize(correlation)
```

```python
# Mirror of diminishing_returns_index over IN-edges: mean cost across each
# node's improving predecessors.
per_node = np.asarray(landscape.graph.strength(mode="in", weights="delta_fit"), dtype=float)
avg_predecessor_cost = np.where(indeg > 0, per_node / indeg, np.nan)
```

Thus, default DRI is Pearson correlation across nodes that have at least one improving outgoing edge: x=f(v), y=mean positive fitness gain over v’s improving one-mutant neighbors. Default ICI is Pearson correlation across nodes with at least one improving incoming edge: x=f(v), y=mean incoming gain magnitude. Each incoming improving edge is the reverse of a deleterious mutation from that fitter target genotype, so its target fitness is the deleterious mutation’s background fitness. Optional modes return Spearman rₛ or a least-squares slope; the Pearson code discards the p value.

### Divergences from surveyed conventions

- **Aggregation is different.** Chou, Khan, Kryazhimskiy, Johnson, Reddy and Bakerlee analyze mutation-specific effects across backgrounds, then report individual-mutation slopes/correlations or their distribution. GraphFLA first averages improving effects for each source node (DRI) or target node (ICI), then correlates the node averages. This loses which mutation generated each effect and gives each node equal weight regardless of how many improving transitions it has.
- **It is not Papkou’s edge pool.** Papkou regresses each of the 324,044 beneficial mutation transitions. GraphFLA replaces each source genotype’s multiple edge gains with one mean before correlation.
- **It is not Huang’s scalar.** Huang pools individual beneficial/deleterious single-step transitions in the Pearson calculation. GraphFLA calculates Pearson across node means. For ICI, mapping an improving edge to its reverse deleterious transition gives the right cost magnitude and background node, but the per-node aggregation still changes the estimand.
- **It is not Lyons et al.’s analysis.** Lyons reports distributions of per-mutation Pearson coefficients, and a pooled all-mutation coefficient; GraphFLA returns one correlation across node-averaged effects.
- **It does not reproduce the per-mutation regression slope convention.** Choosing `method="regression"` changes GraphFLA’s final summary from a correlation to a slope, but it still regresses node means, not individual mutation effects for each locus/background set.
- **No significance test is exposed.** The Pearson function unpacks `(correlation, _)`, so the p value is discarded. Several papers report significance per mutation or per trend.

## Acquisition notes and limits

The new source log is [`papers/_dri_ici/acquisition_codex.jsonl`](../../papers/_dri_ici/acquisition_codex.jsonl). Valid local main-text sources include Chou, Kryazhimskiy, Wiser, Johnson, Reddy & Desai, Bakerlee and Lyons; the existing Papkou Science main-text PDF and extracted Supplementary Fig. S22 text were reused. The Huang NeurIPS PDF was copied from the already-read `_core_lit_epistasis` dossier. Copies and source hashes are identified in the log.

Failed retrievals are preserved as `_attempt_codex` responses and recorded in the log: ResearchGate’s Khan PDF route returned 403 and its SI DOI endpoint returned 404; Kryazhimskiy’s SI DOI endpoint returned 404 and the Europe PMC XML endpoint returned 500; Johnson’s SI route returned 403; Bakerlee’s SI route returned 403; Papkou’s publisher SI PDF route returned 403. The Khan article’s full-text ResearchGate page was browser-readable and exposed the article text and Figure 4 annotations, but no valid local PDF was obtained. The first Reddy & Desai article request failed client-side Brotli decoding, with no status or body available to save; the retry with `Accept-Encoding: identity` succeeded. Papkou’s pre-existing extracted SI text was copied from `/Users/arwen/Documents/GitHub/GraphFLA/_verify/papers/papkou_science_SM.txt`; the separate publisher-PDF 403 response is saved in the dossier.

Lyons’s main publisher version is paywalled; the accepted full-text XML from Europe PMC was used, consistent with the previously read `papers/Lyons2020/definition_audit_claude.md`. The user-supplied Lyons conclusion was treated as established context, not re-derived from scratch.

No older-paper trend values were read from plotted points. Fig. S22’s unlabelled fitted slope, most Chou/Kryazhimskiy per-mutation effects, and Lyons’ embedded chart r/P annotations remain plot-only and are not reproduction targets.
