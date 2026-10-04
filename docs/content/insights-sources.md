# Interactive benchmark data

The homepage relates protein landscape features to prediction accuracy and
optimization outcomes.

## Zero-shot prediction

ProteinGym substitution assays are retained when the mean number of amino-acid
substitutions per measured variant is strictly greater than 1.5. The prediction
scores are ProteinGym's published per-assay Spearman correlations.

The cohort contains 36 assays from the [ProteinGym v1.3 release](https://zenodo.org/records/15293562),
joined by assay ID to 97 model columns in the
[published score table](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv).

Eligibility uses all measured variants before construction. GraphFLA's standard
construction removes variants with no observed one-step neighbor; the recorded
data distinguishes the original assay size, graph size and number of removed
isolates. The analysis features describe that constructed graph.

## Directed evolution

This panel compares four search strategies on 18 measured protein landscapes.
Random-forest acquisition policies and random search
start with the same 96 measured variants, then query four batches of 96 previously
unmeasured variants. Greedy directed evolution uses up to the same 480-evaluation
cap and can stop earlier at a local optimum. Results average ten paired seeds.

The endpoint is the fraction of observed variants whose fitness is no greater than
the best discovered variant. Reaching an observed global maximum scores 100%,
including when it is tied. Ranks are calculated over observed variants in the
measured library.

The 18 protein landscapes come from GraphFLA's
[BioSequence collection](https://github.com/COLA-Laboratory/GraphFLA/tree/main/data/BioSequence).
Their primary studies cover [PhoQ](https://doi.org/10.1126/science.1257360),
[GB1](https://doi.org/10.7554/eLife.16965),
[ParB and Noc](https://doi.org/10.1016/j.celrep.2020.107928),
[TEV and T7](https://doi.org/10.1101/2022.03.09.483646),
[ParD antitoxins](https://doi.org/10.7554/eLife.60924), and
[ten TrpB libraries](https://doi.org/10.1073/pnas.2400439121).

## Landscape features

Both panels use one-substitution neighbor graphs and the same 14 feature definitions.
Fitness–distance correlation uses Spearman correlation;
autocorrelation uses 1,000 walks of length 20 at lag 1. Epistasis fractions use sampled
motifs where needed, with the cutoff recorded per dataset. Seeded
calculations use seed 20261004. Exact calls, source hashes and reasons for unavailable
values are included in the preparation bundle.

Five assays have no roughness-to-slope ratio: three have rank-deficient additive
fits, and two exceed the API's QR workspace limit. ProteinGym's source table also
leaves three Protriever scores empty. Other selections retain all eligible points.

## Reading the plots

Each point is one dataset. The fitted line is an ordinary least-squares fit through
the displayed complete feature–outcome pairs; no confidence interval is shown.
The association does not establish causality. A missing or undefined metric is
omitted only for that selection, and the displayed dataset count updates accordingly.

Hovering or selecting a point opens its study citation and external publication
link. Feature values, outcomes, citations and preparation metadata are available in
[the plotted data](assets/home/data/insights.json). The
[preparation bundle](assets/home/data/insights-preparation.zip) includes the scripts,
source manifests and per-run optimization results.
