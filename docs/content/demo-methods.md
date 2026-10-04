# Reproducible data examples

These five downloadable examples use the same construction and analysis workflow
with different variables, objectives and neighborhoods. Their recorded results
are computed by GraphFLA. They are separate from the homepage's illustrative demo
tables, authored landscapes and example profiles.

| Example | Analyzed input | Objective | Neighborhood |
| --- | --- | --- | --- |
| Protein engineering | 16 measured GB1 variants, using C/V, D/E, A/G and A/V at the four variable sites | Maximize binding strength | One amino-acid substitution |
| Chemistry | 384 Suzuki–Miyaura conditions | Maximize UV product-area percentage | Replace one ligand, base or solvent |
| Materials | 496 W–Re–Os compositions | Maximize hardness at 1,000 °C | One adjacent W or Re level; Os is the remainder |
| Software tuning | 64 measured LLVM configurations; six flags vary and four stay off | Minimize compilation time | Flip one flag |
| Hyperparameters | 48 random-forest settings, evaluated on the built-in scikit-learn diabetes regression dataset | Minimize validation RMSE | One adjacent depth, leaf-size or feature-sampling level |

The HPO grid varies maximum depth (2, 4, 6, 10), minimum leaf size (1, 2, 4, 8)
and feature fraction (0.5, 0.75, 1.0). Each forest has 40 trees and uses one CPU
thread. A fixed 75/25 train/validation split and model seed 42 make the comparison
reproducible. These are measured model evaluations, not invented scores.

## Interpreting the four results

**Local optima** counts configurations or neutral plateaus with no improving move.
The protein example uses **roughness-to-slope ratio**; the other examples use
**lag-one autocorrelation**, estimated from 200 walks of length 20 with seed 42.

**Near-equal neighbors** uses an absolute tolerance equal to 1% of that example's
observed objective range. The displayed tolerance is in the original objective
units. This does not claim exact equality or experimental insignificance.

**Interaction patterns** shows exact directed four-node motif proportions from
`classify_epistasis`: magnitude, sign and reciprocal-sign classes, labeled by
whether local changes retain their direction, reverse one effect, or reverse both.
Missing or equal-outcome edges can exclude a motif. These proportions describe
the constructed graph, not every possible combination of variables.

## Recorded graph coordinates

The recorded coordinates describe at most 32 connected configurations selected from the analyzed
graph, including its best observed configuration. Edges are actual neighbor pairs.
The planar arrangement is a force-directed projection, not a physical distance.
Objective values are normalized so better outcomes are higher even for minimization.
The reported metrics use the full analyzed input in the table above.

## Download and reproduce

- [Protein data](assets/home/data/protein.csv)
- [Chemistry data](assets/home/data/chemistry.csv)
- [Materials data](assets/home/data/materials.csv)
- [Software data](assets/home/data/software.csv)
- [Hyperparameter grid](assets/home/data/hpo.csv)
- [Recorded values, graph coordinates and provenance](assets/home/data/results.json)

The repository script `docs/home/prepare_scenarios.py` reproduces these examples
in an environment with GraphFLA, pandas, scikit-learn and igraph installed. Normal
documentation builds render the recorded data without fitting models or running
analysis. Original experimental sources are described in the
[dataset collection](datasets.md) and [worked tutorials](tutorials/index.md).
