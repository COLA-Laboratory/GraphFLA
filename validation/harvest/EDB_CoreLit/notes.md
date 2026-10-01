# Extradimensional bypass: core literature and exact scalar reproduction

## Sources and scope

Papkou et al., *A rugged yet easily navigable fitness landscape*, *Science* (2023), DOI [10.1126/science.adh3860](https://doi.org/10.1126/science.adh3860), is the numerical anchor. The Supplementary Materials text at `papers/Papkou2023/sources/papkou_science_SM_sol.txt` is a byte-for-byte copy of the pre-existing read-only extraction at `/Users/arwen/Documents/GitHub/GraphFLA/_verify/papers/papkou_science_SM.txt`; SHA-256 `abd424ad6fee922b61e34ccb961490e28f820cdc17a4057dc123dc783b293072`. The paper's title, year, and DOI are verbatim in the supplement title block (unnumbered p. 1). Existing dossiers supplied the main Papkou article, Wu et al. article, and Conrad article; no paper download was needed. The attempted publisher supplement URL returned 403 to the web reader, so the copy's original HTTP status is **NOT STATED**.

## What the field calls a bypass

- Conrad, *The geometry of evolution* (1990), printed p. 68, right column: “We will call this upward-running pathway an extradimensional bypass, since the peaks remain isolated except in one of the newly added dimensions.” This is the geometric origin of the term. A motif-level percentage denominator is **NOT STATED IN PAPER**.
- Wu et al., *Adaptation in protein fitness landscapes is facilitated by indirect paths* (2016), printed pp. 6–7, Figure 2 caption and Results lines 100–121: “a successful bypass would require a conversion step that substitutes one of the two interacting sites with an extra amino acid (00 → 20), followed by the loss of this mutation (21 → 11).” For the other mechanism, lines 109–111 say “taking a detour step to gain a mutation at the third site (000 → 100), followed by the later loss of this mutation (111 → 011).” Their final summary, lines 121–122, says “The two distinct mechanisms of bypass both require the use of indirect paths, where the Hamming distance to the destination is either unchanged (conversion) or increased (detour).” Wu's Figure 2 caption prints “>40%” for conversion, “<20%” for detour, and identifies the tested cases as “∼20,000 randomly sampled reciprocal sign epistasis.” These are mechanism-specific, sampled success rates; they are **not** the full-network union prevalence GraphFLA returns.
- Papkou et al., Supplementary Materials, section “Extradimensional bypasses,” printed p. 19: “However, a fitness landscape may contain indirect path(s) that lead from ab to AB and that are accessible. Such an indirect path is also called an extradimensional bypass.” The algorithm on that page says to “Search for the shortest accessible path, i.e., a path along which fitness increases in each step,” from the rank-2 `ab` to rank-1 `AB` “within the whole network,” using `mode=out`. This supplies the operational definition for this audit. Supplementary Figure S20 caption, printed p. 40, says: “The total number of squares with reciprocal sign epistasis is 85,203.”

The original geometric idea is wider than any one frequency statistic. Wu operationalizes two named short mechanisms; Papkou operationalizes existence of any improving path between the two high-fitness corners of each reciprocal-sign square. GraphFLA's implementation at `graphfla/analysis/epistasis/motifs.py:375–519` selects type-19 squares, picks the highest-fitness node as `AB`, selects the opposite `ab`, and calls `graph.distances(..., mode="out")` over the full graph. Its `bypass_proportion` is `motifs_with_bypass / total_motifs`. This matches Papkou's existence test on the directed landscape. The code reports the mean of shortest distances only among bypassed squares; a corresponding printed mean is **NOT STATED IN PAPER**.

## Printed target and input

Papkou Supplementary text S3, printed p. 20, states: “In our landscape, such paths do exist, but they can overcome only 23% of local fitness reductions caused by reciprocal sign epistasis (fig. S20D).” This is a text value, not an estimate read from a plot. Supplementary Figure S20 caption, printed p. 40, states: “The total number of squares with reciprocal sign epistasis is 85,203.”

Papkou Supplementary Materials, “Determining nonfunctional mutations,” printed p. 14: “This procedure resulted in a relative fitness cut-off of 𝑟 𝑖 − 𝑟 WT = -0.507774.” The following “Constructing a network of variants” procedure says: “Exclude pairs where both variants are nonfunctional” and “Extract the largest connected subgraph (or giant component(55)) of the network using the components function of igraph with the argument ‘mode=weak’.” Printed p. 15 says: “It contains 135,178 variants and 324,044 edges between them.” The existing curated CSV, `data/BioSequence/Papkou2023_DHFR.csv`, has SHA-256 `fca2fb47cad417a049698f43b20f90b320a4927418d502e527ad112ea46c82da`; its provenance and the source archive row discrepancy are documented in `papers/Papkou2023/data_audit.json`. The present run used the already validated component and did not re-estimate the paper's raw fitness values.

## Exact GraphFLA run

The temporary script was executed from this validation workspace with the required command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/arwen/Documents/GitHub/GraphFLA python edb_run_sol.py
```

It called `DNALandscape().build_from_data(data["sequences"], data["fitness"], epsilon=0, tau=-0.507774, filter_mode="both", verbose=False)` and `extradimensional_bypass(landscape, sample_cut_prob=0)`. The script's stdout is the locator for every computed number below:

```json
{"stage": "graph_built", "seconds": 0.6529215408954769, "vertices": 135178, "edges": 324044}
{"stage": "completed", "seconds": 140.2195832079742, "input_sha256": "fca2fb47cad417a049698f43b20f90b320a4927418d502e527ad112ea46c82da", "input_rows": 135178, "result": {"bypass_proportion": 0.22742156966304003, "average_bypass_length": 4.353769933426227, "total_motifs": 85203, "motifs_with_bypass": 19377}}
```

The computed fraction is `19377 / 85203 = 0.22742156966304003`, using the verbatim stdout values above. Rounded to the paper's whole-percent precision, it reproduces the printed `23%`. The `85,203` denominator also equals the printed Figure S20 count. A paper value for `19,377`, for `4.353769933426227`, or for an exact unrounded bypass fraction is **NOT STATED IN PAPER**. No value was read from Figure S20D's plotted curve, and no claim is made that the plotted length distribution was reproduced.

## Verdict

**`reproduced_with_precision` for Papkou's printed scalar bypass prevalence.** The operation, endpoints, path direction, full-network scope, and denominator agree with the paper's supplementary algorithm. Conrad supplies the conceptual origin; Wu's conversion and detour percentages are different statistics and should not be used as direct benchmarks for GraphFLA's combined full-network output.
