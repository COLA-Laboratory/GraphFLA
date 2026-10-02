# Crona, Greene & Barlow (2013), with TEM companion papers

Raw source and reproduction notes for the core-literature peak-count target and its linked TEM analyses.

## Bibliography and acquired sources

The main paper was found by title and DOI, then checked against Crossref: Kristina Crona, Devin Greene, and Miriam Barlow, “The peaks and geometry of fitness landscapes,” *Journal of Theoretical Biology* 317, 1–10 (2013), DOI `10.1016/j.jtbi.2012.09.028`. The supplied author list is correct.

The TEM companion used here is Goulart et al., “Designing Antibiotic Cycling Strategies by Determining and Understanding Local Adaptive Landscapes,” PLOS ONE 8(2), e56040 (2013), DOI `10.1371/journal.pone.0056040`. The PLOS publisher page gives the eight-author list recorded in `record.json`. I also inspected the later Mira et al. TEM-50 treatment-plan paper, DOI `10.1371/journal.pone.0122283`; its publisher-page author list is recorded in `related_references`.

Acquired sources are stored under `papers/_navigability_lit/sources/` and checksummed in `papers/_navigability_lit/acquisition_codex.jsonl`. The JTB article was retrievable as PMC author-manuscript BioC JSON. The attempted Europe PMC XML and ScienceDirect PDF routes failed; those failures are logged. The Aguilar-Rodríguez paper and supplement were reused from the earlier `papers/_core_lit_navigability` dossier.

## Exact small-landscape reproduction from Crona et al.

Example 1 prints “exactly two peaks” and states “2^{L−2} type 2 systems and no type 1 systems.” Locator: §2, Example 1. The displayed equation is “w̃(11s)=4+∑s_i; w̃(10s)=1+∑s_i; w̃(01s)=2+∑s_i; w̃(00s)=3+∑s_i.” For the L=2 case, `s` is empty; specializing this equation gives the four input values `w(00)=3`, `w(01)=2`, `w(10)=1`, and `w(11)=4`.

The paper's Figure 1 caption says “arrows point toward the more fit genotype.” Locator: §1, Figure 1 caption. This makes the two peaks in the L=2 example the two sink vertices. GraphFLA, run read-only with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH` set to the repository, returned `n_lo=2` exactly.

For epistasis classification, the paper states that a “type 2 system corresponds to reciprocal sign epistasis.” Locator: §2, Discussion. The Example 1 formula gives `2^{L−2}` type-2 systems and none of type 1; at L=2 this is the one available two-locus square. GraphFLA returned `reciprocal_sign=1.0` and `sign=0.0`, matching that classification. The complete call and the constructed four rows are in `papers/_navigability_lit/reproduce_codex.py`.

The first script run built the landscape and computed the classification, then failed when `json.dumps` received GraphFLA's `EpistasisClassification` dataclass. I changed the output formatter to `dataclasses.asdict` and reran. The corrected run printed:

```text
{"classify_epistasis": {"magnitude": 0.0, "negative": 0.0, "positive": 1.0, "reciprocal_sign": 1.0, "sign": 0.0}, "configs": 4, "edges": 4, "n_lo": 2}
```

## TEM-85 peak counts and accessibility probabilities

The published TEM-85 result is direct: “The cefotaxime landscape has 2 peaks” and “the ceftazidime landscape has 4 peaks.” Locator: Goulart et al., Results, “TEM-85 landscapes,” printed p. 4. These are candidate `n_lo` overlaps. The paper's fitness-graph construction says the “directed edges ... are determined by the statistical analysis of the resistance differences among alleles” (Methods, “Identifying Paths and Cycles”). It also says: “We used one-way ANOVA testing to determine 95% confidence intervals around the mean resistance phenotypes and to assign direction.” (Results, “TEM-85 landscapes,” printed p. 4.) The Figure 1 caption says: “absence of a line indicates that the adjacent nodes are phenotypically equivalent.” GraphFLA's raw-fitness ranking does not reproduce those statistical edge decisions. I did not run `n_lo` on an unrelated landscape to force a comparison.

The same TEM-85 section prints fixation probabilities of “75%” for cefotaxime and “12.5%” for ceftazidime, under a model that assumes “available beneficial mutations are equally likely to occur and go to fixation.” Those are stochastic fixation probabilities, not a GraphFLA path-accessibility fraction. The user excluded `global_optima_accessibility` from literature validation, so no GraphFLA call was made.

The available `Mira2015_TEM_*` files do not provide the TEM-85 genotype-fitness input for those two published peak counts. Mira et al.'s later paper states: “We created all 16 variant genotypes of the four amino acid substitutions found in TEM-50.” Locator: Results, “From experimental data to mathematical models,” printed p. 3. The later paper's printed results are treatment-plan probabilities, not a peak count; peak count: **NOT STATED IN PAPER**. I did not substitute those TEM-50 files for the TEM-85 landscapes.

## Weinreich input warning cross-check

I checked the flagged all-ones row without running it through GraphFLA. The source-table extraction from Weinreich et al.'s Supplementary Table S1, genotype `11111`, reads:

```text
11111,1,1,1,1,1,4100.0,4100.0,4100.0
```

Locator: Supplementary Table S1, genotype `11111`; verbatim replicate extraction at `papers/ferretti2016/sources/Weinreich2006_TableS1_replicates_codex.csv`, row 33. The repository copy, `data/BioSequence/Weinreich2006Tan2011_Weinreich2006.csv`, row 33, reads:

```text
11111,1,1,1,1,1,15.32333629
```

The repository value is the log2 of 41,000 to the displayed precision, whereas the source table has 4,100 in each replicate. This confirms the flagged tenfold source mismatch. That file was not used in either reproduction here.

## Execution-scope note

The corrected GraphFLA call set `MPLCONFIGDIR` to `papers/_navigability_lit/mplconfig_codex`, inside the validation workspace, as well as the required `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH`. The initial call omitted `MPLCONFIGDIR`; Matplotlib created a temporary font cache at `/var/folders/cr/vx9vlg4j2_74p6mm66j9hs6r0000gn/T/matplotlib-g8nvsqby`, outside the allowed validation directory. This was an accidental scope violation. I did not modify any file under the GraphFLA repository. I have not removed the temporary cache because the active instruction limits writes to the validation workspace.
