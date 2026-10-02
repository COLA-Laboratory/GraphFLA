# Papkou 2023, Fig. S12 — raw reproduction notes

## Bibliography and source trail

Crossref's DOI record verifies the title and the author list Andrei Papkou, Lucia Garcia-Pastor, José Antonio Escudero, and Andreas Wagner. The local supplied entry is wrong: [`selected_references.bib`](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/selected_references.bib:12) says `author = {Andrei Papkou and Aviv Regev and Joanna Masel}`. The downloaded Crossref JSON and its checksum are listed in [record.json](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/papers/Papkou2023/sources/Papkou2023_crossref_adh3860_codex.json) and [acquisition_codex.jsonl](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/papers/Papkou2023/acquisition_codex.jsonl).

The existing dossier supplied the main article PDF and a local extracted supplementary text. I reused those sources. A request to the Science supplementary PDF URL returned HTTP 403 with a Cloudflare verification page, saved as an HTML response; acquisition log line 2 records the attempt and line 3 records the response file and checksum. No PDF was saved. The SI text copy is preserved as `papers/Papkou2023/sources/Papkou2023_science_adh3860_SM_codex.txt` and is identified as a reused local source in line 4 of the log and in the record.

## Basin definition and ties

The main article Methods definition is quoted once in the Asp overlap's `definition_quote` field in [record.json](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/harvest/Papkou_FigS12/record.json), locator: Materials and Methods, printed p. 7. It defines a basin by whether an evolutionarily accessible path exists from a variant to a given peak. The supplementary Methods says an accessible path is one “in which every mutation increases fitness” (Materials and Methods, “Accessible paths,” printed p. 15; [SI text lines 686–691](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/papers/Papkou2023/sources/Papkou2023_science_adh3860_SM_codex.txt:686)).

Tie-breaking for assigning a variant to one peak: **NOT STATED IN PAPER.** The paper's definition is reachability to each peak, so one variant can be included in multiple basins. That agrees with its separate analysis of basin overlaps. The printed target annotations for the four classes are quoted in the `quote` fields of the record, with each locator pointing to the relevant Fig. S12 annotation on printed p. 32. They are caption text, not values read from plotted points. The paper's p-value calculation method: **NOT STATED IN PAPER.**

## Position-27 grouping

The main article describes “nine nucleotide positions,” “three successive amino acids,” the wild-type “26A-27D-28L” sequence (Results, printed p. 1), and the “position 27” axis (Fig. 3 caption, printed p. 4). From this encoding, position 27 is the middle codon of the nine-nucleotide sequence. I therefore translate the second triplet, Python slice `[3:6]`, after applying the already-established nucleotide-label reversal. The paper does not state this string offset: **NOT STATED IN PAPER.**

The paper names the amino-acid groups but does not provide their codon mapping: **NOT STATED IN PAPER.** I applied the standard DNA genetic code; NCBI states that “the genetic code tables shown here use T instead of U” ([NCBI Genetic Codes, lines 9 and 45–49](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?mode=c)). The exact Asp/Glu/Cys codon sets used are in [reproduction_fig_s12_codex.py lines 24–40](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/papers/Papkou2023/reproduction_fig_s12_codex.py:24). The count partition matches the four sample sizes in Fig. S12 and totals GraphFLA's complete peak set; the script checks this before it accesses either basin property or computes a correlation. See [captured output line 3](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/papers/Papkou2023/reproduction_fig_s12_output_codex.txt:3).

## Preprocessing and computation

I reused the preprocessing in [`tests/test_literature.py`](/Users/arwen/Documents/GitHub/GraphFLA/tests/test_literature.py:42), without trying alternate transforms:

```python
str.maketrans({"A": "A", "C": "G", "G": "T", "T": "C"})
tau=-0.507774,
filter_mode="both",
epsilon=0,
verbose=False,
```

The input is `data/BioSequence/Papkou2023_DHFR.csv`; its SHA-256 and row count are in each `input` object in the record. The run command was:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/arwen/Documents/GitHub/GraphFLA python papers/Papkou2023/reproduction_fig_s12_codex.py
```

### Definition comparison

GraphFLA's [`.accessible_paths` property](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/landscape/landscape.py:388) counts incoming ancestors for each peak; its implementation describes them as reachable “via any fitness-increasing path” ([`_compute.py`](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/landscape/_compute.py:68)). This matches the paper's basin definition, including overlapping basin membership. I used those accessible-basin sizes for the four grouped coefficients.

The named [`basin_fitness_correlation`](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/analysis/correlation.py:205) function instead reads `size_basin_greedy` from `landscape.basins` (lines 222–238); the property is documented as “Per-node greedy basin size” ([`landscape.py` line 381](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/landscape/landscape.py:381)). That machinery assigns each variant to one greedy hill-climb endpoint. The walker code describes a “first-maximum tie-break” ([`walk.py` line 125](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/algorithms/walk.py:125)); its next line says the successor order is the graph-neighbor order ([line 126](/Users/arwen/Documents/GitHub/GraphFLA/graphfla/algorithms/walk.py:126)). Its grouped diagnostic values are preserved separately in [captured output line 5](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/papers/Papkou2023/reproduction_fig_s12_output_codex.txt:5); they differ from the printed Fig. S12 coefficients. Thus the named scalar API is definition-incompatible with this paper statistic, even though GraphFLA's separate accessible-path basin property supports a matching calculation.

## Outcomes and run notes

The accessible-path Spearman coefficients round to the printed two-decimal rho values in all four classes, and the sample counts match. The full printed annotations are marked `mismatch` for Asp, Glu, and other because SciPy's default p-values differ from the caption at its stated precision or inequality. Cys matches its printed p inequality. The p-values are ancillary to the rho coefficient, but are retained rather than silently omitted in [record.json](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/harvest/Papkou_FigS12/record.json) and [captured output line 4](/Users/arwen/Documents/GraphFLA-validation/2026-09-30/papers/Papkou2023/reproduction_fig_s12_output_codex.txt:4).

The first invocation failed with `ImportError` because `DNALandscape` is not exported from the top-level `graphfla` module. I corrected the import to `graphfla.landscape.DNALandscape`; the subsequent run completed and produced the captured output. No preprocessing variant was tried to force a match.
