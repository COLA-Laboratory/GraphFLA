# Bank 2016 epistasis class fractions

## Bibliography verification

I queried Crossref by DOI. The target record returns:

> `"title": ["On the (un)predictability of a large intragenic fitness landscape"]`; authors in order: Claudia Bank, Sebastian Matuszewski, Ryan T. Hietpas, Jeffrey D. Jensen; `"DOI": "10.1073/pnas.1612676113"`; volume 113, issue 49, pages 14085–14090.

Locator: [Crossref record for 10.1073/pnas.1612676113](https://api.crossref.org/works/10.1073%2Fpnas.1612676113). The supplied page citation `113:14424` does not match Crossref's 14085–14090 pagination.

The supplied author list matches a different Crossref record:

> `"title": ["A Systematic Survey of an Intragenic Epistatic Landscape"]`; authors Claudia Bank, Ryan T. Hietpas, Jeffrey D. Jensen, Daniel N.A. Bolon; DOI `10.1093/molbev/msu301`.

Locator: [Crossref record for 10.1093/molbev/msu301](https://api.crossref.org/works/10.1093%2Fmolbev%2Fmsu301). The supplied title alternatives and authors therefore combine two distinct records; the target DOI is the 2016 On the (un)predictability paper.

## Overlap, definitions, and denominator

The overlap is GraphFLA `classify_epistasis`. The paper's exact per-square definition is Eq. S1_16:

> `e(g,i,j) = none if |s_i(g[j]) − s_i(g)| = 0; magnitude if s_j(g)s_j(g[i]) ≥ 0 and s_i(g)s_i(g[j]) ≥ 0; reciprocal sign if s_j(g)s_j(g[i]) < 0 and s_i(g)s_i(g[j]) < 0; sign else.`

Locator: Supporting Information, Eq. S1_16, printed p. 3. The supplement gives “ε = 10−6” for assigning `none`. That is a class-assignment threshold on change in a selection effect; it does not delete a square from the denominator.

Eq. S1_17 defines the normalization over all genotype backgrounds and eligible allele pairs:

> `E(x) = c^-1 Σ_g Σ_{k=I(g[i])+1}^{m_i} Σ_{l=k+1}^{m_j} 1_x(e(g,k,l))` [S1_17], with `x ∈ {none, magnitude, sign, reciprocal sign}` and `c = Σ_g Σ_k Σ_l 1`.

Locator: Supporting Information, Eq. S1_17 and following text, printed p. 3. Thus additive/no-epistasis cases are included in the denominator under `none`; they are not filtered out. The printed standard-threshold mean for `none` is 8.24E-06 (Fig. S3_5B). Squares with a neutral single-mutation effect are not listed as a separate excluded class; an exclusion rule for them is **NOT STATED IN PAPER**. Magnitude's formula includes products with `≥ 0`.

No significance test or confidence-interval rule is specified for filtering squares from the denominator. **NOT STATED IN PAPER** for any significance-based denominator exclusion. The paper's Fig. S3_5A reports a sensitivity analysis for “different values of ε”; that changes the tolerance used to assign `none`, not the Eq. S1_17 denominator. I did not recompute that alternate ε because the replicate-specific fitted medians needed to set it are not deposited.

Fitness is growth rate used as a proxy for fitness. The paper's Fig. 2 caption states “growth rate as a proxy for fitness” (printed p. 14087), and SI Eq. S1_1 defines `s_j(g)=w(g[j])−w(g)`. For the class-fraction target, SI Fig. S3_5B says “10,000 posterior samples” at the “standard threshold ε = 10−6” (printed p. 9 of 9). This is posterior-sample averaging, not one class count from one median landscape.

## Printed targets and precision

The main text prints whole-percent fragments “sign (30%),” “reciprocal sign (8%),” and “remaining 62%” (Results, “Adaptive Walks on the Fitness Landscape,” printed pp. 14086–14087). This is printed text, not a plot read-off.

The SI prints a numeric table in Fig. S3_5B. The transcribed table entries are:

> `none 8.24E-06 [0, 9.47E-05]; magnitude 0.62 [0.612, 0.628]; sign 0.298 [0.291, 0.305]; reciprocal sign 0.082 [0.078, 0.086]`.

Its mean fractions and 95% IQRs are:

| Class | Printed mean | Printed 95% IQR |
|---|---:|---:|
| none | 8.24E-06 | [0, 9.47E-05] |
| magnitude | 0.62 | [0.612, 0.628] |
| sign | 0.298 | [0.291, 0.305] |
| reciprocal sign | 0.082 | [0.078, 0.086] |

Locator: Supporting Information, Fig. S3_5B table, printed p. 9 of 9. These values are printed in a table within the figure panel, not read from plotted points. The displayed mean precision is two decimals for magnitude and three for sign and reciprocal sign. The main text rounds them to 62%, 30%, and 8%.

## Available inputs and posterior landscape

The main paper's Data deposition section points to Dryad (printed p. 14090). Its existing Dryad inventory lists one file, `data.csv`. Dryad describes it as:

> “This file contains the deep mutational scanning data from both replicates.”

Locator: Dryad dataset page, Usage notes; [Dryad 10.5061/dryad.th0rj](https://datadryad.org/dataset/doi%3A10.5061%2Fdryad.th0rj). The released file is raw sequencing counts, not the 10,000 fitted growth-rate samples.

The closest recovered full landscape is the existing Reia and Campos archive in this dossier. Its cleaning code says:

> `raw_data.columns = ["aaSeq", "median"]`

Locator: `papers/bank2016/sources/cleaning_file_HSP90.py`, line 184. The previous dossier audit records “640 variants and a stop control” (`papers/bank2016/report.txt`, paragraph 2). I excluded `Q*GWSANME`, used the remaining fitness values as archived, and did not transform them. This is one downstream point-estimate vector, not the original MCMC posterior vectors.

I also checked the later codon-space reanalysis. It describes “576 possible single-codon mutations” (Matuszewski et al., Materials and Methods; [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6180090/)) and is the Bank et al. 2014 single-codon experiment, not the 2016 640-combination landscape.

I checked the later Bank/Bolon Hsp90 study as well. Its abstract describes “44,604 single codon changes encoding 14,160 amino acid variants,” and its Methods say, “For each mutant we obtained 10,000 posterior samples.” Locator: Flynn et al. 2020, abstract and “Determination of selection coefficient”; [eLife article](https://elifesciences.org/articles/53810). Those posterior samples belong to the later single-codon library and do not supply the 2016 multi-mutant posterior landscape.

The public `empiricIST` source archive is saved as `papers/bank2016/sources/empiricIST_codex.zip` and logged in `papers/bank2016/acquisition_codex.jsonl`. Its README says the MCMC program is “written in C++” and is in `empiricIST_MCMC` (archive member `empiricIST-master/README.md`). This supplies an implementation for re-estimation, not the authors' fitted posterior output.

A targeted public search for a Bolon-lab posterior file did not locate one; this is a search result, not proof that no private or unindexed copy exists.

## Reproduction attempts

1. **GraphFLA on the Reia/Campos point-estimate landscape.** From the validation working directory, I ran the installed repository with the required `PYTHONDONTWRITEBYTECODE=1` setting and `PYTHONPATH=/Users/arwen/Documents/GitHub/GraphFLA`, calling `ProteinLandscape().build_from_data(sequences, fitness, epsilon=0, verbose=False)` and `classify_epistasis(landscape, sample_cut_prob=0, seed=0)`. GraphFLA returned magnitude 0.6384469696969697, sign 0.2771780303030303, and reciprocal sign 0.084375 (63.8447%, 27.7178%, 8.4375%).

2. **Independent square enumeration and the paper's ε.** Four-corner enumeration gives 6,742 magnitude, 2,927 sign, and 891 reciprocal-sign squares (10,560 total). I applied Eq. S1_16 with ε = 10−6 to the interaction-effect difference on each square. It classified no square as `none` and returned the same three counts. This paper-justified variant shows that the `none` tolerance cannot account for the gap on this input.

   GraphFLA's source uses:

   > `total_mag_sign_recip = reci_sign_count + sign_count + mag_count`
   > `"magnitude epistasis": mag_count / total_mag_sign_recip`
   > `"sign epistasis": sign_count / total_mag_sign_recip`
   > `"reciprocal sign epistasis": reci_sign_count / total_mag_sign_recip`

   Locator: `GraphFLA/graphfla/analysis/epistasis/motifs.py`, lines 331–342. So GraphFLA does not report the paper's `none` proportion. On this recovered vector that difference is numerically zero under the paper's ε rule.

3. **Fresh MCMC from Dryad raw counts.** I formed one input per replicate and grouped synonymous nucleotide sequences by shared protein sequence, following the paper's “equal growth rates” rule (Materials and Methods, “Estimation of Growth Rates,” printed p. 14089). Each exploratory run used burn-in 100,000, 1,000 saved samples, subsampling 10 accepted states, one set, and a distinct seed (20161101 / 20161102). Replicate 1 had minimum ESS 2.69475 overall (2.74262 for growth rates); replicate 2 had minimum ESS 2.6812 overall (2.70323 for growth rates). These short chains are not converged enough to count as reproductions. The paper reports “minimum 725” effective sample size (same Methods locator), so I did not use the short-chain class fractions as targets.

   A five-sample Replicate-1 smoke run (burn-in 100, subsampling 1, one set of 5, seed 20161001) only checked that the public executable accepted the formatted input; it provided no usable posterior summary.

   The downloaded C++ source stores sampling times as integers and parses the header with `times[...] = atoi(token.c_str())` (`empiricIST_MCMC/OS-Binaries/MacOSX/Binaries/empiricIST_MCMC.cpp`, line 558). The Bank raw-count header is `3.4,5.11,6.81,8.5,10.2,11.9,14.5,17,19.6` (repeated for the two replicate blocks; `data.csv`, row 1); this software version truncates them. That limitation, the short chains, and unavailable original chain settings make these diagnostic reanalyses, not the authors' actual fit.

## Interpretation

The significance-filtered-denominator hypothesis is not supported by Eq. S1_17: the formula normalizes over the stated genotype/allele-pair sum and includes `none`. The tiny `none` mean and unchanged independent counts under ε = 10−6 show that this is not the source of the few-percent mismatch on the recovered point-estimate input.

The mismatch is real for the recovered input and survives independent enumeration. The strongest available explanation is that the paper reports posterior-sample mean fractions, while the archive supplies one `median` value per sequence. This is plausible, not proven, because the authors' posterior vectors are unavailable and the fresh MCMC attempts failed convergence. The result is a source/aggregation limitation, not an arithmetic defect in GraphFLA.
