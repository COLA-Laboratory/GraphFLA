# CombiSEAL curated dataset — data card

## Source
Giddins et al., "Combinatorial protein engineering identifies potent CRISPR activators with reduced toxicity," *Nat. Commun.* **16**, 11114 (2025).
DOI: [10.1038/s41467-025-65986-4](https://doi.org/10.1038/s41467-025-65986-4).

Built from two Excel files on the Nature Communications page:
- `Supplementary_Data_4.xlsx` — high-throughput screen **activation** scores
- `Supplementary_Data_5.xlsx` — high-throughput screen **toxicity** scores

## Background in one paragraph
The authors assembled a library of synthetic CRISPR activators by chaining **activation domains (ADs)** onto the MS2 coat protein (MCP): MCP–AD1, MCP–AD1–AD2, or MCP–AD1–AD2–AD3. They picked 25 "high-value" ADs (a mix of VP16 variants, P65, HHV8_VIRF2, GB1, etc.) and built every possible ordered single / pair / triple. Each construct was delivered into HEK293T cells with a gRNA targeting one of three reporter genes (EPCAM, CXCR4, Reporter), sorted into 4 expression bins, and barcode-sequenced. From barcode distributions across bins they derived an **activation score** (how strong an activator) and a **toxicity score** (how much the construct depletes from the cell pool relative to the plasmid pool).

## Search space X
Every row is one ordered AD construct. There are three tiers (three separate CSV files) that share the same 25 AD identities:

| slot | values | meaning |
|---|---|---|
| `ad1` | `A01` .. `A25` | the AD at position 1 (N-terminal, closest to MCP) |
| `ad2` | `A01` .. `A25` (or empty for singles) | AD at position 2 |
| `ad3` | `A01` .. `A25` (or empty for singles/bipartite) | AD at position 3 |
| `ad*_idx` | 1..25 integer | same info as `ad*` but as int (convenient for ML) |

**Position matters** — the library is ordered with repetition, so `(A01, A02, A03)` and `(A03, A02, A01)` are distinct rows.

Tier sizes:
- `singles.csv`     — 25 rows  (X = ad1)
- `bipartite.csv`   — 625 rows (X = ad1 × ad2)
- `tripartite.csv`  — **15,625 rows** (X = ad1 × ad2 × ad3) — the 25³ benchmark
- `pooled_all.csv`  — 16,275 rows of all three tiers concatenated, with a `tier` column

## The 25 activation domains (A01 – A25)

Source: Supplementary Table 1 of the paper's main SI PDF. Within a domain name the convention is `<species/origin>_<protein>` and an optional trailing `xN` means **N tandem copies** of the same domain on one construct (tandem copies generally amplify activation).

### Grouped by biological role

**VP16 / VP64 family — historically the most potent CRISPR-activator domains (6 variants)**

| A-ID | Name | Brief |
|---|---|---|
| A01 | HHV1_VP16x2      | Human herpesvirus 1 VP16, two tandem copies. VP16 is the classic transactivator used in dCas9-VP64 and many other tools. |
| A02 | CercAHV2_VP16x2  | Cercopithecine (monkey) alphaherpesvirus 2 VP16, two tandem copies. |
| A05 | FBAHV1_VP64      | Fruit-bat alphaherpesvirus 1 VP64 (engineered tetramer of the minimal VP16 core). |
| A06 | CercAHV2_VP16    | Same monkey VP16 as A02 but single copy. |
| A09 | SaHV1_VP16       | Saimiriine (squirrel-monkey) alphaherpesvirus 1 VP16. |
| A19 | HHV1_VP64        | Classic engineered VP64 from HHV1 VP16 (the "standard" VP64 used in SAM's dCas9-VP64). |

**Other viral activation domains (non-VP16, 7 variants)**

| A-ID | Name | Brief |
|---|---|---|
| A04 | JMSFV_BEL1       | Foamy-virus (spumavirus) BEL1 / Tas transactivator. Binds cellular coactivators. |
| A07 | SFV3_BEL1        | Simian foamy virus 3 BEL1. Homolog of A04. |
| A08 | HHV8_VIRF2       | Kaposi's sarcoma-associated herpesvirus (KSHV) viral IRF2; binds cellular coactivators via distinct mechanism from VP16. |
| A11 | BAHV1_ICP4-11    | Bovine alphaherpesvirus 1 ICP4 fragment (ICP4 = infected-cell polypeptide 4, the master immediate-early transactivator of herpesviruses). |
| A12 | HHV8_ORF50       | KSHV ORF50, aka RTA, the replication and transcription activator of Kaposi's sarcoma-associated herpesvirus. |
| A15 | CervAHV2_ICP4    | Cervid alphaherpesvirus 2 ICP4. |
| A16 | HAVF41_E1A       | Human adenovirus type 41 E1A early oncoprotein (strong transactivator that recruits p300/CBP). |

**Human transcription-factor / coactivator domains (6 variants)**

| A-ID | Name | Brief |
|---|---|---|
| A03 | HS_MLLx3         | Human MLL (KMT2A) activation-domain fragment, three tandem copies. MLL recruits CBP/p300. |
| A10 | HS_C3ORF62       | Uncharacterized human C3orf62. Surfaced as a strong activator in the authors' 230-AD screen. |
| A13 | HS_FAM22F        | Uncharacterized human FAM22F family member. Surfaced in the initial screen. |
| A14 | HS_MASTR         | Human MASTR (MRTF-family myocardin-related coactivator). |
| A17 | HS_E2F2          | Human E2F2 (cell-cycle TF activation domain). |
| A18 | HS_CITED2x4      | Human CITED2 activation region, four tandem copies. CITED2 binds the CH1 domain of CBP/p300. |

**"Previously published tools" — included as within-set reference activators (3 variants)**

| A-ID | Name | Brief |
|---|---|---|
| A20 | HS_P65           | RelA / NF-κB p65 activation domain (the **P** in SAM = MCP-**P**65-HSF1). |
| A21 | EBV_RTA          | Epstein-Barr virus R-transactivator (the **R** in the tripartite VP64-p65-RTA "VPR" activator). |
| A22 | HS_HSF1          | Human heat-shock factor 1 activation domain (the **H** in SAM's MPH). |

**Protein-folding helpers — not activators, included to improve solubility / stability (3 variants)**

| A-ID | Name | Brief |
|---|---|---|
| A23 | SG_GB1           | Streptococcus G Ig-binding domain B1. Classic ~56-aa well-folded NMR / stability tag. |
| A24 | EC_MBP           | E. coli maltose-binding protein — a large, highly soluble fusion partner. |
| A25 | EC_TRXA          | E. coli thioredoxin — small, compact, reducing-friendly folding helper. |

### Using this with LLM-as-optimizer

For the LLM prompt, swap the `ad1`/`ad2`/`ad3` A-IDs for the real names in the table above. The LLM can then reason from its training data (e.g. "VP16 variants tend to be potent but toxic", "CITED2 tandems recruit p300", "GB1 is not an activator — if all three slots are PF, expect ~baseline activation"). Crucially, remind the LLM that A23 / A24 / A25 are NOT activators — otherwise it may waste budget on constructs with only folding helpers.

### AD shorthand notation

The paper uses single-letter shorthand for tripartite constructs in the style of MPH = MCP-P65-HSF1 (where M = MLL, P = P65, H = HSF1, V = VIRF2, etc.). Refer to Supplementary Data 8 of the paper for DNA-sequence-level details of individual constructs.

## Objectives f
| column | type | direction | notes |
|---|---|---|---|
| `activation_EPCAM`, `activation_CXCR4`, `activation_Reporter` | float | **maximize** | mean of 2 biological replicates (`*_rep1`, `*_rep2` also present) |
| `toxicity_EPCAM`, `toxicity_CXCR4`, `toxicity_avg` | float | **minimize** | depletion score; higher = more toxic |

Typical optimization framing: **multi-objective** — simultaneously maximize activation (on one or more targets) and minimize toxicity. The goal is to find constructs on the Pareto frontier.

## Caveats
- **Activation coverage is sparse in the tripartite tier**: only ~10% of tripartite constructs passed the read-count threshold needed to score them, so activation columns are NaN for ~90% of rows (10.1% EPCAM, 10.7% CXCR4, 7.9% Reporter). This is realistic for a partial-observation / active-learning benchmark; don't silently drop these rows without thinking.
- **Toxicity is dense** (100% coverage on all 15,625 tripartite rows). Toxicity is computable for every construct because it only needs construct read counts before vs after, not an activation-window threshold.
- **AD identities: A01..A25 map to real domains.** The mapping below comes from Supplementary Table 1 of the paper's main SI PDF. For a purely numerical optimizer the A-IDs are fine; an LLM-as-optimizer should be given the real names so it can leverage biological prior knowledge.
- **Construct scores are on different scales per target** (see the std columns in `describe()`); don't compare absolute numbers across EPCAM/CXCR4/Reporter. Rank-normalize first if you want a single objective.
- **Two biological replicates per score.** The "average" column is the mean of `rep1` and `rep2`; use the replicates if you want noise estimates.

## Files in this directory
```
domain_index.csv     # A01..A25 <-> integer 1..25 mapping
singles.csv          # 25 rows
bipartite.csv        # 625 rows
tripartite.csv       # 15,625 rows — the main benchmark
pooled_all.csv       # 16,275 rows (all tiers)
DATA_CARD.md         # this file
```
