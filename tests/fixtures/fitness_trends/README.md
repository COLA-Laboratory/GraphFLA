# Fitness-trend literature input

`johnson2019.csv.gz` contains 10,645 measured mutation/background pairs for
91 experimental insertions, not synthetic genotype fitness. Keep each mutation
with at least 50 observations in the test; this yields 80 separate regressions.

Original study: Johnson, Martsul, Kryazhimskiy and Desai (2019),
[Higher-fitness yeast genotypes are less robust to deleterious mutations](https://doi.org/10.1126/science.aay4199).
The measurements are redistributed in Johnson and Desai (2022),
[eLife 11:e76491](https://doi.org/10.7554/eLife.76491), Supplementary file 1 v2,
sheets `BYxRM_x`, `BYxRM_s`, and `data_by_mutation` (`Type` selection only).
The eLife article and associated supplement are CC BY 4.0; publisher metadata
identifies the copyright holders as Johnson and Desai. Preserve this attribution.

`manifest.json` pins the source XLSX, source URL, license location, derived
fixture hash and conversion. Rebuild explicitly with:

```sh
python tools/prepare_fitness_trends_fixture.py SOURCE.xlsx tests/fixtures/fitness_trends
```

Tests do not download, open XLSX, regenerate expectations, or require openpyxl.
They parse the compact CSV with float round-trip precision. Source uses the
provided per-generation selection coefficient `s` and background `Fitness`
without further conversion. No duplicate mutation/sample measurements are
permitted. The source has 162 rows in BYxRM_x; the paper's recruited segregant
count is not substituted for the actual paired-data population.

Papkou validation reuses `../papkou2023/` and its original provenance/CC BY 4.0
attribution. Raw and floor-clipped fitness are different registered cases.
