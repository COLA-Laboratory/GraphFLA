# Papkou 2023 construction fixture

Source: Papkou et al., *A rugged yet easily navigable fitness landscape*,
Science 382, eadh3860. [Paper](https://doi.org/10.1126/science.adh3860);
[author archive](https://doi.org/10.5281/zenodo.8228920), CC BY 4.0.

- `fitness.csv.gz`: the unrounded `SV` and `m` columns of the authors'
  `in_data/fitness_data_wt.rds`, renamed to `sequence` and `fitness`.
- `edges.ncol.gz`: the authors' unmodified directed edge list,
  `computation_find_reciprocal_epistasis/graph_largest_component.ncol`.
- `manifest.json`: source and fixture SHA-256 digests.

The archived fitness table contains 261,333 measured variants. The paper
reports 261,382; these are different counts and must not be conflated.
The full archived input reproduces the reported giant component of 135,178
variants and 324,044 directed edges.

The authors' `lib/import_data.R` sets the cutoff to −0.507774.
`00.make_graph_object/01.script.ipynb` removes pairs with both fitnesses below
the cutoff, or with equal fitness, orients edges uphill, and extracts the
largest weak component. GraphFLA retains original fitness and edge differences;
the authors clip lethal fitness values when computing edge weights. This test
compares graph topology exactly and checks GraphFLA's weights against the
original measurements; it does not assert equality to the clipped weights.

Run `python -m validation.tests -k papkou_author_graph`. No downloads or R
installation are needed. Regenerate the fixtures from extracted archive
members with `python tools/prepare_papkou_fixture.py AUTHOR_DIRECTORY`
(requires `pyreadr` for the one-time conversion).
