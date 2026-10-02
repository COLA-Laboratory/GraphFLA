"""Published empirical results with fixed preprocessing and data fingerprints."""

from pathlib import Path

import pandas as pd
import pytest

from graphfla.analysis import classify_epistasis, higher_order_epistasis
from graphfla.landscape import BooleanLandscape, DNALandscape, ProteinLandscape
from validation.contract import load_definitions, matches, verify_artifact


REPO = Path(__file__).resolve().parents[2]
CASES = load_definitions(REPO / "validation")[2]


def _input(case_id):
    [artifact] = CASES[case_id]["inputs"]
    assert artifact["root"] == "repo"
    return verify_artifact(REPO, artifact)


def _assert_published(case_id, observed):
    case = CASES[case_id]
    assert matches(case["expected"], observed, case["comparison"]), (
        case_id,
        case["expected"],
        observed,
    )


@pytest.fixture(scope="module")
def papkou():
    filename = _input("papkou.topology.v1")
    frame = pd.read_csv(filename, usecols=["sequences", "fitness"])
    assert len(frame) == 135178
    assert not frame.isna().any().any()
    # The curated file was decoded with a different nucleotide alphabet order.
    # This inverse map restores biological labels: all 135178 sequence/fitness
    # pairs then match the largest component reconstructed from source-backed RAW.
    # Relabeling preserves Hamming adjacency; the original CSV is left intact.
    frame.sequences = frame.sequences.str.translate(
        str.maketrans({"A": "A", "C": "G", "G": "T", "T": "C"})
    )
    # Science 382, eadh3860 (2023), supplementary methods pp. 14–15.
    # The decoded file contains the published component's vertices. Reapply
    # the edge filter: rebuilding every Hamming-1 edge gives a different graph.
    return DNALandscape().build_from_data(
        frame.sequences,
        frame.fitness,
        tau=-0.507774,
        filter_mode="both",
        epsilon=0,
        verbose=False,
    )


@pytest.mark.literature_case("papkou.topology.v1", role="paper_result")
def test_papkou_published_network_and_peaks(papkou):
    _assert_published(
        "papkou.topology.v1",
        {
            "vertices": papkou.n_configs,
            "edges": papkou.n_edges,
            "peaks": papkou.n_lo,
            "functional": int(
                sum(value >= -0.507774 for value in papkou.graph.vs["fitness"])
            ),
        },
    )
    assert papkou.graph.is_connected(mode="weak")


@pytest.mark.literature_case("papkou.topology.v1", role="input_check")
def test_papkou_edge_filter_and_alignment(papkou):
    fitness = papkou.graph.vs["fitness"]
    variants = papkou._configs_array
    for edge in papkou.graph.es:
        i, j = edge.tuple
        assert fitness[j] > fitness[i]
        assert fitness[j] >= -0.507774
        assert edge["delta_fit"] == pytest.approx(fitness[j] - fitness[i])
        assert sum(variants[i] != variants[j]) == 1


@pytest.mark.literature_case("papkou.classification.v1", role="author_result")
def test_papkou_author_epistasis_classification(papkou):
    """Exact class fractions from the authors' archived Figure S20 notebook.

    Zenodo 8228920, code_analyses_figures.zip:
    14.reciprocal_sign_epistasis/01.count_different_types_of_epistasis.ipynb.
    Saved author output gives 408065 / 246943 / 85203 motifs. The published
    SI caption p. 40 independently states total 740211 and reciprocal 85203.
    Additive cases are included in the magnitude/no-sign class.
    """
    result = classify_epistasis(papkou, sample_cut_prob=0, seed=0)
    _assert_published(
        "papkou.classification.v1",
        {
            category: getattr(result, category)
            for category in ("magnitude", "sign", "reciprocal_sign")
        },
    )


@pytest.mark.literature_case("westmann.peaks.v1", role="paper_result")
def test_westmann_published_peak_count():
    """Nature Communications 15, 10745 (2024), supplementary Table S1."""
    filename = _input("westmann.peaks.v1")
    frame = pd.read_csv(filename, usecols=["sequences", "fitness"])
    landscape = DNALandscape().build_from_data(
        frame.sequences, frame.fitness, epsilon=0, verbose=False
    )
    # Repression is normalized to the wild type in the publisher source data.
    peaks = landscape.get_data(lo_only=True)
    _assert_published(
        "westmann.peaks.v1",
        {
            "vertices": landscape.n_configs,
            "peaks": landscape.n_lo,
            "above_wt": int(sum(peaks.fitness > 1)),
            "below_wt": int(sum(peaks.fitness < 1)),
        },
    )


@pytest.mark.parametrize(
    "antigen,order",
    [pytest.param(a, o, marks=pytest.mark.literature_case(
        f"phillips.cr6261.{a}.order{o}.v1", role="author_result"
    )) for a, o in [("h1", 1), ("h1", 2), ("h9", 1), ("h9", 2)]],
)
def test_phillips_author_regression_outputs(antigen, order):
    """Reproduce full-data OLS outputs, not held-out cross-validation scores.

    Phillips et al., eLife 10:e71393 (2021), Methods pp.21–22.
    klawrence26/bnab-landscapes, commit
    514d62f387070f43b8a172f02e60c70137f8a8b8:
    CR6261/Epistasis_linear_models/model_coefs/H{1,9}_{1,2}order_stat.txt.
    """
    case_id = f"phillips.cr6261.{antigen}.order{order}.v1"
    filename = _input(case_id)
    # Nonmissing genotype/fitness pairs exactly match the authors' filtered input.
    frame = pd.read_csv(filename, dtype={"sequences": str}).dropna(subset=["fitness"])
    assert len(frame) == {"h1": 1887, "h9": 1842}[antigen]
    landscape = BooleanLandscape().build_from_data(
        frame.sequences, frame.fitness, verbose=False
    )
    _assert_published(case_id, float(higher_order_epistasis(landscape, order=order)))


@pytest.mark.literature_case("bank.reia_peaks.v1", role="paper_result")
def test_bank_archived_landscape_reproduces_reia_peak_identities():
    """Reia and Campos (2020), Figure 1: six named peaks in Bank's landscape.

    Dryad 10.5061/dryad.41ns1rn9r, Empirical_Landscapes.zip. The authors
    acknowledge Bank for providing these fitness values. This source is
    separate from the project's Bank2016a.csv, which contains read counts.
    """
    filename = _input("bank.reia_peaks.v1")
    frame = pd.read_csv(filename, sep="\t")
    # The archive also contains a stop-codon control outside the 640 genotypes.
    assert frame.loc[
        frame.sequence.str.contains("*", regex=False), "sequence"
    ].tolist() == ["Q*GWSANME"]
    frame = frame.loc[~frame.sequence.str.contains("*", regex=False)]
    assert len(frame) == 640
    landscape = ProteinLandscape().build_from_data(
        frame.sequence, frame.fitness, epsilon=0, verbose=False
    )
    _assert_published("bank.reia_peaks.v1", landscape.n_lo)
    # get_data indexes input rows, so verify alignment before using its mask.
    data = landscape.get_data()
    assert data.fitness.tolist() == frame.fitness.tolist()
    peaks = frame.sequence.to_numpy()[data.is_lo.to_numpy()]
    assert set(peaks) == {
        "QFGWTPAME",
        "QFGLTALME",
        "QFGFSALTE",
        "QFGLSPLAE",
        "QFGLTPAQE",
        "QFGISALQE",
    }


def _complete_protein_fixture(case_id):
    filename = _input(case_id)
    # Minimal source columns preserve every float64 value, including imputation.
    frame = pd.read_csv(filename, float_precision="round_trip")
    assert len(frame) == 160000
    assert frame.sequence.is_unique
    assert not frame.isna().any().any()
    landscape = ProteinLandscape().build_from_data(
        frame.sequence, frame.fitness, epsilon=0, verbose=False
    )
    data = landscape.get_data()
    assert data.fitness.tolist() == frame.fitness.tolist()
    sequences = data[[f"pos_{i}" for i in range(4)]].agg("".join, axis=1)
    assert sequences.tolist() == frame.sequence.tolist()
    return frame, landscape


@pytest.mark.literature_case("wu.peaks.v1", "wu.accessibility.v1", role="paper_result")
def test_wu_completed_landscape_peaks_and_accessibility():
    """eLife 16965: 30 peaks, 15 above WT (p. 9 / Fig. 4A).

    Access to all 15 is 93% (p. 11 / Figure 4—figure supplement 2C).

    Published Datasets 1 and 2 supply measured and imputed values. The final
    assertion is a graph-based intersection of reachability sets; GraphFLA's
    per-peak accessibility function alone does not return that intersection.
    """
    frame, landscape = _complete_protein_fixture("wu.peaks.v1")
    assert int(frame.imputed.sum()) == 10639
    assert frame.loc[frame.sequence == "VDGV", "fitness"].item() == 1
    peaks = landscape.get_data(lo_only=True)
    high_peaks = peaks.index[peaks.fitness > 1].tolist()
    _assert_published(
        "wu.peaks.v1", {"peaks": landscape.n_lo, "above_wt": len(high_peaks)}
    )
    assert all(delta > 0 for delta in landscape.graph.es["delta_fit"])
    reachable = set(range(landscape.n_configs))
    for peak in high_peaks:
        reachable.intersection_update(landscape.graph.subcomponent(peak, mode="in"))
    # The paper reports a whole percentage: half a percentage point is the
    # allowed rounding error. Do not freeze this implementation's exact count.
    _assert_published("wu.accessibility.v1", len(reachable) / landscape.n_configs)


@pytest.mark.literature_case("johnston.active_peaks.v1", role="paper_result")
def test_johnston_peaks_among_active_measured_candidates():
    """PNAS 2400439121: 520 peaks among 9,783 active measured TrpB variants.

    Candidates use the authors' activity mask. Neighbors include all measured
    and imputed variants; building only the active subset changes the question.
    """
    frame, landscape = _complete_protein_fixture("johnston.active_peaks.v1")
    assert int(frame.imputed.sum()) == 871
    candidates = frame.active & ~frame.imputed
    assert int(candidates.sum()) == 9783
    assert not (frame.active & frame.imputed).any()
    peaks = landscape.get_data().is_lo.to_numpy()
    _assert_published(
        "johnston.active_peaks.v1", int((peaks & candidates.to_numpy()).sum())
    )
