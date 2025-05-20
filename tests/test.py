import pytest
import pandas as pd
import random
from itertools import product
import numpy as np

from graphfla.analysis import *
from graphfla.landscape import (
    BooleanLandscape,
    DNALandscape,
    RNALandscape,
    ProteinLandscape,
    Landscape,
)
from graphfla.problems import NK


def generate_sequences(n, alphabets):
    sequences = ["".join(p) for p in product(alphabets, repeat=n)]
    return sequences


def generate_random_fitness(num_sequences):
    return [random.uniform(0, 100) for _ in range(num_sequences)]


# ------
# Fixtures
# ------


@pytest.fixture(scope="module")
def boolean_landscape_data():
    n = 4
    k = 2
    problem = NK(n, k)
    data = problem.get_data()
    X_list = data["config"].tolist()
    X = pd.DataFrame(X_list)
    fitness = data["fitness"]
    return X, fitness


@pytest.fixture(scope="module")
def boolean_landscape(boolean_landscape_data):
    X, fitness = boolean_landscape_data
    landscape = BooleanLandscape()
    landscape.build_from_data(X, fitness, verbose=False)
    return landscape


# DNA data and landscape
@pytest.fixture(scope="module")
def dna_sequence_data():
    n_seq = 2  # Sequence length
    sequences = generate_sequences(n_seq, alphabets=["A", "C", "G", "T"])
    fitness = generate_random_fitness(len(sequences))
    return sequences, fitness


@pytest.fixture(scope="module")
def dna_landscape(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    landscape = DNALandscape()
    landscape.build_from_data(sequences, fitness, verbose=False)
    return landscape


# RNA data and landscape
@pytest.fixture(scope="module")
def rna_sequence_data():
    n_seq = 2
    sequences = generate_sequences(n_seq, alphabets=["A", "C", "G", "U"])
    fitness = generate_random_fitness(len(sequences))
    return sequences, fitness


@pytest.fixture(scope="module")
def rna_landscape(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    landscape = RNALandscape()
    landscape.build_from_data(sequences, fitness, verbose=False)
    return landscape


# Protein data and landscape
PROTEIN_ALPHABETS_FULL = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
# Use a small subset for faster test execution
PROTEIN_ALPHABETS_SUBSET = PROTEIN_ALPHABETS_FULL[:4]


@pytest.fixture(scope="module")
def protein_sequence_data():
    n_seq = 3  # Keep it small for testing
    sequences = generate_sequences(n_seq, alphabets=PROTEIN_ALPHABETS_SUBSET)
    fitness = generate_random_fitness(len(sequences))
    return sequences, fitness


@pytest.fixture(scope="module")
def protein_landscape(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    landscape = ProteinLandscape()
    landscape.build_from_data(sequences, fitness, verbose=False)
    return landscape


# ------
# Landscape Construction Tests
# ------


def test_build_boolean_landscape(boolean_landscape_data):
    X, fitness = boolean_landscape_data
    landscape = BooleanLandscape()
    landscape.build_from_data(X, fitness, verbose=False)


def test_build_dna_landscape_from_list(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    landscape = DNALandscape()
    landscape.build_from_data(sequences, fitness, verbose=False)


def test_build_dna_landscape_from_df_int_cols(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    X_dna = pd.DataFrame([list(s) for s in sequences])
    landscape = DNALandscape()
    landscape.build_from_data(X_dna, fitness, verbose=False)


def test_build_dna_landscape_from_df_str_cols(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    X_dna = pd.DataFrame([list(s) for s in sequences])
    X_dna.columns = [f"pos_{i}" for i in range(X_dna.shape[1])]
    landscape = DNALandscape()
    landscape.build_from_data(X_dna, fitness, verbose=False)


def test_build_generic_landscape_dna(dna_sequence_data):
    sequences, fitness = dna_sequence_data
    landscape = Landscape(type="dna")
    landscape.build_from_data(sequences, fitness, verbose=False)


def test_build_rna_landscape_from_list(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    landscape = RNALandscape()
    landscape.build_from_data(sequences, fitness, verbose=False)


def test_build_rna_landscape_from_df_int_cols(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    X_rna = pd.DataFrame([list(s) for s in sequences])
    landscape = RNALandscape()
    landscape.build_from_data(X_rna, fitness, verbose=False)


def test_build_rna_landscape_from_df_str_cols(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    X_rna = pd.DataFrame([list(s) for s in sequences])
    X_rna.columns = [f"pos_{i}" for i in range(X_rna.shape[1])]
    landscape = RNALandscape()
    landscape.build_from_data(X_rna, fitness, verbose=False)


def test_build_generic_landscape_rna(rna_sequence_data):
    sequences, fitness = rna_sequence_data
    landscape = Landscape(type="rna")
    landscape.build_from_data(sequences, fitness, verbose=False)


def test_build_protein_landscape_from_list(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    landscape = ProteinLandscape()
    landscape.build_from_data(sequences, fitness, verbose=False)


def test_build_protein_landscape_from_df_int_cols(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    X_protein = pd.DataFrame([list(s) for s in sequences])
    landscape = ProteinLandscape()
    landscape.build_from_data(X_protein, fitness, verbose=False)


def test_build_protein_landscape_from_df_str_cols(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    X_protein = pd.DataFrame([list(s) for s in sequences])
    X_protein.columns = [f"pos_{i}" for i in range(X_protein.shape[1])]
    landscape = ProteinLandscape()
    landscape.build_from_data(X_protein, fitness, verbose=False)


def test_build_generic_landscape_protein(protein_sequence_data):
    sequences, fitness = protein_sequence_data
    landscape = Landscape(type="protein")
    landscape.build_from_data(sequences, fitness, verbose=False)


# ------
# Feature Calculation Tests (Boolean Landscape)
# ------


def test_feature_mag_epistasis(boolean_landscape):
    result = classify_epistasis(boolean_landscape)
    assert "magnitude epistasis" in result


def test_feature_sign_epistasis(boolean_landscape):
    result = classify_epistasis(boolean_landscape)
    assert "sign epistasis" in result


def test_feature_reciprocal_sign_epistasis(boolean_landscape):
    result = classify_epistasis(boolean_landscape)
    assert "reciprocal sign epistasis" in result


def test_feature_positive_epistasis(boolean_landscape):
    result = classify_epistasis(boolean_landscape)
    assert "positive epistasis" in result


def test_feature_negative_epistasis(boolean_landscape):
    result = classify_epistasis(boolean_landscape)
    assert "negative epistasis" in result


def test_feature_idiosyncrasy(boolean_landscape):
    result = global_idiosyncratic_index(boolean_landscape)
    assert "global_index" in result


def test_feature_diminishing_returns_corr(boolean_landscape):
    result = diminishing_returns_index(boolean_landscape, method="spearman")
    assert isinstance(result, (float, np.floating))  # Can be np.nan which is float


def test_feature_diminishing_returns_regr(boolean_landscape):
    result = diminishing_returns_index(boolean_landscape, method="regression")
    assert isinstance(result, (float, np.floating))


def test_feature_increasing_costs_corr(boolean_landscape):
    result = increasing_costs_index(boolean_landscape, method="spearman")
    assert isinstance(result, (float, np.floating))


def test_feature_increasing_costs_regr(boolean_landscape):
    result = increasing_costs_index(boolean_landscape, method="regression")
    assert isinstance(result, (float, np.floating))


def test_feature_gamma(boolean_landscape):
    result = gamma_statistic(boolean_landscape)
    assert "gamma" in result


def test_feature_gamma_star(boolean_landscape):
    result = gamma_statistic(boolean_landscape)
    assert "gamma_star" in result


def test_feature_epistasis_2nd(boolean_landscape):
    result = higher_order_epistasis(boolean_landscape, order=2)
    assert isinstance(result, (float, np.floating))


def test_feature_epistasis_3rd(boolean_landscape):
    # Boolean landscape n=4, so order=3 is possible.
    result = higher_order_epistasis(boolean_landscape, order=3)
    assert isinstance(result, (float, np.floating))


def test_feature_nfc(boolean_landscape):
    result = neighbor_fit_corr(boolean_landscape)
    assert isinstance(result, (float, np.floating))


def test_feature_fdc(boolean_landscape):
    result = fitness_distance_corr(boolean_landscape)
    assert isinstance(result, (float, np.floating))


def test_feature_bfc_greedy(boolean_landscape):
    result = basin_fit_corr(boolean_landscape)
    assert "greedy" in result


def test_feature_bfc_accessible(boolean_landscape):
    result = basin_fit_corr(boolean_landscape)
    assert "accessible" in result


def test_feature_go_accessibility(boolean_landscape):
    result = global_optima_accessibility(boolean_landscape)
    assert isinstance(result, (float, np.floating))


def test_feature_ee_frac(boolean_landscape):
    result = calculate_evol_enhance(boolean_landscape)
    assert "ee_proportion" in result


def test_feature_neutral_frac(boolean_landscape):
    result = neutrality(boolean_landscape)
    assert isinstance(result, (float, np.floating))


def test_feature_peaks_frac(boolean_landscape):  # lo_ratio
    result = lo_ratio(boolean_landscape)
    assert isinstance(result, (float, np.floating))


def test_feature_autocorr(boolean_landscape):
    # For N=4, 2^4 = 16 nodes. Walk length 5 is fine.
    result = autocorrelation(boolean_landscape, walk_length=5)
    assert isinstance(result, (float, np.floating))


def test_feature_r_s_ratio(boolean_landscape):
    result = r_s_ratio(boolean_landscape)
    assert isinstance(result, (float, np.floating))


def test_feature_skewness(boolean_landscape):
    result = fitness_distribution(boolean_landscape)
    assert "skewness" in result


def test_feature_kurtosis(boolean_landscape):
    result = fitness_distribution(boolean_landscape)
    assert "kurtosis" in result


def test_feature_cv(boolean_landscape):
    result = fitness_distribution(boolean_landscape)
    assert "cv" in result


def test_feature_quartile_coefficient(boolean_landscape):
    result = fitness_distribution(boolean_landscape)
    assert "quartile_coefficient" in result


def test_feature_median_mean_ratio(boolean_landscape):
    result = fitness_distribution(boolean_landscape)
    assert "median_mean_ratio" in result


def test_feature_relative_range(boolean_landscape):
    result = fitness_distribution(boolean_landscape)
    assert "relative_range" in result


def test_feature_cauchy_loc(boolean_landscape):
    result = fitness_distribution(boolean_landscape)
    assert "cauchy_loc" in result


# ------
# Feature Calculation Tests (DNA, RNA, Protein Landscapes)
# ------


# DNA Landscape Features
def test_dna_feature_epistasis(dna_landscape):
    result = classify_epistasis(dna_landscape)
    assert "magnitude epistasis" in result
    assert "sign epistasis" in result


def test_dna_feature_fitness_dist(dna_landscape):
    result = fitness_distribution(dna_landscape)
    assert "skewness" in result
    assert "cv" in result


def test_dna_feature_nfc(dna_landscape):
    result = neighbor_fit_corr(dna_landscape)
    assert isinstance(result, (float, np.floating))


# RNA Landscape Features
def test_rna_feature_epistasis(rna_landscape):
    result = classify_epistasis(rna_landscape)
    assert "magnitude epistasis" in result
    assert "sign epistasis" in result


def test_rna_feature_fitness_dist(rna_landscape):
    result = fitness_distribution(rna_landscape)
    assert "skewness" in result
    assert "cv" in result


def test_rna_feature_nfc(rna_landscape):
    result = neighbor_fit_corr(rna_landscape)


# Protein Landscape Features
def test_protein_feature_epistasis(protein_landscape):
    result = classify_epistasis(protein_landscape)
    assert "magnitude epistasis" in result
    assert "sign epistasis" in result


def test_protein_feature_fitness_dist(protein_landscape):
    result = fitness_distribution(protein_landscape)
    assert "skewness" in result
    assert "cv" in result


def test_protein_feature_nfc(protein_landscape):
    result = neighbor_fit_corr(protein_landscape)


# Add a few more diverse features for DNA/RNA/Protein
def test_dna_feature_lo_ratio(dna_landscape):
    result = lo_ratio(dna_landscape)


def test_rna_feature_neutrality(rna_landscape):
    result = neutrality(rna_landscape)


def test_protein_feature_gamma_statistic(protein_landscape):
    result = gamma_statistic(protein_landscape)
    assert "gamma" in result
    assert "gamma_star" in result


# Test higher order epistasis on sequence landscapes (if n_genes >= order)
# For sequence landscapes (DNA, RNA, Protein), n_genes is 2 from fixtures.
# So, order=2 is testable. order=3 would not be for n_genes=2.
def test_dna_feature_epistasis_2nd(dna_landscape):
    result = higher_order_epistasis(dna_landscape, order=2)


def test_rna_feature_epistasis_2nd(rna_landscape):
    result = higher_order_epistasis(rna_landscape, order=2)


def test_protein_feature_epistasis_2nd(protein_landscape):
    result = higher_order_epistasis(protein_landscape, order=2)
