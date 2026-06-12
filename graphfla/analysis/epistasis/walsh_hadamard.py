"""Walsh-Hadamard transform of the fitness landscape (tidy-DataFrame coefficients)."""

import copy
import math
import itertools

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging

logger = logging.getLogger(__name__)


def walsh_hadamard(landscape, max_order=2, max_cells=1e9, chunk_size=1000):
    """
    Compute Walsh-Hadamard coefficients for a fitness landscape.

    This function calculates Walsh-Hadamard coefficients for base and interaction terms
    up to a specified order using the ensemble encoding approach from the extended
    Walsh-Hadamard transform. The coefficients quantify the contribution of individual
    mutations and their interactions to the overall fitness.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object containing genotype-fitness data.
    max_order : int, default=2
        Maximum interaction order to consider. Higher orders capture more complex
        epistatic interactions but increase computational cost.
    max_cells : float, default=1e9
        Maximum matrix cells permitted to prevent excessive memory usage during
        interaction feature generation.
    chunk_size : int, default=1000
        Chunk size for H matrix construction to optimize memory usage for large datasets.

    Returns
    -------
    pandas.DataFrame
        One row per coefficient (tidy long form), with columns:

        - ``order`` (int): interaction order (0 = wildtype/intercept, 1 = single
          mutation, 2 = pairwise interaction, ...).
        - ``positions`` (tuple[int, ...]): the position indices involved (empty
          for the wildtype term).
        - ``term`` (str): the human-readable feature label. Single mutations use
          ``{original}_{position}_{mutant}`` (e.g. ``A_12_C``); higher-order
          interactions join mutations with ``-`` (e.g. ``A_10_C-A_11_G``).
        - ``coefficient`` (float): the fitted Walsh-Hadamard coefficient.

        Rows are sorted by ``(order, term)``. Filter an order with
        ``df[df["order"] == 2]`` rather than indexing a nested dict.

    Raises
    ------
    NotBuiltError
        If the landscape has not been built.
    ValueError
        If memory limit is exceeded during computation or if input data is invalid.

    Notes
    -----
    The Walsh-Hadamard transform provides a complete decomposition of the fitness
    function into additive and epistatic components. Higher-order coefficients
    represent increasingly complex epistatic interactions between mutations.

    Examples
    --------
    >>> # Assuming 'landscape' is a built Landscape object
    >>> coeffs = walsh_hadamard(landscape, max_order=3)
    >>> coeffs[coeffs["order"] == 1][["term", "coefficient"]]  # single mutations
    >>> coeffs.loc[coeffs["coefficient"].abs().idxmax()]       # strongest term
    """

    landscape._check_built()

    if landscape.graph is None or "fitness" not in landscape.graph.vs.attributes():
        raise ValueError(
            "Landscape graph or node 'fitness' attribute not found. "
            "Landscape must be built first."
        )

    data = landscape.get_data()
    X = data[list(landscape.data_types.keys())]
    f = data["fitness"].values

    # Walsh-Hadamard transform operates on one symbol per position, so encode
    # each landscape type to a per-position string.
    if landscape.kind in ["boolean"]:
        X_strings = ["".join(map(str, row.astype(int))) for _, row in X.iterrows()]
    elif landscape.kind in ["dna", "rna", "protein"]:
        if hasattr(landscape, "configs") and landscape.configs is not None:
            # reconstruct original sequences from encoded configs
            X_strings = []
            for config_tuple in landscape.configs.values:
                if landscape.kind == "dna":
                    alphabet = ["A", "C", "G", "T"]
                elif landscape.kind == "rna":
                    alphabet = ["A", "C", "G", "U"]
                else:  # protein
                    alphabet = list("ACDEFGHIKLMNPQRSTVWY")

                sequence = "".join([alphabet[int(pos)] for pos in config_tuple])
                X_strings.append(sequence)
        else:
            # fallback: treat as categorical
            X_strings = ["".join(map(str, row)) for _, row in X.iterrows()]
    else:
        # One symbol per variable so multi-digit codes don't split across positions;
        # +1 so no code collides with "0", which marks WT-matching positions downstream.
        codes = X.apply(lambda col: pd.factorize(col)[0] + 1).to_numpy()
        X_strings = ["".join(chr(48 + c) for c in row) for row in codes]

    if landscape.kind == "boolean":
        wildtype = "0" * landscape.n_vars
    else:
        wildtype = X_strings[0]  # first sequence is the reference WT

    wildtype_split = [c for c in wildtype]

    X_df = pd.DataFrame([list(seq) for seq in X_strings])

    enc = OneHotEncoder(
        handle_unknown="ignore", drop=np.array(wildtype_split), dtype=int
    )
    enc.fit(X_df)

    # Feature names "{original}_{position}_{mutant}", e.g. "0_12_1" (pos 12, 0->1).
    one_hot_names = []
    for i, feature_name in enumerate(enc.get_feature_names_out()):
        pos = int(feature_name.split("_")[0][1:])
        state = feature_name.split("_")[1]
        one_hot_names.append(f"{wildtype_split[pos]}_{pos+1}_{state}")

    Xoh = pd.DataFrame(enc.transform(X_df).toarray(), columns=one_hot_names)

    Xoh = pd.concat([pd.DataFrame({"WT": [1] * len(Xoh)}), Xoh], axis=1)

    Xohi = _generate_interactions(Xoh, max_order, max_cells)

    Xensemble = _ensemble_encode_features(
        X_strings, Xohi.columns, wildtype, X_df, chunk_size
    )

    # Direct least-squares solve, not normal equations: more stable and avoids
    # spurious macOS arm64 Accelerate matmul warnings from pinv(X^T X).
    Xensemble_values = Xensemble.to_numpy(dtype=np.float64, copy=False)
    coefficients, *_ = np.linalg.lstsq(
        Xensemble_values,
        np.asarray(f, dtype=np.float64),
        rcond=None,
    )

    rows = []
    for i, feature_name in enumerate(Xohi.columns):
        if feature_name == "WT":
            order = 0
            positions: tuple = ()
        else:
            components = feature_name.split("-")  # "-"-joined mutations
            order = len(components)
            # Each component encodes {original}_{position}_{mutant}; recover the
            # position index where the encoding matches (best-effort, never raises).
            pos = []
            for comp in components:
                parts = comp.split("_")
                if len(parts) == 3:
                    try:
                        pos.append(int(parts[1]))
                    except ValueError:
                        pass
            positions = tuple(pos)
        rows.append((order, positions, feature_name, float(coefficients[i])))

    return (
        pd.DataFrame(rows, columns=["order", "positions", "term", "coefficient"])
        .sort_values(["order", "term"], kind="stable")
        .reset_index(drop=True)
    )


def _generate_interactions(Xoh, max_order, max_cells):
    """Generate interaction features up to max_order with memory optimization."""
    if max_order < 2:
        return copy.deepcopy(Xoh)

    # observed mutations (drop WT and unobserved columns)
    mut_count = list(Xoh.sum(axis=0))
    pheno_mut = [
        Xoh.columns[i]
        for i in range(len(Xoh.columns))
        if mut_count[i] != 0 and Xoh.columns[i] != "WT"
    ]

    # group mutations by position; name format original_position_mutant ("0_12_1")
    def _get_position(mut_name):
        return mut_name.split("_")[1]

    all_pos = list(set([_get_position(i) for i in pheno_mut]))
    all_pos_mut = {p: [j for j in pheno_mut if _get_position(j) == p] for p in all_pos}

    # enumerate all theoretical interaction features
    all_features = {}
    int_order_dict = {}

    for n in range(2, max_order + 1):
        all_features[n] = []
        pos_comb = list(itertools.combinations(sorted(all_pos_mut.keys(), key=int), n))
        for p in pos_comb:
            all_features[n] += [
                "-".join(c) for c in itertools.product(*[all_pos_mut[j] for j in p])
            ]
        int_order_dict[n] = len(all_features[n])

    logger.info(
        "... Total theoretical features (order:count): "
        + ", ".join(
            [
                str(i) + ":" + str(int_order_dict[i])
                for i in sorted(int_order_dict.keys())
            ]
        )
    )

    all_features_flat = list(itertools.chain(*list(all_features.values())))

    int_list = []
    int_list_names = []
    int_order_dict_retained = {}

    for c in all_features_flat:
        c_split = c.split("-")  # "-"-joined mutations, e.g. "0_10_1-0_11_1"
        int_col = (Xoh.loc[:, c_split].sum(axis=1) == len(c_split)).astype(int)

        if sum(int_col) >= 0:  # min-observation threshold kept at 0 as in original
            int_list.append(int_col)
            int_list_names.append(c)

            order = len(c_split)
            if order not in int_order_dict_retained:
                int_order_dict_retained[order] = 1
            else:
                int_order_dict_retained[order] += 1

        if len(int_list) * len(Xoh) > max_cells:
            logger.info(
                f"Error: Too many interaction terms: number of feature matrix cells >{max_cells:>.0e}"
            )
            raise ValueError("Memory limit exceeded")

    logger.info(
        "... Total retained features (order:count): "
        + ", ".join(
            [
                str(i)
                + ":"
                + str(int_order_dict_retained[i])
                + " ("
                + str(round(int_order_dict_retained[i] / int_order_dict[i] * 100, 1))
                + "%)"
                for i in sorted(int_order_dict_retained.keys())
            ]
        )
    )

    if len(int_list) > 0:
        Xint = pd.concat(int_list, axis=1)
        Xint.columns = int_list_names
        # restore original feature ordering
        Xint = Xint.loc[:, [i for i in all_features_flat if i in Xint.columns]]
        Xohi = pd.concat([Xoh, Xint], axis=1)
    else:
        Xohi = copy.deepcopy(Xoh)

    return Xohi


def _ensemble_encode_features(X, feature_names, wildtype, X_df, chunk_size):
    """Ensemble encode features using Walsh-Hadamard transform with chunking optimization."""

    # mask each variant against WT: positions matching WT become "0"
    geno_list = []
    for seq in X:
        masked = "".join(x if x != y else "0" for x, y in zip(seq, wildtype))
        geno_list.append(masked)

    coef_list = [
        _coefficient_to_sequence(coef, len(wildtype)) for coef in feature_names
    ]

    # number of observed states per position
    state_counts = X_df.apply(lambda col: col.value_counts(), axis=0)
    state_list = [(state_counts[col] > 0).sum() for col in state_counts.columns]

    logger.info("Construction time for H_matrix...")
    hmat_inv = _H_matrix_chunker(
        str_geno=geno_list,
        str_coef=coef_list,
        num_states=state_list,
        invert=True,
        chunk_size=chunk_size,
    )

    vmat_inv = _V_matrix(str_coef=coef_list, num_states=state_list, invert=True)

    # V is diagonal: H @ V == column-scaling H, exactly. Avoids warning-prone
    # np.matmul on macOS arm64 Accelerate.
    return pd.DataFrame(hmat_inv * np.diag(vmat_inv), columns=feature_names)


def _coefficient_to_sequence(coefficient, length):
    """Convert coefficient string to sequence representation.

    Expects format "original_position_mutant" (e.g. "0_12_1") or multiple joined
    with "-" (e.g. "0_10_1-0_11_1" for pairwise).
    """
    coefficient_seq = ["0"] * length

    if coefficient == "WT":
        return "".join(coefficient_seq)

    for mut in coefficient.split("-"):
        parts = mut.split("_")
        if len(parts) >= 3:
            _orig, pos_str, state = parts[0], parts[1], parts[2]
            pos = int(pos_str) - 1  # 1-indexed to 0-indexed
            if 0 <= pos < length:
                coefficient_seq[pos] = state

    return "".join(coefficient_seq)


def _H_matrix_chunker(str_geno, str_coef, num_states=2, invert=False, chunk_size=1000):
    """Construct Walsh-Hadamard matrix in chunks (memory optimization)."""
    if len(str_geno) < chunk_size:
        return _H_matrix(str_geno, str_coef, num_states, invert)

    hmat_list = []
    for i in range(math.ceil(len(str_geno) / chunk_size)):
        from_i = i * chunk_size
        to_i = min((i + 1) * chunk_size, len(str_geno))
        hmat_list.append(_H_matrix(str_geno[from_i:to_i], str_coef, num_states, invert))

    return np.concatenate(hmat_list, axis=0)


def _H_matrix(str_geno, str_coef, num_states=2, invert=False):
    """Construct Walsh-Hadamard matrix."""
    string_length = len(str_geno[0])

    if isinstance(num_states, int):
        num_states = [float(num_states)] * string_length
    else:
        num_states = [float(i) for i in num_states]

    # to numeric (ord codes); "0" -> "." in coefs to mark unconstrained positions
    str_coef_num = [[ord(j) for j in i.replace("0", ".")] for i in str_coef]
    str_geno_num = [[ord(j) for j in i] for i in str_geno]

    num_statesi = np.repeat([num_states], len(str_geno) * len(str_coef), axis=0)
    str_genobi = np.repeat(str_geno_num, len(str_coef), axis=0)
    str_coefbi = np.transpose(
        np.tile(np.transpose(np.asarray(str_coef_num)), len(str_geno))
    )

    str_genobi_eq_str_coefbi = str_genobi == str_coefbi
    row_factor2 = str_genobi_eq_str_coefbi.sum(axis=1)

    if invert:
        row_factor1 = np.prod(str_genobi_eq_str_coefbi * (num_statesi - 2) + 1, axis=1)
        return (row_factor1 * np.power(-1, row_factor2) / np.prod(num_states)).reshape(
            (len(str_geno), -1)
        )
    else:
        row_factor1 = (
            np.logical_or(
                np.logical_or(str_genobi_eq_str_coefbi, str_genobi == ord("0")),
                str_coefbi == ord("."),
            ).sum(axis=1)
            == string_length
        ).astype(float)
        return (row_factor1 * np.power(-1, row_factor2)).reshape((len(str_geno), -1))


def _V_matrix(str_coef, num_states=2, invert=False):
    """Construct diagonal weighting matrix."""
    string_length = len(str_coef[0])

    if isinstance(num_states, int):
        num_states = [float(num_states)] * string_length
    else:
        num_states = [float(i) for i in num_states]

    str_coef_dot = [i.replace("0", ".") for i in str_coef]
    V = np.zeros((len(str_coef), len(str_coef)))

    for i in range(len(str_coef)):
        factor1 = int(
            np.prod(
                [
                    c
                    for a, b, c in zip(str_coef_dot[i], str_coef[i], num_states)
                    if ord(a) != ord(b)
                ]
            )
        )
        factor2 = sum(
            [1 for a, b in zip(str_coef_dot[i], str_coef[i]) if ord(a) == ord(b)]
        )

        if invert:
            V[i, i] = factor1 * np.power(-1, factor2)
        else:
            V[i, i] = 1 / (factor1 * np.power(-1, factor2))

    return V
