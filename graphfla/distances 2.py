import numpy as np
from typing import Dict


def euclidean_distance(
    X: np.ndarray,
    x: np.ndarray,
    data_types: None,
) -> np.ndarray:
    """
    Compute the Euclidean distance between each row of X and the reference vector x.

    Parameters
    ----------
    X : np.ndarray
        A 2D array containing numerical features.
    x : np.ndarray
        A 1D reference vector.

    Returns
    -------
    np.ndarray
        A 1D array of Euclidean distances.
    """
    return np.sqrt(np.sum((X - x) ** 2, axis=1))


def hamming_distance(X: np.ndarray, x: np.ndarray, data_types: None) -> np.ndarray:
    """
    Compute the Hamming distance between each row of X and the reference vector x.

    Parameters
    ----------
    X : np.ndarray
        A 2D array containing categorical or boolean features.
    x : np.ndarray
        A 1D reference vector.

    Returns
    -------
    np.ndarray
        A 1D array of Hamming distances.
    """
    return np.sum(X != x, axis=1)


def manhattan_distance(X: np.ndarray, x: np.ndarray, data_types: None) -> np.ndarray:
    """
    Compute the Manhattan distance between each row of X and the reference vector x.

    Parameters
    ----------
    X : np.ndarray
        A 2D array containing ordinal features.
    x : np.ndarray
        A 1D reference vector.

    Returns
    -------
    np.ndarray
        A 1D array of Manhattan distances.
    """
    return np.sum(np.abs(X - x), axis=1)


def mixed_distance(
    X: np.ndarray,
    x: np.ndarray,
    data_types: Dict[int, str],
) -> np.ndarray:
    """
    Calculate the mixed distance between each row of matrix `X` and a reference vector `x`.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array where each row represents an instance and columns correspond to variables.
        Shape: (n_samples, n_features)

    x : np.ndarray
        A 1D numpy array representing the reference vector, containing values for each feature
        in the dataset. Should match the number of features (columns) in `X`.
        Shape: (n_features,)

    data_types : Dict[int, str]
        A dictionary mapping column indices in `X` and `x` to their respective data types
        ('categorical', 'boolean', 'ordinal').

    Returns
    -------
    np.ndarray
        A 1D numpy array of distances between each row in `X` and the `x`.
        Shape: (n_samples,)
    """
    total_distance = np.zeros(X.shape[0])
    data_types = {i: value for i, value in enumerate(data_types.values())}

    cat_indices = [
        i for i, dtype in data_types.items() if dtype in {"categorical", "boolean"}
    ]
    ord_indices = [i for i, dtype in data_types.items() if dtype == "ordinal"]

    if cat_indices:
        X_cat = X[:, cat_indices]
        x_cat = x[cat_indices]
        total_distance += hamming_distance(X_cat, x_cat, None)

    if ord_indices:
        X_ord = X[:, ord_indices]
        x_ord = x[ord_indices]
        total_distance += manhattan_distance(X_ord, x_ord, None)

    return total_distance
