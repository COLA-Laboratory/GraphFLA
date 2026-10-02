"""Dense reference equations, independent of GraphFLA's encoding and solver."""

import numpy as np


def additive_reference(design, objective):
    """Evaluate Szendro Eqs. (3)-(5) for an explicitly supplied coding."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(objective, dtype=float)
    # Center before solving to avoid cancellation between a large intercept
    # and the fitted contrasts. Keep the oracle on a dense independent path.
    xc, yc = x - x.mean(axis=0), y - y.mean()
    slopes, _, rank, _ = np.linalg.lstsq(xc, yc, rcond=None)
    coefficients = np.r_[y.mean() - np.dot(x.mean(axis=0), slopes), slopes]
    residuals = yc - np.einsum("ij,j->i", xc, slopes)
    roughness = float(np.sqrt(np.mean(residuals**2)))
    slope = float(np.mean(np.abs(coefficients[1:])))
    return {
        "coefficients": coefficients,
        "residuals": residuals,
        "rank": int(rank) + 1,
        "roughness": roughness,
        "slope": slope,
        "ratio": roughness / slope,
    }


def rna_design(sequences, reference):
    """Use an explicit RNA reference state at each of nine variable sites."""
    states = np.array([list(s) for s in sequences])
    retained = [state for state in "ACGU" if state != reference]
    return np.column_stack(
        [
            states[:, site] == state
            for site in range(states.shape[1])
            for state in retained
        ]
    ).astype(float)
