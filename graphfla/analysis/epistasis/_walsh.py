"""Shared fitting and order summaries for the public Walsh-Hadamard analysis."""

from itertools import combinations, product
from numbers import Integral, Real
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.utils import Bunch, check_random_state


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")


def _validate_options(max_order, max_cells, chunk_size, method, alpha, max_iter, tol):
    _positive_integer(max_order, "max_order", minimum=0)
    _positive_integer(chunk_size, "chunk_size")
    if (
        isinstance(max_cells, bool)
        or not isinstance(max_cells, Real)
        or not np.isfinite(max_cells)
        or max_cells < 1
    ):
        raise ValueError(f"max_cells must be finite and >= 1; got {max_cells!r}.")
    if method not in ("ols", "lasso"):
        raise ValueError(f"method must be 'ols' or 'lasso'; got {method!r}.")
    if method == "lasso":
        if not (isinstance(alpha, str) and alpha == "cv") and (
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not np.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError(
                f"alpha must be 'cv' or a positive finite float; got {alpha!r}."
            )
        _positive_integer(max_iter, "max_iter")
        if (
            isinstance(tol, bool)
            or not isinstance(tol, Real)
            or not np.isfinite(tol)
            or tol <= 0
        ):
            raise ValueError(f"tol must be a positive finite float; got {tol!r}.")


def _encode_input(landscape):
    """Retain original labels while using integer states for all arithmetic."""
    data = landscape.get_data()
    if landscape.data_types is None or "fitness" not in data:
        raise ValueError("Configuration columns and a fitness column are required.")
    X = data[list(landscape.data_types)]
    y = data["fitness"].to_numpy(dtype=float)
    if not len(y):
        raise ValueError("At least one observation is required; got n_samples=0.")
    if not np.isfinite(y).all():
        raise ValueError("Fitness values must be finite.")
    if X.isna().any().any():
        raise ValueError("Configuration values must not be missing.")
    if X.duplicated().any():
        raise ValueError("Configurations must be unique; aggregate replicates first.")
    codes, alleles, positions = [], [], []
    for col in X:
        c, states = pd.factorize(X[col], sort=True)
        ref = (
            0
            if landscape.data_types[col] == "boolean" and 0 in states
            else X[col].iloc[0]
        )
        r = states.get_loc(ref)
        permutation = [r] + [j for j in range(len(states)) if j != r]
        inverse = np.argsort(permutation)
        codes.append(inverse[c])
        alleles.append(states.take(permutation).to_numpy())
        # Native sequence/Boolean column names encode positions even when an
        # imported graph omits invariant columns. General frames retain their
        # original feature order in get_data(), including invariant columns.
        prefix = (
            "pos_"
            if landscape.kind in {"dna", "rna", "protein"}
            else "bit_"
            if landscape.kind == "boolean"
            else None
        )
        if (
            prefix
            and isinstance(col, str)
            and col.startswith(prefix)
            and col[len(prefix) :].isdigit()
        ):
            positions.append(int(col[len(prefix) :]) + 1)
        else:
            positions.append(int(data.columns.get_loc(col)) + 1)
    matrix = np.column_stack(codes) if codes else np.empty((len(y), 0), dtype=int)
    return X, y, matrix, alleles, positions


def _term_count(arities, max_order):
    """Count coefficients by degree before constructing any term objects."""
    counts = [1] + [0] * max_order
    for arity in arities:
        for order in range(max_order, 0, -1):
            counts[order] += counts[order - 1] * (arity - 1)
    return sum(counts)


def _design_matrix(codes, arities, terms, chunk_size):
    """Factor H^-1 V^-1 into local centered indicators (Faure Eq. 12).

    For a focal state a at site j the local factor is 1[x_j=a] - 1/s_j.
    Factoring cancels the full-space normalization before floating-point
    arithmetic, so long, low-order models need no product of all state counts.
    """
    design = np.ones((len(codes), len(terms)), dtype=float, order="F")
    for start in range(0, len(codes), chunk_size):
        stop = start + chunk_size
        for k, term in enumerate(terms[1:], 1):
            for j, allele in term:
                design[start:stop, k] *= (codes[start:stop, j] == allele) - 1 / arities[
                    j
                ]
    return design


def _raise_rank(n_samples, n_terms, rank=None):
    status = "underdetermined" if rank is None else f"rank-deficient (rank={rank})"
    raise ValueError(
        f"OLS design is {status}: n_samples={n_samples}, n_terms={n_terms}. "
        "Reduce max_order, provide additional independent observations, "
        "or use method='lasso' for a regularized estimate."
    )


def _lasso_fit(
    design,
    target,
    scale,
    alpha,
    cv,
    random_state,
    max_iter,
    tol,
    *,
    splits=None,
    n_jobs=1,
):
    """Fit an unpenalized constant and penalized W-H coefficients."""
    if design.shape[1] == 1:
        return np.array([target.mean()]), 0.0 if alpha == "cv" else float(alpha)
    features = design[:, 1:]
    options = dict(fit_intercept=True, max_iter=max_iter, tol=tol, precompute=False)
    if alpha == "cv":
        centered = target - target.mean()
        alpha_max = float(np.max(np.abs(features.T @ centered)) / len(target))
        if alpha_max == 0:
            coef = np.zeros(design.shape[1])
            coef[0] = target.mean()
            return coef, 0.0
        alphas = alpha_max * np.geomspace(1.0, 1e-3, 100)
        folds = (
            splits
            if splits is not None
            else KFold(n_splits=cv, shuffle=True, random_state=random_state)
        )
        model = LassoCV(alphas=alphas, cv=folds, n_jobs=n_jobs, **options)
    else:
        normalized_alpha = float(alpha) / scale
        if not np.isfinite(normalized_alpha):
            coef = np.zeros(design.shape[1])
            coef[0] = target.mean()
            return coef, float(alpha)
        if normalized_alpha == 0:
            raise ValueError(
                "alpha is too small relative to fitness; rescale fitness or increase alpha."
            )
        model = Lasso(alpha=normalized_alpha, **options)
    model.fit(features, target)
    selected = float(model.alpha_ * scale) if alpha == "cv" else float(alpha)
    if not np.isfinite(selected):
        raise ValueError("Selected alpha exceeds the float64 range; rescale fitness.")
    return np.r_[model.intercept_, model.coef_], selected


def _allele_labels(alleles):
    labels = [
        str(int(a)) if isinstance(a, (bool, np.bool_)) else str(a) for a in alleles
    ]
    # Mixed-type categories can share a textual spelling, e.g. 1 and "1".
    # Disambiguate only those spellings; labels never drive computation.
    duplicates = {s for s in labels if labels.count(s) > 1}
    labels = [
        f"{type(a).__name__}:{s}" if s in duplicates else s
        for a, s in zip(alleles, labels)
    ]
    return [
        s.replace("%", "%25").replace("_", "%5F").replace("-", "%2D") for s in labels
    ]


def _analyze(
    landscape,
    max_order=2,
    max_cells=1e7,
    chunk_size=1000,
    *,
    method="ols",
    alpha="cv",
    cv=5,
    random_state=0,
    max_iter=10000,
    tol=1e-4,
    n_jobs=1,
    require_coefficients=True,
):
    """Build once, fit each order once, and keep the final coefficient vector."""
    landscape._check_built()
    _validate_options(max_order, max_cells, chunk_size, method, alpha, max_iter, tol)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError(f"n_jobs must be a nonzero integer; got {n_jobs!r}.")
    X, y, codes, alleles, positions = _encode_input(landscape)
    arities = [len(a) for a in alleles]
    active_sites = tuple(j for j, s in enumerate(arities) if s > 1)
    order = min(int(max_order), len(active_sites))
    n_terms, n_samples = _term_count(arities, order), len(y)
    if n_samples * n_terms > max_cells:
        raise ValueError(
            f"Design matrix requires {n_samples * n_terms} cells "
            f"(n_samples={n_samples}, n_terms={n_terms}), exceeding "
            f"max_cells={max_cells:g}. Reduce max_order or increase max_cells."
        )
    if require_coefficients and method == "ols" and n_terms > n_samples:
        _raise_rank(n_samples, n_terms)
    splits = None
    if method == "lasso" and alpha == "cv" and n_terms > 1:
        _positive_integer(cv, "cv", minimum=2)
        if cv > n_samples:
            raise ValueError(f"cv={cv} cannot exceed n_samples={n_samples}.")
        check_random_state(random_state)
        # Materialize once so even random_state=None/RandomState instances use
        # the same held-out observations at every candidate interaction order.
        splits = list(KFold(cv, shuffle=True, random_state=random_state).split(y))

    terms = [()]
    widths = [1]
    for degree in range(1, order + 1):
        for sites in combinations(active_sites, degree):
            terms.extend(
                tuple(zip(sites, states))
                for states in product(*(range(1, arities[j]) for j in sites))
            )
        widths.append(len(terms))
    design = _design_matrix(codes, arities, terms, int(chunk_size))
    scale = float(np.max(np.abs(y))) or 1.0
    target = y / scale
    center = float(target.mean())
    centered = target - center
    sst = float(np.dot(centered, centered))
    if sst == 0:
        warnings.warn(
            "Fitness is constant; R-squared and variance fractions are undefined (NaN).",
            UserWarning,
            stacklevel=3,
        )
    fit_design, fit_target = design, centered
    if method == "ols" and n_terms > 1 and n_samples > 2 * (n_terms + 1):
        # A single augmented QR preserves every prefix least-squares problem
        # and its residual norm, without retaining Q or forming normal equations.
        augmented = np.empty((n_samples, n_terms + 1), order="F")
        augmented[:, :-1], augmented[:, -1] = design, centered
        reduced = np.linalg.qr(augmented, mode="r")
        del augmented
        fit_design, fit_target = reduced[:, :-1], reduced[:, -1]

    baseline = np.array([center])
    fits = {1: (baseline, 1 if method == "ols" else None, None, sst)}
    # Fit the largest model first so an unidentifiable coefficient request
    # fails before any reduced-model work. Reuse this fit in the last row.
    for width in dict.fromkeys([n_terms, *widths[1:-1]]):
        if width == 1:
            continue
        if method == "ols":
            beta, _, rank, _ = np.linalg.lstsq(
                fit_design[:, :width],
                fit_target,
                rcond=np.finfo(float).eps * max(n_samples, width),
            )
            if require_coefficients and width == n_terms and rank != n_terms:
                _raise_rank(n_samples, n_terms, rank)
            residual = fit_target - np.einsum("ij,j->i", fit_design[:, :width], beta)
            beta[0] += center
            penalty = None
        else:
            beta, penalty = _lasso_fit(
                design[:, :width],
                target,
                scale,
                alpha,
                cv,
                random_state,
                max_iter,
                tol,
                splits=splits,
                n_jobs=n_jobs,
            )
            rank = None
            residual = target - np.einsum("ij,j->i", design[:, :width], beta)
        fits[width] = (beta, rank, penalty, float(np.dot(residual, residual)))

    beta, rank, chosen_alpha, _ = fits[n_terms]
    identifiable = bool(rank == n_terms) if method == "ols" else None
    fractions = (
        _variance_fractions(beta, arities, order)
        if identifiable is not False
        else np.full(order + 1, np.nan)
    )
    degrees = np.fromiter((len(t) for t in terms), dtype=int, count=n_terms)
    rows = []
    previous = 0.0 if sst else np.nan
    for degree, width in enumerate(widths):
        _, fit_rank, penalty, sse = fits[width]
        r2 = (1.0 - sse / sst) if sst else np.nan
        delta = r2 - previous
        rows.append(
            (
                degree,
                r2,
                delta,
                np.sqrt(sse / n_samples) * scale,
                width,
                fit_rank,
                penalty,
                fractions[degree],
                int(np.count_nonzero(beta[degrees == degree]))
                if method == "lasso"
                else None,
            )
        )
        previous = r2
    summary = pd.DataFrame(
        rows,
        columns=[
            "order",
            "r2",
            "delta_r2",
            "rmse",
            "n_terms",
            "rank",
            "alpha",
            "model_variance_fraction",
            "n_nonzero",
        ],
    )
    summary["rank"] = summary["rank"].astype("Int64")
    summary["n_nonzero"] = summary["n_nonzero"].astype("Int64")
    summary["alpha"] = pd.to_numeric(summary["alpha"], errors="coerce").astype(float)
    info = {
        "method": method,
        "n_samples": n_samples,
        "n_terms": n_terms,
        "rank": None if rank is None else int(rank),
        "max_order": order,
        "alpha": chosen_alpha,
        "cv": int(cv) if splits is not None else None,
        "r2": float(summary.r2.iloc[-1]),
        "rmse": float(summary.rmse.iloc[-1]),
        "score_population": "training",
        "spectrum_population": "uniform_product",
        "spectrum_model_order": order,
        "identifiable": identifiable,
    }
    positions_map = dict(zip(positions, X.columns))
    reference = {
        p: a[0].item() if isinstance(a[0], np.generic) else a[0]
        for p, a in zip(positions, alleles)
    }
    info.update(position_labels=positions_map, reference=reference)
    coefficients = None
    if require_coefficients:
        with np.errstate(over="ignore", invalid="ignore"):
            values = beta * scale
        if not np.isfinite(values).all():
            raise ValueError("Coefficients exceed the float64 range; rescale fitness.")
        labels = [_allele_labels(a) for a in alleles]
        coef_rows = []
        for term, value in zip(terms, values):
            label = (
                "-".join(
                    f"{labels[j][0]}_{positions[j]}_{labels[j][a]}" for j, a in term
                )
                or "WT"
            )
            coef_rows.append(
                (len(term), tuple(positions[j] for j, _ in term), label, value)
            )
        coefficients = pd.DataFrame(
            coef_rows, columns=["order", "positions", "term", "coefficient"]
        )
        coefficients = coefficients.sort_values(
            ["order", "term"], kind="stable"
        ).reset_index(drop=True)
        coefficients.attrs.update(
            fit_info=info.copy(), position_labels=positions_map, reference=reference
        )
    summary.attrs["fit_info"] = info.copy()
    return Bunch(coefficients=coefficients, order_summary=summary, fit_info=info)


def _variance_fractions(coefficients, arities, max_order):
    """Exact model spectrum under uniform independent categorical states.

    A site's nonreference contrasts have covariance I/s - 11^T/s^2.
    Apply its square root along each coefficient-tensor axis and sum squares.
    This avoids both a dense Gram matrix and full-genotype enumeration; distinct
    supports are orthogonal, even though alleles within one support are not.
    """
    result = np.zeros(max_order + 1)
    amplitude = float(np.max(np.abs(coefficients[1:]))) if len(coefficients) > 1 else 0
    if amplitude == 0:
        return np.full(max_order + 1, np.nan)
    offset = 1
    active_sites = tuple(j for j, s in enumerate(arities) if s > 1)
    for degree in range(1, max_order + 1):
        for sites in combinations(active_sites, degree):
            shape = tuple(arities[j] - 1 for j in sites)
            size = int(np.prod(shape))
            if size == 0:
                continue
            tensor = (coefficients[offset : offset + size] / amplitude).reshape(shape)
            for axis, site in enumerate(sites):
                mean = tensor.mean(axis=axis, keepdims=True)
                s = arities[site]
                tensor = (tensor - mean) / np.sqrt(s) + mean / s
            result[degree] += float(np.sum(tensor * tensor))
            offset += size
    total = result.sum()
    return result / total if total else np.full(max_order + 1, np.nan)
