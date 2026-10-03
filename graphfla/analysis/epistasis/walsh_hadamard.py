"""Background-averaged epistatic coefficients on discrete product spaces."""

from itertools import combinations, product
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state


def walsh_hadamard(
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
):
    r"""Fit background-averaged epistatic coefficients in an extended W-H basis.

    Use the multistate Walsh-Hadamard representation of Faure et al. [1]_.
    Fit terms through ``max_order`` by ordinary least squares or explicit
    Lasso regularization. Incomplete landscapes are allowed; OLS requires
    a design matrix with full column rank.

    Parameters
    ----------
    landscape : Landscape
        Built landscape with finite fitness and unique, nonmissing
        configurations. Use the nodes retained in ``get_data()``; construction
        filters are not reversed. All variable types, including ordinal
        variables, are treated as discrete states without numeric spacing.
        States absent from the retained data are not inferred. Fitness is used
        in its supplied units, without changing sign for minimization.
    max_order : int, default=2
        Largest number of distinct interacting variables. Must be nonnegative.
        Zero fits only a constant; values above the number of variable sites
        include all orders. Truncation specifies a model, not evidence that
        omitted interactions are absent.
    max_cells : int or float, default=1e7
        Maximum number of entries in the dense design, including its constant
        column. Checked before enumerating terms or allocating the design.
        Each entry uses eight bytes. This is not a total memory limit: solver
        copies and workspace also require memory.
    chunk_size : int, default=1000
        Maximum number of rows in temporary feature-construction blocks.
        The final design is dense regardless of this value.
    method : {"ols", "lasso"}, default="ols"
        Coefficient estimator. OLS raises on a rank-deficient design. Lasso
        minimizes squared prediction error plus an L1 coefficient penalty;
        it does not establish identifiability of unregularized coefficients.
        Neither method automatically switches to the other.
    alpha : float or {"cv"}, default="cv"
        Positive Lasso penalty in the objective
        ``sum((y - prediction)**2) / (2 * n_samples) + alpha * sum(abs(coef))``.
        The constant term is not penalized and features are not standardized.
        ``"cv"`` selects among 100 logarithmically spaced penalties from the
        smallest all-zero-slope penalty to one thousandth of that value, using
        mean validation squared error. Ignored for OLS.
    cv : int, default=5
        Number of shuffled folds when ``method="lasso", alpha="cv"``.
        Must be between 2 and the number of retained observations. The selected
        model is refitted on all observations; no held-out score is returned.
    random_state : int, RandomState instance or None, default=0
        Controls shuffled cross-validation folds. Pass an integer for
        reproducible splits. Used only for cross-validated Lasso.
    max_iter : int, default=10000
        Maximum coordinate-descent iterations for Lasso.
    tol : float, default=1e-4
        Positive convergence tolerance for Lasso.

    Returns
    -------
    coefficients : pandas.DataFrame
        Rows sorted by ``(order, term)`` with columns:

        - ``order``: number of interacting variables.
        - ``positions``: tuple of one-based original feature positions.
        - ``term``: label ``source_position_target``, joined by ``-`` for
          interactions. Allele-label characters ``%``, ``_`` and ``-`` are
          percent-escaped; ordinary sequence labels are unchanged.
        - ``coefficient``: fitted effect in fitness units.

        The order-zero row retains the legacy label ``WT``. It is the model's
        uniform full-space mean, not the reference configuration's fitness.
        ``attrs["fit_info"]`` records method, sample/term counts, OLS rank
        (None for Lasso), effective order, selected alpha (None for OLS), and
        CV folds (None without CV). CV records alpha=0 for a constant solution.
        ``attrs["position_labels"]`` maps positions
        to input column names; ``attrs["reference"]`` maps them to reference
        alleles. DataFrame attributes are not preserved by every export format.

    Raises
    ------
    RuntimeError
        If the landscape has not been built.
    ValueError
        If inputs or parameters are invalid, the design exceeds ``max_cells``,
        OLS coefficients are not identifiable, or coefficients exceed float64.
    numpy.linalg.LinAlgError
        If the least-squares decomposition fails to converge.

    Warns
    -----
    sklearn.exceptions.ConvergenceWarning
        If Lasso does not converge within ``max_iter`` iterations.

    See Also
    --------
    higher_order_epistasis : Cumulative fit quality of interaction models.

    Notes
    -----
    Use observed allele 0 as reference for Boolean variables and the first retained
    observation for other variables. Construct columns of :math:`H^{-1}V^{-1}`
    as products of centered state indicators, then fit equally weighted
    observations [1]_. Backgrounds are uniform over the Cartesian product of
    observed states. On a complete landscape with all orders, OLS equals the
    direct transform; incomplete or truncated fits estimate model coefficients.

    Multistate columns are not generally orthogonal. Changing the reference
    can change coefficients and Lasso estimates. The unpenalized constant and
    adaptive CV grid follow general regression conventions; they differ from
    the penalized constant and fixed grid in the authors' analysis scripts.

    References
    ----------
    .. [1] Faure, A. J., Lehner, B., Miro Pina, V., Serrano Colome, C., and
       Weghorn, D. (2024). An extension of the Walsh-Hadamard transform to
       calculate and model epistasis in genetic landscapes of arbitrary shape
       and complexity. PLOS Computational Biology 20(5): e1012132.
       https://doi.org/10.1371/journal.pcbi.1012132. Eqs. (9), (12), (23)-(25).

    Examples
    --------
    >>> from graphfla.landscape import BooleanLandscape
    >>> from graphfla.analysis import walsh_hadamard
    >>> landscape = BooleanLandscape().build_from_data(
    ...     ["00", "01", "10", "11"], [10., 13., 12., 20.], verbose=False
    ... )
    >>> coefficients = walsh_hadamard(landscape)
    >>> round(float(coefficients.set_index("term").loc["0_1_1-0_2_1", "coefficient"]), 6)
    5.0
    >>> regularized = walsh_hadamard(landscape, method="lasso", alpha=0.1)
    >>> regularized.attrs["fit_info"]["alpha"]
    0.1
    """
    landscape._check_built()
    _validate_options(max_order, max_cells, chunk_size, method, alpha, max_iter, tol)
    X, y, codes, alleles, positions = _encode_input(landscape)
    order = min(int(max_order), len(alleles))
    arities = [len(a) for a in alleles]
    n_terms = _term_count(arities, order)
    n_samples = len(y)
    if n_samples * n_terms > max_cells:
        raise ValueError(
            f"Design matrix requires {n_samples * n_terms} cells "
            f"(n_samples={n_samples}, n_terms={n_terms}), exceeding "
            f"max_cells={max_cells:g}. Reduce max_order or increase max_cells."
        )
    if method == "ols" and n_terms > n_samples:
        _raise_rank(n_samples, n_terms)
    if method == "lasso" and alpha == "cv":
        _positive_integer(cv, "cv", minimum=2)
        if cv > n_samples:
            raise ValueError(f"cv={cv} cannot exceed n_samples={n_samples}.")
        check_random_state(random_state)

    terms = [()]
    for degree in range(1, order + 1):
        for sites in combinations(range(len(arities)), degree):
            terms.extend(
                tuple(zip(sites, states))
                for states in product(*(range(1, arities[j]) for j in sites))
            )
    design = _design_matrix(codes, arities, terms, int(chunk_size))
    # Normalize only fitness, preserving the physical basis and L1 penalty.
    # This prevents solver products from overflowing in large fitness units.
    scale = float(np.max(np.abs(y))) or 1.0
    target = y / scale
    if method == "ols":
        center = float(target.mean())
        coef, _, rank, _ = np.linalg.lstsq(design, target - center, rcond=None)
        if rank != n_terms:
            _raise_rank(n_samples, n_terms, rank)
        coef[0] += center
        chosen_alpha, folds = None, None
    else:
        coef, chosen_alpha = _lasso_fit(
            design, target, scale, alpha, cv, random_state, max_iter, tol
        )
        rank, folds = None, int(cv) if alpha == "cv" else None
    with np.errstate(over="ignore", invalid="ignore"):
        coef *= scale
    if not np.isfinite(coef).all():
        raise ValueError("Coefficients exceed the float64 range; rescale fitness.")

    labels = [_allele_labels(a) for a in alleles]
    rows = []
    for term, value in zip(terms, coef):
        label = (
            "-".join(f"{labels[j][0]}_{positions[j]}_{labels[j][a]}" for j, a in term)
            or "WT"
        )
        rows.append((len(term), tuple(positions[j] for j, _ in term), label, value))
    result = pd.DataFrame(rows, columns=["order", "positions", "term", "coefficient"])
    result = result.sort_values(["order", "term"], kind="stable").reset_index(drop=True)
    result.attrs["fit_info"] = {
        "method": method,
        "n_samples": n_samples,
        "n_terms": n_terms,
        "rank": None if rank is None else int(rank),
        "max_order": order,
        "alpha": chosen_alpha,
        "cv": folds,
    }
    result.attrs["position_labels"] = dict(zip(positions, X.columns))
    result.attrs["reference"] = {
        p: a[0].item() if isinstance(a[0], np.generic) else a[0]
        for p, a in zip(positions, alleles)
    }
    return result


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


def _lasso_fit(design, target, scale, alpha, cv, random_state, max_iter, tol):
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
        folds = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        model = LassoCV(alphas=alphas, cv=folds, n_jobs=1, **options)
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
