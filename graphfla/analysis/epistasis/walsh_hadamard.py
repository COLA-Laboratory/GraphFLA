"""Integrated Walsh-Hadamard decomposition and order attribution."""

from ._walsh import _analyze


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
    n_jobs=1,
):
    r"""Fit W-H coefficients and summarize contributions through each order.

    Use the multistate Walsh-Hadamard representation of Faure et al. [1]_.
    Fit terms through ``max_order`` by ordinary least squares or explicit
    Lasso regularization. Incomplete landscapes are allowed; OLS requires
    a design matrix with full column rank. Build the design once and reuse it
    to fit the nested models in the order summary.

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

    n_jobs : int, default=1
        Number of parallel cross-validation jobs for Lasso. ``-1`` uses all
        available processors. Does not parallelize the sequence of orders.

    Returns
    -------
    result : sklearn.utils.Bunch
        Result with ``coefficients``, ``order_summary`` and ``fit_info``.
        ``result.coefficients`` is a DataFrame sorted by ``(order, term)`` with:

        - ``order``: number of interacting variables.
        - ``positions``: tuple of one-based original feature positions.
        - ``term``: label ``source_position_target``, joined by ``-`` for
          interactions. Allele-label characters ``%``, ``_`` and ``-`` are
          percent-escaped; ordinary sequence labels are unchanged.
        - ``coefficient``: fitted effect in fitness units.

        The order-zero row retains the legacy label ``WT``. It is the model's
        uniform full-space mean, not the reference configuration's fitness.
        ``result.order_summary`` has one row per order from zero through the
        effective maximum, with columns:

        - ``order``: maximum order in the corresponding nested fit.
        - ``r2``: cumulative training R-squared of that fit.
        - ``delta_r2``: increase over the preceding order (zero at order zero).
        - ``rmse``: training root mean squared error in fitness units.
        - ``n_terms`` and ``rank``: design width and OLS rank (NA for Lasso).
        - ``alpha``: penalty selected for that nested fit (NaN for OLS or the
          constant baseline; zero for a constant CV solution).
        - ``model_variance_fraction``: that order's share of the highest-order
          fitted model's variance over uniform product-space backgrounds.
          This is a model spectrum, not an observed-data R-squared increment.
        - ``n_nonzero``: number of nonzero coefficients of exactly that order
          in the highest-order Lasso fit (NA for OLS).

        R-squared and its increments are NaN for constant fitness. Model
        variance fractions are NaN for a constant fitted model. Negative
        regularized R-squared increments are retained, not clipped.
        ``result.fit_info`` records the estimator, dimensions, final rank and
        alpha, reference alleles, original column names and scoring conventions.
        The same metadata is attached to both tables; save it separately when
        exporting to formats such as CSV.

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
    UserWarning
        If fitness is constant and R-squared is undefined.
    sklearn.exceptions.ConvergenceWarning
        If Lasso does not converge within ``max_iter`` iterations.

    Notes
    -----
    Use observed allele 0 as reference for Boolean variables and the first retained
    observation for other variables. Construct columns of :math:`H^{-1}V^{-1}`
    as products of centered state indicators, then fit equally weighted
    observations [1]_. Backgrounds are uniform over the Cartesian product of
    observed states. On a complete landscape with all orders, OLS equals the
    direct transform; incomplete or truncated fits estimate model coefficients.

    Refit each nested model; CV selects alpha within each order using the same
    folds. Scores describe training fit, not held-out prediction or causal
    attribution. The spectrum uses exact product-space covariance, accounting
    for correlated multistate columns. Lasso leaves the constant unpenalized;
    this and the adaptive CV grid differ from the authors' analysis scripts.

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
    >>> result = walsh_hadamard(landscape)
    >>> coefficients = result.coefficients
    >>> round(float(coefficients.set_index("term").loc["0_1_1-0_2_1", "coefficient"]), 6)
    5.0
    >>> regularized = walsh_hadamard(landscape, method="lasso", alpha=0.1)
    >>> regularized.fit_info["alpha"]
    0.1
    >>> result.order_summary.order.tolist()
    [0, 1, 2]
    """
    return _analyze(
        landscape,
        max_order=max_order,
        max_cells=max_cells,
        chunk_size=chunk_size,
        method=method,
        alpha=alpha,
        cv=cv,
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
    )
