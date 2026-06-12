"""Higher-order epistasis via a regression R^2 decomposition."""

import numpy as np

from .._utils import _pythonize
import logging

logger = logging.getLogger(__name__)


def higher_order_epistasis(landscape, order=2, verbose=False, n_jobs=1):
    """
    Calculates the fraction of variance in fitness that can be explained
    by interactions between variables up to the specified order using polynomial regression.

    Parameters
    ----------
    landscape : Landscape
        The fitness landscape object to analyze.
    order : int, optional
        The maximum order of polynomial features to consider. This controls the degree
        of the polynomial, where an order of k allows for modeling interactions between
        up to k variables. Must be between 1 and the total number of variables in the landscape.
        Default is 2 (quadratic terms and pairwise interactions).
    verbose : bool, optional
        Whether to print progress information. Default is False.
    n_jobs : int, optional
        Number of CPU cores used by the underlying linear regression. Default is 1.

    Returns
    -------
    float
        The R² score representing the fraction of variance explained by
        polynomial terms up to the specified order. Values closer to 1.0 indicate
        stronger epistasis of the given order.

    Notes
    -----
    This function uses polynomial regression with degree=order to model interactions
    up to the specified order. The resulting R² score indicates how well these
    interactions explain the observed fitness values.

    A high R² score suggests that most of the fitness variance can be
    explained by considering interactions up to the specified order,
    indicating strong epistatic effects of that order in the landscape.

    """
    try:
        from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
    except ImportError:
        raise ImportError(
            "This function requires scikit-learn. "
            "Please install it with 'pip install scikit-learn'."
        )

    landscape._check_built()

    if landscape.configs is None or len(landscape.configs) == 0:
        raise ValueError("Landscape has no configuration data.")

    if not isinstance(order, int):
        raise TypeError(f"Order must be an integer, got {type(order).__name__}")

    if order < 1:
        raise ValueError(f"Order must be at least 1, got {order}")

    if order > landscape.n_vars:
        raise ValueError(
            f"Order cannot exceed the number of variables in the landscape "
            f"({landscape.n_vars}), got {order}"
        )

    if verbose:
        logger.info(f"Calculating order-{order} epistasis using polynomial regression...")

    X = np.vstack(landscape.configs.values)
    y = np.array(landscape.graph.vs["fitness"])

    # Boolean is already 0/1; other types need one-hot with a reference level
    # dropped for a numerically stable design matrix.
    if verbose:
        logger.info(f"Encoding {X.shape[1]} variables...")

    if landscape.kind == "boolean":
        X_encoded = np.asarray(X, dtype=np.float64)
    else:
        encoder = OneHotEncoder(
            sparse_output=False,
            drop="first",
            dtype=np.float64,
        )
        try:
            X_encoded = encoder.fit_transform(X)
        except Exception as e:
            raise ValueError(f"Failed to one-hot encode configurations: {e}") from e

    if verbose:
        logger.info(f"Encoded data shape: {X_encoded.shape}")
        logger.info(f"Creating polynomial features of degree {order}...")

    # Use interaction-only features and let LinearRegression handle the intercept.
    poly = PolynomialFeatures(
        degree=order,
        include_bias=False,
        interaction_only=True,
    )
    model = LinearRegression(n_jobs=n_jobs)

    try:
        if verbose:
            logger.info(f"Fitting polynomial regression model...")
        X_poly = poly.fit_transform(X_encoded)
        model.fit(X_poly, y)
        # Manual dot instead of np.matmul: Accelerate (macOS arm64) emits
        # spurious RuntimeWarnings on finite inputs.
        coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        y_pred = (
            np.sum(
                np.asarray(X_poly, dtype=np.float64) * coefficients,
                axis=1,
                dtype=np.float64,
            )
            + float(model.intercept_)
        )
        r2 = r2_score(y, y_pred)
    except Exception as e:
        raise RuntimeError(f"Error fitting polynomial regression model: {e}") from e

    if verbose:
        logger.info(f"Order-{order} epistasis R² score: {r2:.4f}")

    return _pythonize(r2)
