"""The data-preparation pipeline: filter -> prepare -> clean -> encode.

The processing steps used by ``Landscape.build_from_data`` to turn validated
input into the encoded ``PreparedData`` container.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ._validation import (
    PreparedData,
    _apply_mask,
    _warn_if_missing,
    _drop_duplicates,
    _invariant_columns,
    _build_config_dict,
)
from .handlers import DefaultHandler
import logging

logger = logging.getLogger(__name__)


# ===================================================================
# Pipeline — public API
# ===================================================================


def filter_data(X, f, maximize, tau, filter_mode, verbose):
    """Filter raw configuration data by the functional threshold."""
    if tau is not None:
        if filter_mode == "any":
            if verbose:
                logger.info(
                    f" - Applying functional threshold filter "
                    f"(tau={tau})..."
                )

            initial_count = len(f)
            fitness_values = f.to_numpy(copy=False) if isinstance(f, pd.Series) else np.asarray(f)

            if maximize:
                mask = fitness_values >= tau
                comparison_op = ">="
            else:
                mask = fitness_values <= tau
                comparison_op = "<="

            X = _apply_mask(X, mask)
            f = _apply_mask(f, mask)

            final_count = len(f)
            removed_count = initial_count - final_count

            if verbose:
                opposite_op = "<" if comparison_op[0] == ">" else ">"
                opposite_op += "=" if len(comparison_op) > 1 else ""

                logger.info(
                    f"   - Removed {removed_count} configurations with fitness "
                    f"{opposite_op} {tau}"
                )
                logger.info(f"   - Kept {final_count}/{initial_count} configurations")

            if final_count == 0:
                raise ValueError(
                    f"All configurations removed by functional threshold filter "
                    f"(tau={tau})"
                )

    return X, f


def prepare_data(handler, X, f, data_types=None, verbose=True):
    """Dispatch to the appropriate handler based on its type.

    Parameters
    ----------
    handler : InputHandler
        The handler instance selected for the current landscape type.
    X : Any
        Raw configuration data.
    f : Series, list, or ndarray
        Fitness values.
    data_types : dict or None
        Required when *handler* is a ``DefaultHandler``; ignored otherwise.
    verbose : bool
        Whether to print processing information.

    Returns
    -------
    X_processed : DataFrame
    f_processed : Series
    data_types : dict
    n_vars : int
    """
    if isinstance(handler, DefaultHandler):
        if not isinstance(data_types, dict):
            raise ValueError(
                "Data_types must be a dictionary, e.g., "
                "{'var_0': 'boolean', 'var_1': 'categorical'}"
                f"got {type(data_types)}."
            )
        return handler.prepare(X, f, data_types, verbose=verbose)
    else:
        return handler.prepare(X, f, verbose=verbose)


def clean_data(
    X_in: pd.DataFrame,
    f_in: pd.Series,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Warn on missing values (no row drops) and remove duplicate configurations."""
    if verbose:
        logger.info(" - Handling missing values and duplicates...")

    X_clean, f_clean = _warn_if_missing(X_in, f_in, verbose=verbose)
    if len(X_clean) == 0:
        raise ValueError("All data removed during missing value handling.")

    X_final, f_final = _drop_duplicates(X_clean, f_clean, verbose=verbose)
    if len(X_final) == 0:
        raise ValueError("All data removed after handling duplicates.")

    return X_final, f_final


def encode_data(
    X: pd.DataFrame,
    f: pd.Series,
    data_types: Dict[str, str],
    verbose: bool = False,
) -> PreparedData:
    """Encode configurations and assemble metadata for graph construction."""
    if verbose:
        logger.info("Preparing data for landscape construction (encoding variables)...")

    prepared_data_types = data_types.copy()
    invariant_cols = _invariant_columns(X)

    # Reduced view for internal neighbor/graph computation; full X is kept so
    # invariant columns stay visible to the user via get_data().
    X_for_encoding = X
    if invariant_cols:
        X_for_encoding = X.drop(columns=invariant_cols)
        prepared_data_types = {
            key: value
            for key, value in prepared_data_types.items()
            if key not in invariant_cols
        }
        if verbose:
            logger.info(
                f" - Removed {len(invariant_cols)} invariant variable(s) from "
                f"internal encoding; {len(prepared_data_types)} variable(s) used "
                f"for neighbor computation."
            )

    if not prepared_data_types:
        raise ValueError(
            "All variables are invariant (take a single unique value across "
            "every configuration). This typically happens when all input rows "
            "are identical and deduplication reduces the data to a single "
            "configuration. The landscape cannot be built from data with no "
            "variation — there are no pairs of neighbors to connect. "
            "Ensure your dataset contains at least two distinct configurations "
            "that differ in at least one variable."
        )

    encoded_columns = {}
    for col, dtype in prepared_data_types.items():
        col_data = X_for_encoding[col]
        if dtype == "boolean":
            num = pd.to_numeric(col_data, errors="coerce")
            invalid = col_data.notna() & (num.isna() | ~((num == 0) | (num == 1)))
            if invalid.any():
                bad = col_data.loc[invalid].head(5).tolist()
                raise ValueError(
                    f"Column {col!r} is declared 'boolean' but contains values that "
                    f"are not 0/1 (or True/False) after numeric coercion (examples: {bad}). "
                    "Fix the column or choose a different data_types entry."
                )
            if num.isnull().any():
                raise ValueError(
                    f"Column {col!r} contains NaN. Impute, drop those rows, or remove "
                    "the site before building the landscape."
                )
            encoded_columns[col] = num.astype(int)
        elif dtype == "categorical":
            if isinstance(col_data.dtype, pd.CategoricalDtype):
                encoded_columns[col] = col_data.cat.codes
            else:
                encoded_columns[col] = pd.factorize(col_data)[0]
        elif dtype == "ordinal":
            if isinstance(col_data.dtype, pd.CategoricalDtype) and col_data.cat.ordered:
                encoded_columns[col] = col_data.cat.codes
            else:
                encoded_columns[col] = pd.Categorical(col_data, ordered=True).codes
        else:
            raise ValueError(
                f"Unsupported data type '{dtype}' encountered during encoding."
            )

    if X_for_encoding.isnull().any().any():
        nan_cols = X_for_encoding.columns[X_for_encoding.isnull().any()].tolist()
        raise ValueError(
            "Cannot encode configurations: NaN in feature column(s) "
            f"{nan_cols[:25]}{'...' if len(nan_cols) > 25 else ''}. "
            "Impute, drop affected rows, or remove sites before building."
        )

    f_values = np.asarray(f, dtype=float)
    if not np.isfinite(f_values).all():
        n_nan = int(np.isnan(f_values).sum())
        n_inf = int(np.isinf(f_values).sum())
        raise ValueError(
            f"Fitness `f` contains non-finite values ({n_nan} NaN, {n_inf} inf). "
            "Remove or impute them before building — non-finite fitness breaks "
            "optimum detection and distance calculations."
        )

    X_encoded = pd.DataFrame(encoded_columns, index=X_for_encoding.index, copy=False)
    encoded_array = X_encoded.to_numpy(copy=False)
    max_code = int(encoded_array.max()) if encoded_array.size else 0
    if max_code <= np.iinfo(np.uint8).max:
        encoded_dtype = np.uint8
    elif max_code <= np.iinfo(np.uint16).max:
        encoded_dtype = np.uint16
    elif max_code <= np.iinfo(np.uint32).max:
        encoded_dtype = np.uint32
    else:
        encoded_dtype = np.uint64

    configs_array = np.ascontiguousarray(encoded_array, dtype=encoded_dtype)

    # configs tuple Series is built lazily (PreparedData.configs); eager
    # materialisation of all boxed ints + tuples was this function's largest
    # cost on high-dim inputs and most builds only consume configs_array.
    # The old post-encoding "duplicate configs" check is dropped: encoders are
    # injective on rows and clean_data already deduped over all columns, so the
    # encoding is duplicate-free by construction and the warning never fired.

    config_dict = _build_config_dict(prepared_data_types, X_encoded)

    # Display frame keeps the FULL original X in its ORIGINAL column order, so
    # invariant columns stay both visible AND in their input positions via
    # get_data(). Internal encoding above already dropped invariant columns;
    # downstream feature access selects columns by name (data_types.keys()), so
    # the user-facing column order is independent of the internal encoding.
    X_display = X
    data_for_attributes = X_display.reset_index(drop=True)
    data_for_attributes["fitness"] = f.to_numpy(copy=False)

    return PreparedData(
        data_for_attributes=data_for_attributes,
        data_types=prepared_data_types,
        n_vars=len(prepared_data_types),
        config_dict=config_dict,
        configs_array=configs_array,
        configs_index=X_encoded.index,
    )
