import pandas as pd
import numpy as np
from typing import Dict, List, Union, Callable, Tuple, Any, Optional, Set


class LandscapeFilter:
    """Filter for landscape data based on user-defined rules.

    This class allows filtering configurations (X) and fitness values (f)
    based on various criteria. It can be used both during landscape construction
    by passing to the `from_data` method or after retrieving data via `get_data()`.

    Parameters
    ----------
    rules : list of dicts
        A list of filtering rules, each defined as a dictionary with:
        - 'column': The column name to filter on (use 'fitness' for f)
        - 'operation': The filtering operation ('>', '<', '>=', '<=', '==', '!=', 'in', 'not_in', 'contains', 'custom')
        - 'value': The threshold/reference value for the operation
        - 'function': For 'custom' operation, a callable taking a Series and returning a boolean mask

    combine_with : str, default='and'
        How to combine multiple rules: 'and' (all must pass) or 'or' (at least one must pass)

    Examples
    --------
    >>> # Filter for fitness > 0.7
    >>> fitness_filter = LandscapeFilter.fitness_threshold(0.7)
    >>>
    >>> # Pass filter directly to from_data
    >>> landscape = Landscape.from_data(X, f, data_types="boolean", filter=fitness_filter)
    """

    def __init__(self, rules: List[Dict[str, Any]], combine_with: str = "and"):
        self.rules = rules
        self.combine_with = combine_with.lower()
        if self.combine_with not in ["and", "or"]:
            raise ValueError("combine_with must be 'and' or 'or'")
        self._validate_rules()

    def _validate_rules(self) -> None:
        """Validates the structure of provided rules."""
        if not isinstance(self.rules, list):
            raise TypeError("Rules must be provided as a list of dictionaries")

        for i, rule in enumerate(self.rules):
            if not isinstance(rule, dict):
                raise TypeError(f"Rule {i} must be a dictionary")

            # Check required keys
            required_keys = ["column", "operation"]
            for key in required_keys:
                if key not in rule:
                    raise ValueError(f"Rule {i} missing required key: {key}")

            # Check operation validity
            valid_ops = [
                ">",
                "<",
                ">=",
                "<=",
                "==",
                "!=",
                "in",
                "not_in",
                "contains",
                "custom",
            ]
            if rule["operation"] not in valid_ops:
                raise ValueError(f"Rule {i} has invalid operation: {rule['operation']}")

            # For 'custom' operation, ensure 'function' is provided and is callable
            if rule["operation"] == "custom":
                if "function" not in rule:
                    raise ValueError(
                        f"Rule {i} with 'custom' operation must provide a 'function' key"
                    )
                if not callable(rule["function"]):
                    raise TypeError(f"Rule {i} 'function' must be callable")

            # For other operations, ensure 'value' is provided (except 'custom')
            elif rule["operation"] != "custom" and "value" not in rule:
                raise ValueError(f"Rule {i} missing required key: 'value'")

    def apply(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Apply the filter rules to the provided data.

        Parameters
        ----------
        data : DataFrame or Series
            The data to filter, either configurations with fitness (DataFrame)
            or just fitness values (Series)

        Returns
        -------
        DataFrame or Series
            Filtered data of the same type as the input
        """
        if len(data) == 0:
            return data

        # For Series input, assume it's fitness data
        if isinstance(data, pd.Series):
            # Create a temporary DataFrame with the Series as a column
            temp_df = pd.DataFrame({"fitness": data})
            filtered_df = self._apply_rules(temp_df)
            # Return a Series with the same name and index structure
            return filtered_df["fitness"]

        # For DataFrame input, apply rules directly
        return self._apply_rules(data)

    def _apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal method to apply filtering rules to a DataFrame."""
        # Initialize mask based on combine_with mode
        if self.combine_with == "and":
            mask = pd.Series(True, index=df.index)
        else:  # 'or'
            mask = pd.Series(False, index=df.index)

        for rule in self.rules:
            column = rule["column"]

            # Skip rules for columns not in the data
            if column not in df.columns:
                continue

            op = rule["operation"]
            rule_mask = None

            if op == ">":
                rule_mask = df[column] > rule["value"]
            elif op == "<":
                rule_mask = df[column] < rule["value"]
            elif op == ">=":
                rule_mask = df[column] >= rule["value"]
            elif op == "<=":
                rule_mask = df[column] <= rule["value"]
            elif op == "==":
                rule_mask = df[column] == rule["value"]
            elif op == "!=":
                rule_mask = df[column] != rule["value"]
            elif op == "in":
                rule_mask = df[column].isin(rule["value"])
            elif op == "not_in":
                rule_mask = ~df[column].isin(rule["value"])
            elif op == "contains":
                if df[column].dtype == object:  # For string/object columns
                    rule_mask = df[column].str.contains(rule["value"], na=False)
                else:
                    # For non-string columns, try a different approach if sensible
                    rule_mask = pd.Series(False, index=df.index)
            elif op == "custom":
                # Custom function should return a boolean Series with the same index
                rule_mask = rule["function"](df[column])

            # Combine the rule mask with the overall mask
            if self.combine_with == "and":
                mask = mask & rule_mask
            else:  # 'or'
                mask = mask | rule_mask

        return df[mask]

    def filter_data(
        self, X: Union[pd.DataFrame, np.ndarray], f: Union[pd.Series, List, np.ndarray]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Filter data for landscape construction.

        Parameters
        ----------
        X : DataFrame or array-like
            Configuration data
        f : Series or array-like
            Fitness values

        Returns
        -------
        X_filtered : DataFrame
            Filtered configuration data
        f_filtered : Series
            Filtered fitness values
        """
        # Convert to standard types if needed
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        f_series = f if isinstance(f, pd.Series) else pd.Series(f, name="fitness")

        # Create a combined DataFrame for filtering
        combined = X_df.copy()
        combined["fitness"] = f_series

        # Apply the rules
        filtered = self._apply_rules(combined)

        # Split back into X and f
        X_filtered = filtered.drop(columns=["fitness"])
        f_filtered = pd.Series(
            filtered["fitness"], name="fitness", index=filtered.index
        )

        return X_filtered, f_filtered

    @classmethod
    def fitness_threshold(
        cls, threshold: float, operation: str = ">", combine_with: str = "and"
    ) -> "LandscapeFilter":
        """Create a filter for fitness values based on a threshold.

        Parameters
        ----------
        threshold : float
            The fitness threshold value
        operation : str, default='>'
            The comparison operation: '>', '<', '>=', '<=', '==', '!='
        combine_with : str, default='and'
            How to combine with other rules

        Returns
        -------
        LandscapeFilter
            A filter configured with the threshold rule
        """
        rules = [{"column": "fitness", "operation": operation, "value": threshold}]
        return cls(rules, combine_with)

    @classmethod
    def config_values(
        cls,
        column: str,
        allowed_values: Union[List, Set],
        include: bool = True,
        combine_with: str = "and",
    ) -> "LandscapeFilter":
        """Create a filter for configuration values.

        Parameters
        ----------
        column : str
            The column name to filter on
        allowed_values : list or set
            The values to include or exclude
        include : bool, default=True
            If True, include rows where column is in allowed_values
            If False, exclude rows where column is in allowed_values
        combine_with : str, default='and'
            How to combine with other rules

        Returns
        -------
        LandscapeFilter
            A filter configured with the configuration value rule
        """
        operation = "in" if include else "not_in"
        rules = [{"column": column, "operation": operation, "value": allowed_values}]
        return cls(rules, combine_with)

    @classmethod
    def custom_filter(
        cls,
        column: str,
        filter_function: Callable[[pd.Series], pd.Series],
        combine_with: str = "and",
    ) -> "LandscapeFilter":
        """Create a filter with a custom filtering function.

        Parameters
        ----------
        column : str
            The column name to filter on
        filter_function : callable
            A function that takes a Series and returns a boolean Series
        combine_with : str, default='and'
            How to combine with other rules

        Returns
        -------
        LandscapeFilter
            A filter configured with the custom function
        """
        rules = [{"column": column, "operation": "custom", "function": filter_function}]
        return cls(rules, combine_with)

    @classmethod
    def combine_filters(
        cls, *filters: "LandscapeFilter", combine_with: str = "and"
    ) -> "LandscapeFilter":
        """Combine multiple filters into a single filter.

        Parameters
        ----------
        *filters : LandscapeFilter
            The filters to combine
        combine_with : str, default='and'
            How to combine the filters

        Returns
        -------
        LandscapeFilter
            A new filter that combines all the rules from the input filters
        """
        combined_rules = []
        for filter_obj in filters:
            combined_rules.extend(filter_obj.rules)
        return cls(combined_rules, combine_with)
