"""Normalize prepared research data for the two homepage scatter plots."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "insights"
FEATURE_LABELS = {
    "magnitude": "Magnitude epistasis", "sign": "Sign epistasis",
    "reciprocal_sign": "Reciprocal sign epistasis",
    "diminishing_returns_index": "Diminishing returns", "increasing_costs_index": "Increasing costs",
    "global_idiosyncratic_index": "Global idiosyncrasy", "r_s_ratio": "Roughness-to-slope ratio",
    "gamma": "Gamma", "gamma_star": "Gamma star", "local_optima_ratio": "Local optima ratio",
    "autocorrelation": "Autocorrelation", "fdc": "Fitness–distance correlation",
    "evolvability_enhancing_fraction": "Evolvability-enhancing fraction",
    "global_optima_accessibility": "Global optimum accessibility",
}


def load_insights(include_partial=False):
    """Read only recorded data; page builds never download or calculate metrics."""
    panels = []
    for name in ("proteingym", "evolution"):
        path = ROOT / name / "data.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        metadata = data.get("metadata", {})
        if not include_partial and (data.get("partial") or metadata.get("status") == "partial" or data.get("status") == "partial"):
            continue
        records = [{key: record[key] for key in (
            "id", "label", "protein", "mean_mutations", "n_variants", "n_graph_configs", "n_isolated_removed",
            "n_configs", "coverage_fraction",
            "n_sites", "features", "models", "outcomes", "publication", "source_url", "source_file", "source_sha256",
        ) if key in record} for record in data.get("records", [])]
        if not records:
            continue
        outcomes = data.get("outcomes", data.get("models", []))
        if isinstance(outcomes, dict):
            outcomes = [{"key": key, "label": value if isinstance(value, str) else value.get("label", key)}
                        for key, value in outcomes.items()]
        outcome_keys = {item["label"]: item["key"] for item in outcomes}
        features = data.get("features", [{"key": k, "label": v} for k, v in FEATURE_LABELS.items()])
        if isinstance(features, dict):
            features = [{"key": k, "label": v if isinstance(v, str) else v.get("label", k)} for k, v in features.items()]
        features = [{"key": item["key"], "label": FEATURE_LABELS.get(item["key"].removeprefix("epistasis."), item["label"])}
                    for item in features]
        seen = set()
        for record in records:
            if record["id"] in seen:
                raise ValueError(f"Duplicate dataset: {name}/{record['id']}")
            seen.add(record["id"])
            record["outcomes"] = record.get("outcomes", record.pop("models", {}))
            record["outcomes"] = {outcome_keys.get(key, key): value
                                  for key, value in record["outcomes"].items()}
            record.setdefault("label", record["id"])
            record.setdefault("publication", {})
            for field in ("features", "outcomes"):
                for key, value in record[field].items():
                    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                        raise ValueError(f"Invalid {name}/{record['id']}/{key}: {value!r}")
                    if field == "outcomes" and value is not None and not (-1 if name == "proteingym" else 0) <= value <= 1:
                        raise ValueError(f"Outcome outside its declared range: {name}/{record['id']}/{key}")
        if not outcomes:
            keys = sorted({key for r in records for key in r["outcomes"]})
            outcomes = [{"key": key, "label": key} for key in keys]
        default_feature = data.get("default_feature", metadata.get("default_feature", "epistasis.reciprocal_sign"))
        if default_feature not in {item["key"] for item in features}:
            default_feature = features[0]["key"]
        default_outcome = data.get("default_outcome", metadata.get("default_model", outcomes[0]["key"]))
        default_outcome = next((item["key"] for item in outcomes
                                if default_outcome in (item["key"], item["label"])), outcomes[0]["key"])
        panels.append({"id": name, "features": features, "outcomes": outcomes,
                       "records": records, "provenance": data.get("provenance", {}),
                       "y_format": ".2f" if name == "proteingym" else "%",
                       "default_feature": default_feature, "default_outcome": default_outcome})
    return panels
