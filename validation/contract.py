"""Validate portable benchmark definitions and append-only research events.

This module uses only the standard library. Records are data, never executable
commands. Large source artifacts and event history live in an external store.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import tempfile

SCHEMA_VERSION = 1
STATES = {
    "queued",
    "assigned",
    "source_triage",
    "ready_for_trial",
    "needs_review",
    "validated",
    "triaged",
    "blocked_access",
    "blocked_artifact",
    "blocked_method",
    "closed_no_overlap",
    "closed_low_yield",
}
OUTCOMES = {
    "reproduced_exact",
    "reproduced_with_precision",
    "independent_crosscheck",
    "definition_mismatch",
    "mismatch_unresolved",
    "suspected_graphfla_defect",
    "blocked_missing_artifact",
    "blocked_missing_method",
    "access_blocked",
    "no_overlapping_metric",
    "not_attempted",
}
TIERS = {"published_numeric", "author_artifact_numeric", "independent_equation"}
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A record cannot safely enter the benchmark history."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def text_field(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be text")


def member(value, choices, label):
    require(isinstance(value, str) and value in choices, f"Invalid {label}")


def object_list(value, label):
    require(
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value),
        f"{label} must be a nonempty list of objects",
    )
    return value


def digest_field(value, label):
    require(
        isinstance(value, str) and bool(SHA256.fullmatch(value)), f"Invalid {label}"
    )


def canonical_bytes(record):
    try:
        return json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ContractError(f"Record must be finite JSON: {exc}") from exc


def fingerprint(record):
    return hashlib.sha256(canonical_bytes(record)).hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ContractError(f"Non-finite JSON value: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {path}: {exc}") from exc


def header(record, kind):
    require(isinstance(record, dict), "Record must be an object")
    require(
        type(record.get("schema_version")) is int
        and record["schema_version"] == SCHEMA_VERSION,
        "Unsupported schema_version; migrate explicitly, never reset records",
    )
    require(record.get("kind") == kind, f"Expected kind={kind}")
    canonical_bytes(record)


def valid_id(value):
    return isinstance(value, str) and bool(ID.fullmatch(value))


def validate_catalog(catalog):
    header(catalog, "catalog")
    studies = object_list(catalog.get("studies"), "Catalog studies")
    ids, dois, names = set(), set(), set()
    for study in studies:
        key = study.get("id")
        require(valid_id(key) and key not in ids, f"Duplicate/invalid study ID: {key}")
        ids.add(key)
        text_field(study.get("title"), f"Title: {key}")
        member(study.get("origin"), {"supplied", "metric_search"}, "origin")
        aliases = study.get("aliases", [])
        require(isinstance(aliases, list), "Aliases must be a list")
        for name in [key, *aliases]:
            require(
                valid_id(name) and name not in names, f"Duplicate/invalid alias: {name}"
            )
            names.add(name)
        doi = study.get("doi")
        if doi is not None:
            text_field(doi, "DOI")
            normalized = doi.lower()
            if normalized.startswith("https://doi.org/"):
                normalized = normalized[len("https://doi.org/") :]
            require(
                normalized.startswith("10.") and normalized not in dois,
                f"Duplicate/invalid DOI: {doi}",
            )
            dois.add(normalized)
        member(study.get("state"), STATES, "study state")
        text_field(study.get("next_action"), f"Next action: {key}")
        if study.get("dossier") is not None:
            safe_relative(study["dossier"])
        for field in ("measured_variants", "theoretical_variants"):
            n = study.get(field)
            require(n is None or (type(n) is int and n >= 0), f"Invalid {field}: {key}")
    return ids


def validate_metrics(metrics):
    header(metrics, "metrics")
    seen = set()
    for item in object_list(metrics.get("metrics"), "Metrics"):
        key = item.get("id")
        require(
            valid_id(key) and key not in seen, f"Duplicate/invalid metric ID: {key}"
        )
        seen.add(key)
        functions = item.get("functions")
        require(
            isinstance(functions, list) and bool(functions),
            "Functions must be a nonempty list",
        )
        for function in functions:
            text_field(function, "Function name")
        require(bool(item.get("synthetic_plan")), f"Missing synthetic plan: {key}")
        require(
            bool(item.get("literature_next_action")),
            f"Missing literature review action: {key}",
        )
        member(
            item.get("provenance"),
            {"canonical", "adapted", "unconfirmed", "infrastructure"},
            "provenance",
        )
    require(bool(seen), "Metric inventory cannot be empty")
    return seen


def validate_case(case, studies, metrics):
    header(case, "case")
    require(valid_id(case.get("id")), "Invalid case ID")
    member(case.get("study_id"), studies, "study_id")
    require(
        type(case.get("revision")) is int and case["revision"] > 0,
        "Invalid case revision",
    )
    ids = case.get("metric_ids")
    require(isinstance(ids, list) and bool(ids), "metric_ids must be a nonempty list")
    for key in ids:
        member(key, metrics, "metric_id")
    member(case.get("evidence_tier"), TIERS, "evidence tier")
    member(
        case.get("definition_match"), {"yes", "no", "unresolved"}, "definition match"
    )
    for field in ("scope", "preprocessing"):
        text_field(case.get(field), field)
    for source in object_list(case.get("sources"), "Sources"):
        text_field(source.get("url"), "Source URL")
        require(
            source["url"].startswith(("https://", "http://")),
            "Source URL must be HTTP(S)",
        )
        text_field(source.get("locator"), "Source locator")
    require(
        case.get("expected") is not None,
        "Cases require frozen expected values; use a checkpoint for triage",
    )
    comparison = case.get("comparison", {})
    require(isinstance(comparison, dict), "Comparison must be an object")
    member(comparison.get("kind"), {"exact", "absolute"}, "comparison")
    text_field(comparison.get("justification"), "Comparison justification")
    tolerance = comparison.get("tolerance")
    require(type(tolerance) in {int, float} and tolerance >= 0, "Invalid tolerance")
    if comparison["kind"] == "exact":
        require(tolerance == 0, "Exact comparisons cannot have a tolerance")
    for item in object_list(case.get("inputs"), "Inputs"):
        validate_artifact(item)
        member(item.get("root"), {"repo", "store"}, "input root")
    return fingerprint(case)


def safe_relative(value):
    text_field(value, "Artifact path")
    require(
        "\\" not in value and ":" not in value and "\0" not in value,
        "Artifact path must be portable",
    )
    p = Path(value)
    require(
        not p.is_absolute() and ".." not in p.parts and str(p) != ".",
        "Artifact path must remain under its root",
    )
    return p


def resolve_artifact(root, relative):
    base = Path(root).resolve()
    resolved = (base / safe_relative(relative)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ContractError("Artifact symlink escapes its root") from exc
    return resolved


def validate_artifact(artifact):
    require(isinstance(artifact, dict), "Artifact must be an object")
    safe_relative(artifact.get("path"))
    digest_field(artifact.get("sha256"), "artifact SHA-256")


def verify_artifact(root, artifact):
    validate_artifact(artifact)
    path = resolve_artifact(root, artifact["path"])
    require(path.is_file(), f"Missing artifact: {path}")
    require(
        file_digest(path) == artifact["sha256"],
        f"Changed artifact: {path}",
    )
    return path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(expected, observed, comparison):
    if isinstance(expected, dict):
        return (
            isinstance(observed, dict)
            and expected.keys() == observed.keys()
            and all(matches(v, observed[k], comparison) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(expected) == len(observed)
            and all(matches(a, b, comparison) for a, b in zip(expected, observed))
        )
    if type(expected) in {int, float}:
        return (
            type(observed) in {int, float}
            and abs(expected - observed) <= comparison["tolerance"]
        )
    return type(expected) is type(observed) and expected == observed


def validate_event(event, studies, cases):
    header(event, "event")
    member(event.get("study_id"), studies, "event study_id")
    for field in ("recorded_at", "actor", "reason"):
        text_field(event.get(field), field)
    event_time(event)
    member(event.get("type"), {"checkpoint", "trial"}, "event type")
    if "artifacts" in event:
        for item in object_list(event["artifacts"], "Event artifacts"):
            validate_artifact(item)
    if event["type"] == "checkpoint":
        member(event.get("state"), STATES, "checkpoint state")
        for field in ("next_action", "last_completed_step"):
            text_field(event.get(field), field)
        if event.get("dossier") is not None:
            safe_relative(event["dossier"])
        if event["state"].startswith("closed_"):
            text_field(event.get("reopen_when"), "Closed study reopening criterion")
    else:
        require(valid_id(event.get("case_id")), "Invalid case_id")
        case = cases.get(event.get("case_id"))
        require(case is not None, "Unknown case_id")
        require(case["study_id"] == event["study_id"], "Trial study differs from case")
        require(
            event.get("case_fingerprint") == fingerprint(case),
            "Case changed; preserve old trial and create a new revision",
        )
        member(event.get("outcome"), OUTCOMES, "trial outcome")
        member(
            event.get("implementation"),
            {"graphfla", "independent_equation", "author_code"},
            "implementation",
        )
        for field in ("implementation_fingerprint", "environment_fingerprint"):
            digest_field(event.get(field), field)
        if event["outcome"].startswith("reproduced_"):
            require(
                case["definition_match"] == "yes",
                "Cannot certify an unmatched definition",
            )
            require(
                event.get("observed") is not None,
                "Successful trial needs observed values",
            )
            require(
                matches(case["expected"], event["observed"], case["comparison"]),
                "Claimed successful result does not match its frozen target",
            )
            if event["outcome"] == "reproduced_exact":
                require(
                    matches(case["expected"], event["observed"], {"tolerance": 0}),
                    "Exact outcome cannot use a tolerance",
                )
        validate_artifact(event.get("result_artifact"))


def verify_event_artifacts(store, event):
    """Bind a recorded observation to its archived result and supporting files."""
    checked = 0
    for artifact in event.get("artifacts", []):
        verify_artifact(store, artifact)
        checked += 1
    if event["type"] == "trial":
        result = read_json(verify_artifact(store, event["result_artifact"]))
        checked += 1
        header(result, "result")
        require(
            result.get("case_fingerprint") == event["case_fingerprint"],
            "Result belongs to a different case",
        )
        require(
            "observed" in result and "observed" in event,
            "Result and trial need an observation",
        )
        require(
            matches(result["observed"], event["observed"], {"tolerance": 0}),
            "Trial observation differs from its result artifact",
        )
        for artifact in object_list(
            result.get("artifacts"), "Result supporting artifacts"
        ):
            verify_artifact(store, artifact)
            checked += 1
    return checked


def event_time(event):
    try:
        value = datetime.fromisoformat(event["recorded_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError) as exc:
        raise ContractError("recorded_at must be a UTC ISO-8601 timestamp") from exc
    require(value.utcoffset() == timedelta(0), "recorded_at must include UTC timezone")
    return value


def store_result(store, source):
    """Snapshot a result before logging it so later exploratory reruns are safe."""
    payload = Path(source).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    folder = Path(store) / "results" / "sha256"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / digest
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            require(destination.read_bytes() == payload, "Corrupt result snapshot")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination.relative_to(store)), "sha256": digest}


def append_event(store, event, studies, cases):
    """Atomically publish a content-addressed event, never replace history."""
    validate_event(event, studies, cases)
    verify_event_artifacts(store, event)
    folder = Path(store) / "events"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{fingerprint(event)}.json"
    payload = canonical_bytes(event) + b"\n"
    if path.exists():
        require(path.read_bytes() == payload, "Existing event was modified")
        return path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # atomic no-clobber, including concurrent writers
        except FileExistsError:
            require(path.read_bytes() == payload, "Conflicting event content")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_definitions(directory):
    directory = Path(directory)
    catalog = read_json(directory / "catalog.json")
    metrics = read_json(directory / "metrics.json")
    studies = validate_catalog(catalog)
    metric_ids = validate_metrics(metrics)
    cases = {}
    for path in sorted((directory / "cases").glob("*.json")):
        case = read_json(path)
        validate_case(case, studies, metric_ids)
        require(case["id"] not in cases, f"Duplicate case ID: {case['id']}")
        cases[case["id"]] = case
    return catalog, metrics, cases


def load_events(store, studies, cases):
    events = []
    for path in sorted((Path(store) / "events").glob("*.json")):
        event = read_json(path)
        require(path.stem == fingerprint(event), f"Edited event history: {path}")
        validate_event(event, studies, cases)
        events.append(event)
    return sorted(events, key=lambda e: (event_time(e), fingerprint(e)))
