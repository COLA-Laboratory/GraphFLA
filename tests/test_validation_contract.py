"""Protect evidence identity, truthful result labels and resumable history."""

import ast
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from validation.contract import (
    ContractError,
    append_event,
    fingerprint,
    load_definitions,
    load_events,
    read_json,
    resolve_artifact,
    store_result,
    validate_case,
    validate_event,
    validate_catalog,
    validate_metrics,
    verify_artifact,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def definitions():
    catalog, metrics, cases = load_definitions(REPO / "validation")
    return (
        {s["id"] for s in catalog["studies"]},
        {m["id"] for m in metrics["metrics"]},
        cases,
    )


@pytest.fixture
def trial(tmp_path, definitions):
    studies, _, cases = definitions
    case = cases["bank.reia_peaks.v1"]
    result = tmp_path / "working_result.json"
    raw = tmp_path / "raw_result.json"
    raw.write_text('{"peak_count": 6}\n')
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "result",
                "case_fingerprint": fingerprint(case),
                "observed": 6,
                "artifacts": [store_result(tmp_path, raw)],
            }
        )
    )
    event = {
        "schema_version": 1,
        "kind": "event",
        "type": "trial",
        "study_id": case["study_id"],
        "case_id": case["id"],
        "case_fingerprint": fingerprint(case),
        "implementation": "graphfla",
        "implementation_fingerprint": "a" * 64,
        "environment_fingerprint": "b" * 64,
        "recorded_at": "2026-09-30T12:00:00Z",
        "actor": "contract-test",
        "outcome": "reproduced_exact",
        "observed": 6,
        "reason": "Exact independent published count.",
        "result_artifact": store_result(tmp_path, result),
    }
    return event, studies, cases


def test_every_public_analysis_function_has_a_literature_review_plan():
    _, metrics, _ = load_definitions(REPO / "validation")
    tree = ast.parse((REPO / "graphfla/analysis/__init__.py").read_text())
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    planned = {name for metric in metrics["metrics"] for name in metric["functions"]}
    assert set(exports) <= planned


def test_result_snapshot_survives_rerunning_a_working_file(tmp_path, trial):
    event, studies, cases = trial
    path = append_event(tmp_path, event, studies, cases)
    (tmp_path / "working_result.json").write_text('{"peak_count": 999}\n')
    saved = load_events(tmp_path, studies, cases)
    assert saved == [event]
    frozen = read_json(verify_artifact(tmp_path, saved[0]["result_artifact"]))
    assert frozen["observed"] == 6
    assert (
        verify_artifact(tmp_path, frozen["artifacts"][0]).read_text()
        == '{"peak_count": 6}\n'
    )
    assert read_json(path)["observed"] == 6


def test_identical_concurrent_records_are_idempotent(tmp_path, trial):
    event, studies, cases = trial
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(
            pool.map(lambda _: append_event(tmp_path, event, studies, cases), range(2))
        )
    assert paths[0] == paths[1]
    assert len(load_events(tmp_path, studies, cases)) == 1


def test_editing_a_frozen_case_cannot_relabel_old_results(trial):
    event, studies, cases = trial
    changed = deepcopy(cases)
    changed[event["case_id"]]["expected"] = 17
    with pytest.raises(ContractError, match="Case changed"):
        validate_event(event, studies, changed)


@pytest.mark.parametrize("observed", [17, None, True, "6"])
def test_incorrect_observations_cannot_be_recorded_as_reproduced(trial, observed):
    event, studies, cases = trial
    event["observed"] = observed
    with pytest.raises(ContractError):
        validate_event(event, studies, cases)


def test_definition_mismatch_cannot_be_certified(trial):
    event, studies, cases = trial
    changed = deepcopy(cases)
    case = changed[event["case_id"]]
    case["definition_match"] = "no"
    event["case_fingerprint"] = fingerprint(case)
    with pytest.raises(ContractError, match="unmatched definition"):
        validate_event(event, studies, changed)


def test_history_and_artifact_tampering_are_detected(tmp_path, trial):
    event, studies, cases = trial
    path = append_event(tmp_path, event, studies, cases)
    artifact = resolve_artifact(tmp_path, event["result_artifact"]["path"])
    artifact.write_text("changed")
    with pytest.raises(ContractError, match="Changed artifact"):
        verify_artifact(tmp_path, event["result_artifact"])
    path.write_text(path.read_text().replace('"observed":6', '"observed":7'))
    with pytest.raises(ContractError, match="Edited event history"):
        load_events(tmp_path, studies, cases)


@pytest.mark.parametrize("path", ["../outside", "/absolute/file"])
def test_artifacts_cannot_escape_the_store(tmp_path, path):
    with pytest.raises(ContractError, match="under its root"):
        resolve_artifact(tmp_path, path)


def test_symlink_cannot_escape_the_store(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "outside").symlink_to(tmp_path)
    with pytest.raises(ContractError, match="symlink escapes"):
        resolve_artifact(store, "outside/file")


def test_unknown_schema_and_nonfinite_targets_require_explicit_repair(definitions):
    studies, metrics, cases = definitions
    case = deepcopy(cases["bank.reia_peaks.v1"])
    case["schema_version"] = 999
    with pytest.raises(ContractError, match="migrate explicitly"):
        validate_case(case, studies, metrics)
    case["schema_version"] = 1
    case["expected"] = float("nan")
    with pytest.raises(ContractError, match="finite JSON"):
        validate_case(case, studies, metrics)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("study_id", []),
        ("metric_ids", "construction"),
        ("metric_ids", [{}]),
        ("sources", [None]),
        ("inputs", [7]),
        ("inputs", [{"root": "repo", "path": None, "sha256": 42}]),
        ("comparison", []),
        ("evidence_tier", {}),
        ("scope", True),
    ],
)
def test_malformed_case_fields_raise_contract_errors(definitions, field, value):
    studies, metrics, cases = definitions
    case = deepcopy(cases["bank.reia_peaks.v1"])
    case[field] = value
    with pytest.raises(ContractError):
        validate_case(case, studies, metrics)


@pytest.mark.parametrize(
    "field,value",
    [
        ("study_id", []),
        ("case_id", {}),
        ("outcome", []),
        ("implementation_fingerprint", 42),
        ("result_artifact", [1]),
    ],
)
def test_malformed_event_fields_raise_contract_errors(trial, field, value):
    event, studies, cases = trial
    event[field] = value
    with pytest.raises(ContractError):
        validate_event(event, studies, cases)


def test_import_cannot_attach_an_unrelated_successful_observation(tmp_path, trial):
    event, studies, cases = trial
    result = read_json(verify_artifact(tmp_path, event["result_artifact"]))
    result["observed"] = 17
    wrong = tmp_path / "wrong_result.json"
    wrong.write_text(json.dumps(result))
    event["result_artifact"] = store_result(tmp_path, wrong)
    with pytest.raises(ContractError, match="differs from its result"):
        append_event(tmp_path, event, studies, cases)
    assert load_events(tmp_path, studies, cases) == []


def test_case_revisions_preserve_previous_trials(tmp_path, trial):
    event, studies, cases = trial
    append_event(tmp_path, event, studies, cases)
    revised = deepcopy(cases[event["case_id"]])
    revised.update(id="bank.reia_peaks.v2", revision=2, expected=7)
    cases[revised["id"]] = revised
    assert load_events(tmp_path, studies, cases) == [event]


def test_exact_outcome_cannot_hide_rounding_error(trial):
    event, studies, cases = trial
    case = deepcopy(cases[event["case_id"]])
    case["comparison"].update(kind="absolute", tolerance=0.01)
    cases[case["id"]] = case
    event.update(case_fingerprint=fingerprint(case), observed=6.001)
    with pytest.raises(ContractError, match="Exact outcome"):
        validate_event(event, studies, cases)
    event["outcome"] = "reproduced_with_precision"
    validate_event(event, studies, cases)


def test_duplicate_json_keys_cannot_silently_replace_targets(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"observed": 6, "observed": 17}')
    with pytest.raises(ContractError, match="Duplicate JSON key"):
        read_json(path)


def test_malformed_inventories_raise_contract_errors():
    for kind, field, validate in [
        ("catalog", "studies", validate_catalog),
        ("metrics", "metrics", validate_metrics),
    ]:
        with pytest.raises(ContractError):
            validate({"schema_version": 1, "kind": kind, field: [None]})


def test_cli_resumes_an_alias_and_preserves_closed_checkpoint(tmp_path, capsys):
    from validation.__main__ import main

    catalog, _, cases = load_definitions(REPO / "validation")
    studies = {s["id"] for s in catalog["studies"]}
    event = {
        "schema_version": 1,
        "kind": "event",
        "type": "checkpoint",
        "study_id": "PapkouRM23",
        "recorded_at": "2026-09-30T12:00:00Z",
        "actor": "contract-test",
        "state": "closed_low_yield",
        "last_completed_step": "Read the original methods.",
        "reason": "Definition differs.",
        "next_action": "Await new source.",
        "reopen_when": "Author supplies the missing method.",
    }
    append_event(tmp_path, event, studies, cases)
    assert main(["--store", str(tmp_path), "resume", "Papkou2023"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["study"]["state"] == "closed_low_yield"
    assert output["study"]["reopen_when"] == event["reopen_when"]
    assert output["history"] == [event]


def test_missing_store_cannot_look_like_a_fresh_queue(tmp_path, capsys):
    from validation.__main__ import main

    assert main(["--store", str(tmp_path / "typo"), "queue"]) == 2
    assert "does not exist" in capsys.readouterr().err
