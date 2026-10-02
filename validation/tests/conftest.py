"""Fail closed on uncited/unregistered literature tests; permit targeted runs."""

import json
from pathlib import Path

import pytest

from validation.contract import ContractError
from validation.testing import definitions, evidence_metadata, input_paths


def pytest_addoption(parser):
    parser.addoption(
        "--literature-store",
        type=Path,
        help="Explicit root for cases with external inputs; no downloads",
    )
    parser.addoption(
        "--literature-study",
        action="append",
        default=[],
        help="Run only these registered study IDs (repeatable)",
    )
    parser.addoption(
        "--literature-case",
        action="append",
        default=[],
        help="Run only these frozen case IDs (repeatable)",
    )


def pytest_collection_modifyitems(config, items):
    studies, cases = definitions()
    study_filter = set(config.getoption("--literature-study"))
    case_filter = set(config.getoption("--literature-case"))
    if study_filter - studies.keys() or case_filter - cases.keys():
        raise pytest.UsageError("Unknown literature study/case filter")
    selected, deselected = [], []
    for item in items:
        marks = list(item.iter_markers("literature_case"))
        if not marks:
            raise pytest.UsageError(f"Missing literature_case contract: {item.nodeid}")
        rows = []
        try:
            for mark in marks:
                if set(mark.kwargs) != {"role"}:
                    raise ContractError(
                        "literature_case requires exactly the role keyword"
                    )
                rows.extend(
                    evidence_metadata(
                        mark.args, mark.kwargs.get("role"), studies, cases
                    )
                )
        except ContractError as exc:
            raise pytest.UsageError(f"{item.nodeid}: {exc}") from exc
        ids = {row["case_id"] for row in rows}
        sids = {row["study_id"] for row in rows}
        if (study_filter and not study_filter & sids) or (
            case_filter and not case_filter & ids
        ):
            deselected.append(item)
            continue
        item.user_properties.append(("literature_evidence", json.dumps(rows)))
        selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_report_header(config):
    return "Literature validation: hash-pinned inputs; explicit paper/author/independent evidence roles"


@pytest.fixture(scope="session", autouse=True)
def verified_literature_inputs(request):
    """Verify selected inputs before module fixtures can calculate results."""
    store = request.config.getoption("--literature-store")
    case_ids = {
        case_id
        for item in request.session.items
        for mark in item.iter_markers("literature_case")
        for case_id in mark.args
    }
    return {
        case_id: tuple(input_paths(case_id, store=store))
        for case_id in sorted(case_ids)
    }


@pytest.fixture(autouse=True)
def literature_inputs(request, verified_literature_inputs):
    """Return this test's verified paths; fixtures are read-only."""
    return {
        case_id: verified_literature_inputs[case_id]
        for mark in request.node.iter_markers("literature_case")
        for case_id in mark.args
    }
