"""Execution contract for literature tests, layered over immutable v1 cases.

No datasets are loaded and no commands or downloads are executed here.
Historical case fingerprints remain unchanged when tests move directories.
"""

from functools import lru_cache
from pathlib import Path

from .contract import (
    ContractError,
    load_definitions,
    matches,
    verify_artifact,
    fingerprint,
)

REPO = Path(__file__).resolve().parents[1]
ROLES = {"paper_result", "author_result", "independent_check", "input_check"}


@lru_cache(maxsize=1)
def definitions():
    catalog, _, cases = load_definitions(REPO / "validation")
    return {study["id"]: study for study in catalog["studies"]}, cases


def evidence_metadata(case_ids, role, studies=None, cases=None):
    """Validate a test's claim, returning references suitable for JUnit reports."""
    if studies is None or cases is None:
        studies, cases = definitions()
    if not isinstance(role, str) or role not in ROLES:
        raise ContractError(f"Unknown evidence role: {role!r}")
    if (
        not isinstance(case_ids, (tuple, list))
        or not case_ids
        or not all(isinstance(case_id, str) for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise ContractError("A literature test needs distinct registered case IDs")
    rows = []
    for case_id in case_ids:
        if case_id not in cases:
            raise ContractError(f"Unregistered literature case: {case_id}")
        case = cases[case_id]
        study = studies[case["study_id"]]
        if not study.get("doi") or not study.get("title") or not study.get("year"):
            raise ContractError(f"Paper DOI, title and year are required: {case_id}")
        if role == "paper_result" and (
            case["evidence_tier"] != "published_numeric"
            or case["definition_match"] != "yes"
        ):
            raise ContractError(
                "A paper-result claim requires a published target and matched definition"
            )
        if role == "author_result" and case["evidence_tier"] == "independent_equation":
            raise ContractError("An independent calculation is not an author result")
        rows.append(
            {
                "case_id": case_id,
                "study_id": case["study_id"],
                "reference": f"{study['title']} ({study['year']}). https://doi.org/{study['doi']}",
                "sources": case["sources"],
                "role": role,
                "evidence_tier": case["evidence_tier"],
                "definition_match": case["definition_match"],
                "scope": case["scope"],
                "case_fingerprint": fingerprint(case),
                "inputs": case["inputs"],
                "comparison": case["comparison"],
            }
        )
    return rows


def case_record(case_id):
    """Return the validated frozen case; do not alter its expected values."""
    return definitions()[1][case_id]


def input_paths(case_id, store=None):
    """Verify every input hash before exposing paths; missing data is an error."""
    paths = []
    for artifact in case_record(case_id)["inputs"]:
        root = REPO if artifact["root"] == "repo" else store
        if root is None:
            raise ContractError(
                f"Case {case_id} needs an explicitly configured external store"
            )
        paths.append(verify_artifact(root, artifact))
    return paths


def assert_case_matches(case_id, observed):
    case = case_record(case_id)
    assert matches(case["expected"], observed, case["comparison"]), (
        f"{case_id}: expected={case['expected']!r}, observed={observed!r}; "
        f"comparison={case['comparison']!r}; sources={case['sources']!r}"
    )
