"""Fail-closed literature execution, without loading any empirical dataset."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from validation.contract import ContractError, fingerprint
from validation import testing

pytest_plugins = ["pytester"]
REPO = Path(__file__).resolve().parents[1]
CASE = "wagner.protein.ee.general.v1"


def test_case_metadata_carries_reference_and_frozen_identity():
    (row,) = testing.evidence_metadata([CASE], "independent_check")
    assert "10.1038/s41467-023-39321-8" in row["reference"]
    assert row["case_fingerprint"] == fingerprint(testing.case_record(CASE))
    assert row["sources"] and row["inputs"] and row["comparison"]


@pytest.mark.parametrize(
    "case_ids, role, message",
    [
        ([], "input_check", "distinct"),
        ([CASE, CASE], "input_check", "distinct"),
        (["unknown"], "input_check", "Unregistered"),
        ([CASE], "made_up", "Unknown evidence"),
        ([CASE], "paper_result", "published target"),
        ([CASE], "author_result", "not an author"),
    ],
)
def test_invalid_evidence_claims_fail(case_ids, role, message):
    with pytest.raises(ContractError, match=message):
        testing.evidence_metadata(case_ids, role)


def test_incomplete_citation_and_unmatched_definition_fail():
    studies, cases = deepcopy(testing.definitions())
    studies["Wagner2023"]["doi"] = None
    with pytest.raises(ContractError, match="DOI"):
        testing.evidence_metadata([CASE], "input_check", studies, cases)
    studies, cases = deepcopy(testing.definitions())
    cases[CASE].update(evidence_tier="published_numeric", definition_match="no")
    with pytest.raises(ContractError, match="matched definition"):
        testing.evidence_metadata([CASE], "paper_result", studies, cases)


def test_pinned_inputs_and_external_store_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "tiny.txt"
    path.write_text("independent source\n")
    artifact = {
        "root": "store",
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(testing, "case_record", lambda _: {"inputs": [artifact]})
    with pytest.raises(ContractError, match="explicitly configured"):
        testing.input_paths("synthetic")
    assert testing.input_paths("synthetic", tmp_path) == [path]
    path.write_text("tampered\n")
    with pytest.raises(ContractError, match="Changed artifact"):
        testing.input_paths("synthetic", tmp_path)
    path.unlink()
    with pytest.raises(ContractError, match="Missing artifact"):
        testing.input_paths("synthetic", tmp_path)


def test_wrong_observation_is_not_silently_accepted():
    case = testing.case_record(CASE)
    testing.assert_case_matches(CASE, case["expected"])
    wrong = {**case["expected"], "beneficial": -1}
    with pytest.raises(AssertionError, match="sources="):
        testing.assert_case_matches(CASE, wrong)


@pytest.fixture
def harness(pytester, monkeypatch):
    # Test the actual hooks in a fresh process. Replace only input acquisition
    # with a tiny synthetic artifact, so this basic test loads no paper data.
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    pytester.makeini(
        "[pytest]\naddopts = --strict-markers\nmarkers =\n    literature_case(*ids, role): citation contract\n"
    )
    pytester.makeconftest("""
from pathlib import Path
import hashlib
from validation.tests.conftest import *
import validation.tests.conftest as harness
from validation.contract import verify_artifact

def tiny_input(case_id, store):
    root = Path(__file__).parent
    artifact = {"path": "source.txt", "sha256": hashlib.sha256(b"source").hexdigest()}
    return [verify_artifact(root, artifact)]
harness.input_paths = tiny_input
""")
    (pytester.path / "source.txt").write_bytes(b"source")
    return pytester


@pytest.mark.parametrize(
    "decorator, message",
    [
        ("", "Missing literature_case contract"),
        ('@pytest.mark.literature_case("unknown", role="input_check")', "Unregistered"),
        (f'@pytest.mark.literature_case("{CASE}")', "role keyword"),
        (
            f'@pytest.mark.literature_case("{CASE}", role="paper_result")',
            "published target",
        ),
    ],
)
def test_collection_rejects_unregistered_or_false_claims(harness, decorator, message):
    harness.makepyfile(f"import pytest\n{decorator}\ndef test_claim(): pass\n")
    result = harness.runpytest_subprocess("--collect-only", "-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert message in result.stderr.str()


def test_filters_and_junit_preserve_evidence(harness):
    harness.makepyfile(f'''
import pytest
@pytest.mark.literature_case("{CASE}", role="independent_check")
def test_ee(literature_inputs):
    assert literature_inputs["{CASE}"][0].read_bytes() == b"source"
@pytest.mark.literature_case("lyons.trna.iid.v1", role="paper_result")
def test_other(): raise AssertionError("should be deselected")
''')
    result = harness.runpytest_subprocess(
        "--literature-study",
        "Wagner2023",
        "--literature-case",
        CASE,
        "--junitxml=result.xml",
        "-q",
    )
    result.assert_outcomes(passed=1, deselected=1)
    report = ET.parse(harness.path / "result.xml")
    (prop,) = report.findall(".//property[@name='literature_evidence']")
    (row,) = json.loads(prop.attrib["value"])
    assert row["case_id"] == CASE and row["role"] == "independent_check"
    assert row["case_fingerprint"] == fingerprint(testing.case_record(CASE))
    result = harness.runpytest_subprocess(
        "--collect-only", "--literature-study", "unknown", "-q"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "Unknown literature" in result.stderr.str()
    result = harness.runpytest_subprocess(
        "--collect-only",
        "--literature-study",
        "Lyons2020",
        "--literature-case",
        CASE,
        "-q",
    )
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED


def test_autouse_verification_prevents_test_body_on_corrupt_input(harness):
    harness.makepyfile(f'''
import pytest
@pytest.mark.literature_case("{CASE}", role="independent_check")
def test_claim(): raise AssertionError("body must not execute")
''')
    (harness.path / "source.txt").write_text("tampered")
    result = harness.runpytest_subprocess("-q")
    result.assert_outcomes(errors=1)
    assert "Changed artifact" in result.stdout.str()
    assert "AssertionError: body must not execute" not in result.stdout.str()
