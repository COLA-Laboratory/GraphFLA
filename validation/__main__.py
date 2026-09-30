"""Read the queue, validate evidence, or append a reviewed event; never run code."""

import argparse
import json
from pathlib import Path
import sys

from .contract import (
    ContractError,
    append_event,
    load_definitions,
    load_events,
    read_json,
    verify_artifact,
    verify_event_artifacts,
    store_result,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, help="External artifact/event store")
    parser.add_argument("--definitions", type=Path, default=Path(__file__).parent)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check", help="Validate definitions/history; optionally verify artifact bytes"
    )
    check.add_argument("--artifacts", action="store_true")
    queue = sub.add_parser(
        "queue", help="Show resumable work, supplied large studies first"
    )
    queue.add_argument(
        "--min-variants",
        type=int,
        help="Default: 1025 for supplied papers, 0 otherwise",
    )
    queue.add_argument("--include-closed", action="store_true")
    queue.add_argument(
        "--scope", choices=["supplied", "metric_search", "all"], default="supplied"
    )
    resume = sub.add_parser(
        "resume", help="Show existing state and sources before doing more work"
    )
    resume.add_argument("study_id")
    record = sub.add_parser(
        "record", help="Append a validated, content-addressed event"
    )
    record.add_argument("event_file", type=Path)
    archive = sub.add_parser(
        "archive-result", help="Snapshot a result before recording a trial"
    )
    archive.add_argument("result_file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.store is not None and not args.store.is_dir():
            raise ContractError(
                "Artifact store does not exist; check the path before resuming"
            )
        catalog, metrics, cases = load_definitions(args.definitions)
        studies = {s["id"]: s for s in catalog["studies"]}
        events = load_events(args.store, set(studies), cases) if args.store else []
        effective = {k: dict(v) for k, v in studies.items()}
        for event in events:
            if event["type"] == "checkpoint":
                effective[event["study_id"]].update(
                    {
                        k: event[k]
                        for k in (
                            "state",
                            "next_action",
                            "last_completed_step",
                            "reopen_when",
                            "dossier",
                        )
                        if k in event
                    }
                )
        if args.command == "check":
            checked = 0
            if args.artifacts:
                repo = Path(__file__).resolve().parents[1]
                for case in cases.values():
                    for item in case["inputs"]:
                        root = repo if item["root"] == "repo" else args.store
                        if root is None:
                            raise ContractError(
                                "--store is required for external artifacts"
                            )
                        verify_artifact(root, item)
                        checked += 1
                for event in events:
                    checked += verify_event_artifacts(args.store, event)
            print(
                json.dumps(
                    {
                        "studies": len(studies),
                        "metric_families": len(metrics["metrics"]),
                        "cases": len(cases),
                        "events": len(events),
                        "artifacts_checked": checked,
                        "store": str(args.store) if args.store else None,
                    }
                )
            )
        elif args.command == "queue":
            rows = []
            minimum = (
                args.min_variants
                if args.min_variants is not None
                else (1025 if args.scope == "supplied" else 0)
            )
            for study in effective.values():
                size = max(
                    study.get("measured_variants") or 0,
                    study.get("theoretical_variants") or 0,
                )
                if size < minimum or (
                    args.scope != "all" and study["origin"] != args.scope
                ):
                    continue
                if not args.include_closed and (
                    study["state"].startswith("closed_")
                    or study["state"] in {"validated", "triaged"}
                ):
                    continue
                rows.append((size, study))
            for size, study in sorted(rows, key=lambda x: (-x[0], x[1]["id"])):
                print(
                    f"{study['id']}\t{size}\t{study['state']}\t{study['next_action']}"
                )
        elif args.command == "resume":
            matches = [
                s["id"]
                for s in studies.values()
                if args.study_id == s["id"] or args.study_id in s.get("aliases", [])
            ]
            if not matches:
                raise ContractError(f"Unknown study: {args.study_id}")
            study_id = matches[0]
            study = effective[study_id]
            print(
                json.dumps(
                    {
                        "study": study,
                        "cases": [
                            c for c in cases.values() if c["study_id"] == study_id
                        ],
                        "history": [e for e in events if e["study_id"] == study_id],
                        "resume_rule": "Read existing dossier and source indexes. Reuse verified artifacts. Append trials; never reset or replace history.",
                    },
                    indent=2,
                )
            )
        elif args.command == "archive-result":
            if args.store is None:
                raise ContractError("archive-result requires --store")
            print(json.dumps(store_result(args.store, args.result_file)))
        else:
            if args.store is None:
                raise ContractError("record requires --store")
            print(
                append_event(
                    args.store, read_json(args.event_file), set(studies), cases
                )
            )
        return 0
    except (ContractError, OSError) as exc:
        print(f"Validation contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
