"""One entry point for building, serving and checking the documentation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS / "_support"))
from checks import site_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "serve", "check"])
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Explicit alternate package checkout; defaults to this checkout",
    )
    args = parser.parse_args()
    env = os.environ.copy()
    if args.source_root:
        env["GRAPHFLA_DOCS_SOURCE_ROOT"] = str(args.source_root.resolve())
    if args.command == "check":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(DOCS / "tests"),
                "-v",
            ],
            env=env,
            check=True,
        )
    command = [
        sys.executable,
        "-m",
        "mkdocs",
        "serve" if args.command == "serve" else "build",
        "-f",
        str(DOCS / "mkdocs.yml"),
    ]
    if args.command != "serve":
        command.append("--strict")
    subprocess.run(command, env=env, check=True)
    if args.command == "check":
        errors = site_errors(DOCS / ".build/site")
        if errors:
            raise SystemExit("\n".join(errors))
        print("Built-site links and anchors: OK")


if __name__ == "__main__":
    main()
