"""One-time migration of the existing Problems article to source API directives."""

from pathlib import Path
import argparse
import hashlib
import json
import re

from bs4 import BeautifulSoup

DOCS = Path(__file__).resolve().parents[1]


def migrate(source):
    records = []
    blocks = re.split(r"(?=^#{2,3} )", source, flags=re.M)
    output = []
    for block in blocks:
        match = re.search(r'^<span class="method-signature">.*$', block, re.M)
        if not match:
            output.append(block)
            continue
        signature = BeautifulSoup(match[0], "html.parser").get_text()
        name = re.search(r"graphfla\.problems\.(\w+)\(", signature)[1]
        before, body = block[: match.start()], block[match.end() :]
        intro, contract = body.split("**Parameters**", 1)
        introduction = intro.strip()
        records.append(
            {
                "api": "graphfla.problems." + name,
                "introduction_sha256": hashlib.sha256(
                    introduction.encode()
                ).hexdigest(),
                "words": len(introduction.split()),
            }
        )
        extra_before, extra_after = "", ""
        if name == "OptimizationProblem":
            extra_before = (
                contract[contract.index("**Custom problems**") :]
                .strip()
                .removesuffix("---")
                .strip()
            )
        elif name == "NK":
            note = contract[contract.index('!!! note "Statistical signatures"') :]
            extra_before, extra_after = note.split("**Example —", 1)
            extra_after = "**Example —" + extra_after
        elif name == "NumberPartitioning":
            extra_before = (
                contract[contract.index("**Notes**") :]
                .strip()
                .removesuffix("---")
                .strip()
            )
        output.append(
            before
            + introduction
            + "\n\n"
            + (extra_before.strip() + "\n\n" if extra_before else "")
            + "::: graphfla.problems."
            + name
            + "\n\n"
            + (extra_after.strip() + "\n\n" if extra_after else "")
        )
    result = "".join(output)
    result = result.replace(
        "icon: material/file", "api_grouped_classes: true"
    )
    result = result.replace("# :material-file: Problems", "# Problems")
    result = result.replace("(landscape/landscape.md)", "(index.md)").replace(
        "(landscape/boolean_landscape.md)", "(boolean-landscape.md)"
    )
    # Mechanical updates for the finalized analysis API; explanatory prose stays intact.
    result = re.sub(r"\blo_ratio\b", "local_optima_ratio", result)
    result = re.sub(r"\bfitness_distance_corr\b", "fdc", result)
    result = result.replace(
        "approximate=True, sample_cut_prob=0.1", "sample_cut_prob=0.1, seed=0"
    )
    result = result.replace(
        "autocorrelation(landscape)", "autocorrelation(landscape, seed=0)"
    )
    return result, records


def split_pages(text):
    """Keep the two original model families as separate continuous articles."""
    prefix, biological = text.split("## Biological Models\n\n", 1)
    biological, combinatorial = biological.split("## Combinatorial Models\n\n", 1)
    combinatorial, example = combinatorial.split("## Example — Full Pipeline", 1)
    example, references = example.split("## References\n\n", 1)
    reference_lines = references.strip().splitlines()

    # The shared calling convention is authored once and included in both articles.
    opening, rest = prefix.split("All problems inherit", 1)
    interface, rest = ("All problems inherit" + rest).split("## Overview\n\n", 1)
    table, base = rest.split("## Base Class\n\n", 1)
    rows = [line for line in table.splitlines() if line.startswith("|")]
    header, biological_rows, combinatorial_rows = rows[:2], rows[2:8], rows[8:]
    interface = interface.replace(
        "[`OptimizationProblem`](#base-class)",
        "[`OptimizationProblem`][graphfla.problems.OptimizationProblem]",
    )
    include = '--8<-- "problem-interface.md"\n\n'

    opening = opening.replace("# Problems\n", "# Biological Models\n")
    for anchor in ("max-3-sat", "01-knapsack", "number-partitioning"):
        opening = opening.replace(f"(#{anchor})", f"(problems/combinatorial.md#{anchor})")
    biological_page = (
        opening + include + "## Overview\n\n"
        + "\n".join(header + biological_rows) + "\n\n## Base Class\n\n" + base
        + re.sub(r"^### ", "## ", biological, flags=re.M)
        + "## Example — Full Pipeline" + example + "## References\n\n"
        + "\n".join(reference_lines[:3]) + "\n"
    )
    introduction, models = combinatorial.split("### Max-3-SAT", 1)
    combinatorial_page = (
        "---\napi_grouped_classes: true\n---\n\n"
        "# Combinatorial Problems\n\n"
        + introduction.replace("(boolean-landscape.md)", "(../boolean-landscape.md)")
        + include + "## Overview\n\n" + "\n".join(header + combinatorial_rows)
        + "\n\n" + re.sub(r"^### ", "## ", "### Max-3-SAT" + models, flags=re.M)
        + "## References\n\n" + "\n".join(reference_lines[3:]) + "\n"
    )
    return {
        "content/problems.md": biological_page,
        "content/problems/combinatorial.md": combinatorial_page,
        "includes/problem-interface.md": interface,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    text, records = migrate(source)
    if len(records) != 9:
        raise SystemExit(
            f"Expected nine original problem sections; found {len(records)}"
        )
    pages = split_pages(text)
    for relative, content in pages.items():
        destination = DOCS / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    for record in records:
        record["page"] = next(
            path for path, content in pages.items()
            if "::: " + record["api"] + "\n" in content
        )
    (DOCS / "problems-migration.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "source_description": "Original website docs/api/problems.md",
                "package_revision": "23d39d806b0a9bb75278611262d3a921f1079417",
                "pages": [path for path in pages if path.startswith("content/")],
                "introductions": records,
                "example_updates": [
                    "lo_ratio -> local_optima_ratio",
                    "fitness_distance_corr -> fdc",
                    "use current sampling API and explicit seed",
                ],
            },
            indent=2,
        )
        + "\n"
    )
