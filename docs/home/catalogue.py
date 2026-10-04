"""Inventory the repository's measured landscape tables and data-card collections."""
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"BioSequence": "Biological sequences", "ChemBio": "Chemical biology",
          "Chemistry": "Chemistry", "Materials": "Materials", "Microbiome": "Microbiology",
          "Pharmacology": "Pharmacology"}
BASE = "https://github.com/COLA-Laboratory/GraphFLA/tree/main/"


def inventory():
    groups = []
    for folder, title in LABELS.items():
        root = ROOT / "data" / folder
        sources = sorted(root.glob("*.csv")) if folder == "BioSequence" else sorted(root.rglob("DATA_CARD.md"))
        entries = []
        for path in sources:
            name = path.stem.replace("_", " ") if path.suffix == ".csv" else path.parent.name
            target = path if path.suffix == ".csv" else path.parent
            entries.append({"name": name, "url": BASE + quote(target.relative_to(ROOT).as_posix())})
        groups.append({"title": title, "entries": entries, "count": len(entries)})
    return groups


def markdown():
    groups = inventory()
    lines = ["# Datasets", "", "Explore the experimental data distributed with GraphFLA. "
             f"The collection contains **{groups[0]['count']} biological sequence datasets and "
             f"{sum(group['count'] for group in groups[1:])} experimental collections** "
             "across six research domains. Individual sequence datasets may describe different assays "
             "or subsets from the same study; a collection can include measurements and supporting lookup tables.",
             "", "[Worked dataset tutorials](tutorials/index.md) show how to prepare and analyze nine "
             "experimental and computational examples. [Reproducible data examples](demo-methods.md) "
             "provide five small, reproducible inputs, including a generated hyperparameter search.", "",
             "Repository links below open the original tables or collection folders. Data cards and "
             "tutorials record sources, response definitions and interpretation limits.", ""]
    for group in groups:
        lines += [f"## {group['title']}", "", f"{group['count']} " + ("datasets" if group == groups[0] else "collections") + ".", ""]
        lines += [f"- [{item['name']}]({item['url']})" for item in group["entries"]]
        lines.append("")
    return "\n".join(lines)
