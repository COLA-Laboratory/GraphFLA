"""Extract selected public function docs directly from Python source with AST."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "content" / "functions.json"
FUNCTIONS = [
    ("profile", "Analysis", "graphfla.analysis", "graphfla/analysis/profile.py"),
    ("r_s_ratio", "Analysis", "graphfla.analysis", "graphfla/analysis/ruggedness.py"),
    ("classify_epistasis", "Analysis", "graphfla.analysis", "graphfla/analysis/epistasis/motifs.py"),
    ("draw_fitness_distance_corr", "Plotting", "graphfla.plotting", "graphfla/plotting/plotting.py"),
    ("sobol_sampling", "Sampling", "graphfla.sampling", "graphfla/sampling.py"),
    ("get_lon", "Networks", "graphfla.lon", "graphfla/lon.py"),
]
KEYS = {
    "profile": "profile",
    "r_s_ratio": "r-s-ratio",
    "classify_epistasis": "classify-epistasis",
    "draw_fitness_distance_corr": "draw-fitness-distance-corr",
    "sobol_sampling": "sobol-sampling",
    "get_lon": "get-lon",
}


def signature(source, node):
    """Return the original function header, including its return annotation."""
    lines = source.splitlines()
    # The first body statement marks the end of the header, including for
    # signatures that span multiple lines.
    header = "\n".join(lines[node.lineno - 1 : node.body[0].lineno - 1])
    header = header.strip()
    return header.removeprefix("def ").removesuffix(":")


def extract():
    records = []
    for name, category, module, relative_path in FUNCTIONS:
        path = ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
        matches = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one top-level definition for {name} in {relative_path}; found {len(matches)}")
        node = matches[0]
        records.append({
            "key": KEYS[name],
            "category": category,
            "name": name,
            "module": module,
            "kind": "function",
            "signature": signature(source, node),
            "docstring": ast.get_docstring(node, clean=False),
            "source": relative_path,
            "lineno": node.lineno,
            "methods": [],
            "properties": [],
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = [item["name"] for item in records if item["docstring"] is None]
    print(f"Wrote {len(records)} functions to {OUTPUT}")
    print(f"Missing docstrings: {', '.join(missing) if missing else 'none'}")


if __name__ == "__main__":
    extract()
