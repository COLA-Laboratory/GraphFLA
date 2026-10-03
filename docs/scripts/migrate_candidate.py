"""One-time migration from frozen G narrative prototypes to source references.

Existing conceptual prose stays verbatim. Hand-copied API signatures/contracts,
per-function references and per-function examples are replaced by source data.
The integrated tutorial Examples section remains authored Markdown.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "fitness_distance_corr": "fdc",
    "basin_fit_corr": "basin_fitness_correlation",
    "evol_enhance_mutations": "evolvability_enhancing_mutations",
    "mean_path_lengths": "mean_path_length_to_local_optima",
    "mean_path_lengths_go": "mean_path_length_to_global_optimum",
    "mean_dist_lo": "mean_distance_to_local_optima",
    "mean_dist_go": "mean_distance_to_global_optimum",
    "lo_ratio": "local_optima_ratio",
    "neighbor_fit_corr": "neighbor_fitness_correlation",
    "walsh_hadamard_coefficient": "walsh_hadamard",
    "gamma_statistic": "gamma",
    "extradimensional_bypass_analysis": "extradimensional_bypass",
}


def migrate(source):
    parts = re.split(r"^```api\n([^\n]+)\n```\s*\n", source, flags=re.M)
    out = parts[0].replace(
        "icon: material/cube-outline",
        "icon: material/cube-outline\napi_narrative: true",
    )
    for signature, following in zip(parts[1::2], parts[2::2]):
        old = re.search(r"graphfla\.analysis\.(\w+)\(", signature)[1]
        name = NAMES.get(old, old)
        first_contract = re.search(r"^!!! api-(?:parameters|returns) ", following, re.M)
        if not first_contract:
            raise ValueError(f"Missing API boundary for {old}")
        intro = following[: first_contract.start()]
        next_heading = re.search(r"^#{2,3} ", following[first_contract.start() :], re.M)
        tail = (
            following[first_contract.start() + next_heading.start() :]
            if next_heading
            else ""
        )
        out += intro.rstrip() + "\n\n::: graphfla.analysis." + name + "\n\n" + tail
    for old, new in NAMES.items():
        out = out.replace("[`" + old + "`]", "[`" + new + "`]")
    return out


if __name__ == "__main__":
    for name in ("epistasis", "navigability", "robustness", "ruggedness"):
        source = (ROOT / "candidates/narratives" / f"{name}.md").read_text()
        result = migrate(source)
        (ROOT / "content/analysis" / f"{name}.md").write_text(result)
        print(name, result.count("::: graphfla.analysis."))
