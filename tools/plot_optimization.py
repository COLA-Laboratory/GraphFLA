"""Plot runtime and process peak RSS for recorded construction rounds."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rounds", type=Path, nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/optimization.png")
    )
    args = parser.parse_args()
    rounds = [json.loads(path.read_text()) for path in args.rounds]
    available = list(rounds[0]["datasets"])
    datasets = args.datasets or available
    for result in rounds[1:]:
        if set(result["datasets"]) != set(available):
            raise ValueError("Plot requires the same datasets in every round")
        for name in datasets:
            if (
                result["datasets"][name]["input_sha256"]
                != rounds[0]["datasets"][name]["input_sha256"]
            ):
                raise ValueError(f"Input changed between rounds: {name}")
    labels = [result["label"] for result in rounds]
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    rows = math.ceil(len(datasets) / 3)
    figure, axes = plt.subplots(
        rows, 3, figsize=(15, 3 * rows), constrained_layout=True, squeeze=False
    )
    x = np.arange(len(rounds))
    for axis, dataset in zip(axes.ravel(), datasets):
        values = [result["datasets"][dataset] for result in rounds]
        times = np.array([value["median_seconds"] for value in values])
        memory = np.array([value["median_peak_rss_bytes"] for value in values])
        spread = np.array([value["mad_seconds"] for value in values])
        axis.plot(
            x, times / times[0], "o-", color="#126e9b", label="Runtime / baseline"
        )
        axis.plot(
            x, memory / memory[0], "s--", color="#b14b29", label="Peak RSS / baseline"
        )
        axis.fill_between(
            x,
            (times - spread) / times[0],
            (times + spread) / times[0],
            color="#126e9b",
            alpha=0.15,
        )
        rejected = [
            i for i, label in enumerate(labels) if label in {"attributes", "bounded"}
        ]
        axis.scatter(
            x[rejected],
            (times / times[0])[rejected],
            marker="x",
            color="black",
            s=35,
            zorder=3,
            label="Superseded trial",
        )
        axis.axhline(1, color="0.6", linewidth=0.7)
        axis.set_title(
            f"{dataset}\n{times[0] * 1000:.1f} → {times[-1] * 1000:.1f} ms", loc="left"
        )
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.18)
    for axis in axes.ravel()[len(datasets) :]:
        axis.set_visible(False)
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside upper right", frameon=False)
    figure.suptitle(
        "GraphFLA construction optimization\nLower is better; runtime band = ±MAD; each dataset normalized to its baseline",
        fontsize=17,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    figure.savefig(args.output.with_suffix(".svg"))
    plt.close(figure)


if __name__ == "__main__":
    main()
