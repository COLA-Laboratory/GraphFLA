"""Compare compatible construction rounds without treating timing noise as gains."""

import argparse
import json
from pathlib import Path
import statistics


def process_summary(result):
    medians = [
        statistics.median(worker["seconds"])
        for worker in result["workers"]
        if "seconds" in worker
    ]
    center = statistics.median(medians)
    spread = statistics.median(abs(value - center) for value in medians)
    return center, max(spread, result["mad_seconds"])


def compare(before, after):
    if before["environment"] != after["environment"]:
        raise ValueError(
            "Environment mismatch; rerun both revisions in the same environment"
        )
    if before["protocol"] != after["protocol"]:
        raise ValueError("Measurement protocol mismatch")
    if before["datasets"].keys() != after["datasets"].keys():
        raise ValueError("Dataset selection mismatch")
    rows = []
    for name, left in before["datasets"].items():
        right = after["datasets"][name]
        for key in ("input_sha256", "shape", "variable_sites"):
            if left[key] != right[key]:
                raise ValueError(f"{name}: {key} mismatch")
        a, noise_a = process_summary(left)
        b, noise_b = process_summary(right)
        gate = max(0.05, 3 * (noise_a + noise_b) / a)
        change = (b - a) / a
        status = (
            "faster"
            if change < -gate
            else "slower"
            if change > gate
            else "inconclusive"
        )
        rows.append(
            {
                "dataset": name,
                "baseline_seconds": a,
                "candidate_seconds": b,
                "speedup": a / b,
                "noise_gate": gate,
                "status": status,
                "peak_rss_ratio": right["median_peak_rss_bytes"]
                / left["median_peak_rss_bytes"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = compare(
        json.loads(args.before.read_text()), json.loads(args.after.read_text())
    )
    for row in rows:
        print(
            f"{row['dataset']:22} {row['baseline_seconds']:8.4f} -> {row['candidate_seconds']:8.4f} s "
            f"{row['speedup']:5.2f}x  RSS {row['peak_rss_ratio']:5.2f}x  {row['status']}"
        )
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
