#!/usr/bin/env python3
"""
Aggregate Results for Bar Chart

Collects judge results from multiple configurations and outputs
JSON formatted for plotting comparison charts.

Usage:
    cd benchmark_xtinkt
    python aggregate_results.py --group static
    python aggregate_results.py --group transition
    python aggregate_results.py  # all groups
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def find_judge_results(judge_dir: Path, group: str = None) -> list[dict]:
    """Find all judge result files, optionally filtered by group."""
    results = []

    for file in judge_dir.glob("judge_*.json"):
        # Parse filename: judge_{group}_{config}.json
        parts = file.stem.split("_")
        if len(parts) >= 3:
            file_group = parts[1]
            config_name = "_".join(parts[2:])

            if group and file_group != group:
                continue

            with open(file, "r") as f:
                data = json.load(f)

            results.append({
                "file": str(file),
                "group": file_group,
                "config": config_name,
                "mean_score": data.get("mean_score", 0),
                "num_examples": data.get("num_examples", 0),
                "details": data.get("details", [])
            })

    return results


def aggregate_by_group(results: list[dict]) -> dict:
    """Aggregate results by group for chart output."""
    groups = {}

    for r in results:
        group = r["group"]
        config = r["config"]

        if group not in groups:
            groups[group] = {"group": group, "results": {}}

        groups[group]["results"][config] = {
            "accuracy": r["mean_score"],
            "examples": r["num_examples"]
        }

    return groups


def print_table(groups: dict):
    """Print results as ASCII table."""
    for group_name, group_data in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Group: {group_name}")
        print("="*60)
        print(f"{'Config':<25} {'Accuracy':>10} {'Examples':>10}")
        print("-"*60)

        for config, data in sorted(group_data["results"].items()):
            acc = f"{data['accuracy']:.1%}"
            print(f"{config:<25} {acc:>10} {data['examples']:>10}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate judge results")
    parser.add_argument("--group", choices=["static", "transition"],
                        help="Filter by group")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--judge-dir", default="judge/results",
                        help="Directory with judge results")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    judge_dir = script_dir / args.judge_dir

    if not judge_dir.exists():
        print(f"Judge results directory not found: {judge_dir}")
        print("Run judge first: python -m judge.judge --config judge/config.yaml")
        return

    # Find and aggregate results
    results = find_judge_results(judge_dir, args.group)

    if not results:
        print(f"No judge results found in {judge_dir}")
        return

    groups = aggregate_by_group(results)

    # Print table
    print_table(groups)

    # Output JSON
    output = {
        "aggregated_at": datetime.now().isoformat(),
        "groups": list(groups.values())
    }

    if args.output:
        output_path = script_dir / args.output
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved to {output_path}")
    else:
        print("\n--- JSON Output ---")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
