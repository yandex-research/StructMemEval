#!/usr/bin/env python3
"""
Plot benchmark results from judge output files.

Usage:
    cd benchmark_xtinkt/judge
    python plot_results.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_judge_results(results_dir: Path) -> dict:
    """Load all judge results and organize by (case_id, config).

    Returns: {case_id: {config_name: score}}
    """
    results = {}
    for f in sorted(results_dir.glob("judge_transition_*tr_*.json")):
        data = json.loads(f.read_text())
        name = f.stem.replace("judge_", "")

        m = re.match(r"(transition_\d+tr_\d+)_(.*)", name)
        if not m:
            continue
        case_id = m.group(1)
        config_name = m.group(2)
        score = data.get("mean_score", 0)

        if case_id not in results:
            results[case_id] = {}
        results[case_id][config_name] = score

    return results


def aggregate_by_transitions(results: dict, configs: list[str]) -> dict:
    """Aggregate scores by number of transitions.

    Returns: {num_transitions: {config: (mean, count, total)}}
    """
    # Group cases by transition count
    groups = {}  # {num_tr: [case_ids]}
    for case_id in results:
        m = re.match(r"transition_(\d+)tr_\d+", case_id)
        if m:
            num_tr = int(m.group(1))
            groups.setdefault(num_tr, []).append(case_id)

    agg = {}
    for num_tr in sorted(groups):
        agg[num_tr] = {}
        for config in configs:
            scores = []
            for case_id in groups[num_tr]:
                if config in results.get(case_id, {}):
                    scores.append(results[case_id][config])
            if scores:
                agg[num_tr][config] = (
                    sum(scores) / len(scores),
                    sum(int(s) for s in scores),
                    len(scores),
                )
    return agg


def plot_bar_chart(agg: dict, configs: list[str], labels: list[str], colors: list[str]):
    """Plot grouped bar chart: accuracy by number of transitions."""
    tr_counts = sorted(agg.keys())
    x = np.arange(len(tr_counts))
    width = 0.18
    n = len(configs)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (config, label, color) in enumerate(zip(configs, labels, colors)):
        vals = []
        for tr in tr_counts:
            entry = agg[tr].get(config)
            vals.append(entry[0] * 100 if entry else 0)
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Number of State Transitions", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Memory System Accuracy by Number of State Transitions", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{tr} transitions" for tr in tr_counts])
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_detail_table(results: dict, configs: list[str], labels: list[str]):
    """Print detailed results table."""
    # Sort cases
    def sort_key(case):
        m = re.match(r"transition_(\d+)tr_(\d+)", case)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    cases = sorted(results.keys(), key=sort_key)

    header = f"{'Case':<22}" + "".join(f"{l:<16}" for l in labels)
    print(header)
    print("-" * len(header))

    totals = {c: [] for c in configs}
    current_group = None

    for case in cases:
        m = re.match(r"transition_(\d+)tr", case)
        group = m.group(1) if m else ""
        if group != current_group:
            if current_group is not None:
                print()
            current_group = group

        row = f"{case:<22}"
        for c in configs:
            val = results.get(case, {}).get(c)
            if val is not None:
                totals[c].append(val)
                mark = "PASS" if val >= 0.5 else "FAIL"
                row += f"{mark:<16}"
            else:
                row += f"{'---':<16}"
        print(row)

    print()
    print("-" * len(header))
    row = f"{'TOTAL':<22}"
    for c, l in zip(configs, labels):
        if totals[c]:
            s = sum(int(x) for x in totals[c])
            n = len(totals[c])
            row += f"{s}/{n} ({s/n:.0%}){'':>6}"
        else:
            row += f"{'N/A':<16}"
    print(row)


def main():
    results_dir = Path(__file__).parent / "results"

    configs = ["mem0_top5", "mem0_top20", "mem0_agent_hint", "mem_agent_hint"]
    labels = ["mem0 top-5", "mem0 top-20", "mem0 agent", "mem-agent"]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    results = load_all_judge_results(results_dir)

    if not results:
        print("No judge results found in", results_dir)
        return

    # Print table
    print("\n=== Detailed Results ===\n")
    plot_detail_table(results, configs, labels)

    # Aggregate and plot
    agg = aggregate_by_transitions(results, configs)

    print("\n\n=== Aggregated by Transitions ===\n")
    header = f"{'Transitions':<14}" + "".join(f"{l:<16}" for l in labels)
    print(header)
    print("-" * len(header))
    for tr in sorted(agg):
        row = f"{tr} tr{'':>9}"
        for c in configs:
            entry = agg[tr].get(c)
            if entry:
                mean, correct, total = entry
                row += f"{correct}/{total} ({mean:.0%}){'':>6}"
            else:
                row += f"{'N/A':<16}"
        print(row)

    fig = plot_bar_chart(agg, configs, labels, colors)
    out_path = results_dir / "benchmark_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Chart saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
