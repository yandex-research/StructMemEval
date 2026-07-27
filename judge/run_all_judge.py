#!/usr/bin/env python3
"""
Batch-run the judge on all eval results.

This script recursively finds all directories named eval_results* (or nested deeper)
that contain files matching results_*.json, runs the judge model on each example,
and saves the judgment outputs preserving the same directory structure under judge/results/.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from httpx import Client
from openai._base_client import DEFAULT_TIMEOUT, DEFAULT_CONNECTION_LIMITS
from tqdm import tqdm

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def load_prompt(prompt_path: str) -> str:
    """Read the judge prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read()


def judge_single(client: OpenAI, model: str, prompt_template: str, result: dict) -> int:
    """
    Judge a single response using the provided LLM.

    Args:
        client: OpenAI client
        model: model name (e.g., gpt-4o)
        prompt_template: template string with placeholders
        result: dict containing 'reference_answer', 'llm_response', 'query'

    Returns:
        1 if the judge says the response is correct, 0 otherwise.
    """
    reference = result['reference_answer']
    prompt = prompt_template.format(
        reference_text=reference['text'],
        llm_response=result['llm_response'],
        question=result["query"]
    )
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=5, temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return 1 if answer == "1" else 0


def get_fold_name(filename: str) -> str:
    """
    Determine the fold/category from the filename.

    Returns one of: 'recommendations', 'accounting', 'graph', 'state_machine'.
    Raises NameError if unknown.
    """
    if "recommendations" in filename:
        return "recommendations"
    elif "accounting" in filename:
        return "accounting"
    elif "graph" in filename:
        return "graph"
    elif "static" in filename or "transition" in filename:
        return "state_machine"
    else:
        raise NameError(f"Unknown fold name in filename: {filename}")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load judge prompt template
    prompt_template = load_prompt(script_dir / "prompt_new_2.txt")

    # Judge model configuration from environment
    model = os.environ.get('JUDGE_MODEL', 'gpt-4o')
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL') or None
    http_client = Client(
        verify=False,
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True
    )
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    print(f"Using judge model: {model}")

    # ------------------------------------------------------------------
    # 1. Find all top-level directories starting with "eval_results"
    #    (excluding those with "test" in the name)
    # ------------------------------------------------------------------
    project_root = script_dir.parent

    eval_top_dirs = sorted(
        path
        for path in project_root.glob("eval_results*")
        if path.is_dir() and "test" not in path.name.lower()
    )

    # ------------------------------------------------------------------
    # 2. Recursively walk each top-level directory and collect every
    #    subdirectory that contains at least one results_*.json file.
    # ------------------------------------------------------------------
    eval_dirs = []

    for top_dir in eval_top_dirs:
        for root, dirs, files in os.walk(top_dir):
            root_path = Path(root)

            dirs[:] = [
                dirname for dirname in dirs
                if "test" not in dirname.lower()
            ]

            if any(
                filename.startswith("results_") and filename.endswith(".json")
                for filename in files
            ):
                eval_dirs.append(root_path)

    eval_dirs = sorted(set(eval_dirs))
    print(f"Found {len(eval_dirs)} directories with result files.")

    # ------------------------------------------------------------------
    # 3. Process each directory independently
    # ------------------------------------------------------------------
    for eval_dir in eval_dirs:
        # Relative path from the project root (benchmark_xtinkt)
        rel_path = eval_dir.relative_to(script_dir.parent)
        # Output directory: judge/results/<relative_path>
        out_dir = results_dir / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find all result files in this directory
        eval_files = sorted(eval_dir.glob("results_*.json"))
        if not eval_files:
            continue

        # Dictionary to accumulate statistics per fold (category)
        # Structure: { fold_name: [pass_count, total_count, pass_rate] }
        fold_stats = {}

        # Process each results_*.json file
        for eval_file in eval_files:
            # Determine the fold (category) from the filename
            try:
                fold_name = get_fold_name(eval_file.name)
            except NameError as e:
                print(f"Skipping {eval_file.name}: {e}")
                continue

            # Build output filename: judge_<original_stem_without_results_>.json
            stem = eval_file.stem  # e.g., results_recommendations
            if stem.startswith("results_"):
                judge_stem = "judge_" + stem[len("results_"):]
            else:
                judge_stem = "judge_" + stem
            out_file = out_dir / f"{judge_stem}.json"

            # Skip if already processed
            if out_file.exists():
                print(f"Skipping {out_file} (already exists)")
                # Still we need to count it for overall stats? We can skip reading.
                # For simplicity, we'll just skip and not update stats.
                continue

            # Load the evaluation data
            with open(eval_file) as f:
                eval_data = json.load(f)

            # Flatten all results from all cases
            all_results = []
            for case in eval_data['cases']:
                for result in case['results']:
                    all_results.append({'case_id': case['case_id'], **result})

            # Judge each example
            scores = []
            details = []
            for result in all_results:
                score = judge_single(client, model, prompt_template, result)
                scores.append(score)
                details.append({
                    'case_id': result.get('case_id', ''),
                    'score': score
                })

            mean_score = sum(scores) / len(scores) if scores else 0

            output = {
                'input_file': str(eval_file.relative_to(script_dir.parent)),
                'num_examples': len(scores),
                'mean_score': mean_score,
                'details': details
            }

            with open(out_file, 'w') as f:
                json.dump(output, f, indent=2)

            # Update statistics for this fold
            if fold_name not in fold_stats:
                fold_stats[fold_name] = [0, 0]  # [passes, total]
            fold_stats[fold_name][0] += sum(scores)
            fold_stats[fold_name][1] += len(scores)

            status = "PASS" if mean_score >= 0.5 else "FAIL"
            print(f"  {out_file.relative_to(results_dir)}: {mean_score:.0%} ({sum(scores)}/{len(scores)}) [{status}]")

        # ------------------------------------------------------------------
        # 4. Write aggregate statistics for this directory (if any files were processed)
        # ------------------------------------------------------------------
        if fold_stats:
            # Calculate pass rates
            total_pass = 0
            total_count = 0
            agg_stats = {}
            for fold, (passes, total) in fold_stats.items():
                rate = passes / total if total > 0 else None
                agg_stats[fold] = [passes, total, rate]
                total_pass += passes
                total_count += total

            # Overall statistics
            overall_rate = total_pass / total_count if total_count > 0 else None
            agg_stats["overall"] = [total_pass, total_count, overall_rate]

            # Save aggregate stats to a JSON file in the same output directory
            agg_file = out_dir / "0judge_total.json"
            with open(agg_file, 'w') as f:
                json.dump(agg_stats, f, indent=2)

            print(f"\nDirectory {rel_path} processed. Aggregate stats saved to {agg_file.relative_to(results_dir)}")
            print(f"  Overall: {total_pass}/{total_count} = {overall_rate:.2%}" if overall_rate is not None else "  Overall: 0/0")

    print("\nAll judging completed. Results are stored in:", results_dir)


if __name__ == "__main__":
    main()