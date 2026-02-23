#!/usr/bin/env python3
"""
Batch-run the judge on all eval results.

Usage:
    cd benchmark_xtinkt/judge
    python run_all_judge.py
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, 'r') as f:
        return f.read()


def judge_single(client: OpenAI, model: str, prompt_template: str, result: dict) -> int:
    reference = result['reference_answer']
    prompt = prompt_template.format(
        reference_text=reference['text'],
        llm_response=result['llm_response']
    )
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=5, temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return 1 if answer == "1" else 0


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = load_prompt(script_dir / "prompt.txt")

    # Judge model from env (default: gpt-4o)
    model = os.environ.get('JUDGE_MODEL', 'gpt-4o')
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL') or None
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Using judge model: {model}")

    # Scan all eval_results* directories (supports nested experiment subdirs)
    eval_top_dirs = sorted(script_dir.parent.glob("eval_results*"))
    eval_top_dirs = [d for d in eval_top_dirs if d.is_dir() and "test" not in d.name]

    eval_dirs = []
    for top_dir in eval_top_dirs:
        # Check for experiment subdirectories (new format)
        subdirs = [d for d in sorted(top_dir.iterdir()) if d.is_dir() and not d.name.startswith('.')]
        if subdirs:
            eval_dirs.extend(subdirs)
        else:
            # Flat directory (old format)
            eval_dirs.append(top_dir)

    eval_files = []
    for eval_dir in eval_dirs:
        eval_files.extend(sorted(eval_dir.glob("results_*.json")))

    print(f"Scanning dirs: {[d.name for d in eval_dirs]}")
    print(f"Found {len(eval_files)} eval result files")

    # Skip already judged
    existing = {f.stem.replace("judge_", "") for f in results_dir.glob("judge_*.json")}
    to_judge = []
    for f in eval_files:
        key = f.stem.replace("results_", "")
        # Include experiment/dir name for unique keys
        parent = f.parent
        grandparent = parent.parent
        if grandparent.name.startswith("eval_results"):
            # Nested: eval_results/gpt-4o-mini/results_*.json
            key = f"{parent.name}_{key}"
        elif parent.name != "eval_results":
            dir_suffix = parent.name.replace("eval_results", "")
            key = key + dir_suffix
        if key not in existing:
            to_judge.append((f, key))

    print(f"Already judged: {len(existing)}, remaining: {len(to_judge)}")

    for eval_file, key in tqdm(to_judge, desc="Judging files"):
        with open(eval_file) as f:
            eval_data = json.load(f)

        all_results = []
        for case in eval_data['cases']:
            for result in case['results']:
                all_results.append({'case_id': case['case_id'], **result})

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

        output_path = results_dir / f"judge_{key}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        status = "PASS" if mean_score >= 0.5 else "FAIL"
        print(f"  {key}: {mean_score:.0%} ({sum(scores)}/{len(scores)}) [{status}]")

    print(f"\nDone! Results in {results_dir}")


if __name__ == "__main__":
    main()
