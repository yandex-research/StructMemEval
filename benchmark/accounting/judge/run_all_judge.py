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
load_dotenv(Path(__file__).parent.parent.parent / ".env")


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
        max_tokens=5, temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return 1 if answer == "1" else 0


def main():
    script_dir = Path(__file__).parent
    eval_dir = script_dir.parent / "eval_results"
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = load_prompt(script_dir / "prompt.txt")

    # Use OpenRouter if available, else OpenAI
    openrouter_key = os.environ.get('LLM_API_KEY')
    if openrouter_key:
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        )
        model = "openai/gpt-4o-mini"
        print("Using OpenRouter")
    else:
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        model = "gpt-4o-mini"
        print("Using OpenAI")

    # Find all transition eval results
    eval_files = sorted(eval_dir.glob("results_transition_*tr_*.json"))
    print(f"Found {len(eval_files)} eval result files")

    # Skip already judged
    existing = {f.stem.replace("judge_", "") for f in results_dir.glob("judge_transition_*tr_*.json")}
    to_judge = []
    for f in eval_files:
        key = f.stem.replace("results_", "")
        if key not in existing:
            to_judge.append(f)

    print(f"Already judged: {len(existing)}, remaining: {len(to_judge)}")

    for eval_file in tqdm(to_judge, desc="Judging files"):
        key = eval_file.stem.replace("results_", "")
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
