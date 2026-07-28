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
from httpx import Client
from openai._base_client import DEFAULT_TIMEOUT, DEFAULT_CONNECTION_LIMITS
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
        llm_response=result['llm_response'],
        question=result["query"]
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

    prompt_template = load_prompt(script_dir / "prompt_new_2.txt")

    # Judge model from env (default: gpt-4o)
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

        
        for eval_dir in eval_dirs:
            eval_files = []
            eval_files.extend(sorted(eval_dir.glob("results_*.json")))

            print(f"Scanning dirs: {[d.name for d in eval_dirs]}")
            print(f"Found {len(eval_files)} eval result files")

            # Skip already judged (per file, so new results in an
            # already-judged experiment dir still get picked up)
            to_judge = []
            for f in eval_files:
                key = f.stem.replace("results_", "")
                # Include experiment/dir name for unique keys
                parent = f.parent
                    # Nested: eval_results/gpt-4o-mini/results_*.json
                key = f"{parent.name}/judge_{key}.json"
                if (results_dir / key).exists():
                    continue
                to_judge.append((f, key))

            print(f"To judge: {len(to_judge)}")
            successes = {"recommendations": [0, 0], "accounting": [0, 0], "graph": [0, 0], "state_machine": [0, 0]}
            for eval_file, key in tqdm(to_judge, desc="Judging files"):
                with open(eval_file) as f:
                    eval_data = json.load(f)
                print(eval_file.name)
                if "recommendations" in eval_file.name:
                    fold_name = "recommendations"
                elif "accounting" in eval_file.name:
                    fold_name = "accounting"
                elif "graph" in eval_file.name:
                    fold_name = "graph"
                elif "static" in eval_file.name or "transition" in eval_file.name:
                    fold_name = "state_machine"
                else:
                    raise NameError("Wrong fold name") 

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

                if not os.path.exists(results_dir / parent.name):
                    os.makedirs(results_dir / parent.name)

                output_path = results_dir / f"{key}"
                with open(output_path, 'w') as f:
                    json.dump(output, f, indent=2)
                successes[fold_name][1] += 1
                if mean_score >= 0.5:
                    status = "PASS"
                    successes[fold_name][0] += 1
                else:
                    status = "FAIL"
                
                print(f"  {key}: {mean_score:.0%} ({sum(scores)}/{len(scores)}) [{status}]")
            if len(to_judge) > 0:
                print(f"\nDirectory {results_dir.name} done! Results in {results_dir}. Newly judged {successes} / {len(to_judge)}")
                # Recompute totals from ALL judge files of this dir, not just
                # the newly judged ones — incremental runs would otherwise
                # overwrite 0judge_total.json with partial counts.
                totals = {"recommendations": [0, 0], "accounting": [0, 0], "graph": [0, 0], "state_machine": [0, 0]}
                for jf in sorted((results_dir / eval_dir.name).glob("judge_*.json")):
                    if "recommendations" in jf.name:
                        fold_name = "recommendations"
                    elif "accounting" in jf.name:
                        fold_name = "accounting"
                    elif "graph" in jf.name:
                        fold_name = "graph"
                    elif "static" in jf.name or "transition" in jf.name:
                        fold_name = "state_machine"
                    else:
                        continue
                    with open(jf) as f:
                        mean_score = json.load(f).get('mean_score', 0)
                    totals[fold_name][1] += 1
                    if mean_score >= 0.5:
                        totals[fold_name][0] += 1
                for key, value in totals.items():
                    value.append(value[0] / value[1] if value[1] else None)
                with open(results_dir / eval_dir.name / "0judge_total.json", 'w') as f:
                    json.dump(totals, f, indent=2)


if __name__ == "__main__":
    main()
