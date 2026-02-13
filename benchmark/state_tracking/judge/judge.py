#!/usr/bin/env python3
"""
Judge Tool для оценки результатов бенчмарка.
Бинарная оценка соответствия ответа референсу.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import yaml
from openai import OpenAI
from tqdm import tqdm

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def load_config(config_path: str) -> dict:
    """Load YAML config with environment variable substitution"""
    with open(config_path, 'r') as f:
        config_str = f.read()
    config_str = os.path.expandvars(config_str)
    return yaml.safe_load(config_str)


def load_prompt(prompt_path: str) -> str:
    """Load judge prompt template"""
    with open(prompt_path, 'r') as f:
        return f.read()


def load_eval_results(input_path: str) -> dict:
    """Load evaluation results JSON"""
    with open(input_path, 'r') as f:
        return json.load(f)


def judge_single(client: OpenAI, model: str, prompt_template: str, result: dict) -> int:
    """Judge a single example. Returns 1 (match) or 0 (no match)."""
    reference = result['reference_answer']

    prompt = prompt_template.format(
        reference_text=reference['text'],
        llm_response=result['llm_response']
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0
    )

    answer = response.choices[0].message.content.strip()
    return 1 if answer == "1" else 0


def main():
    parser = argparse.ArgumentParser(description="Judge Tool for benchmark evaluation")
    parser.add_argument('--config', required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    script_dir = Path(__file__).parent

    prompt = load_prompt(script_dir / config['prompt_path'])
    eval_data = load_eval_results(script_dir.parent / config['input_path'])

    client = OpenAI(api_key=config['api_key'])

    # Flatten cases[].results[] into single list
    all_results = []
    for case in eval_data['cases']:
        for result in case['results']:
            all_results.append({'case_id': case['case_id'], **result})

    scores = []
    details = []

    print(f"Judging {len(all_results)} examples...")

    for result in tqdm(all_results, desc="Judging"):
        score = judge_single(client, config['model'], prompt, result)
        scores.append(score)
        details.append({
            'case_id': result.get('case_id', ''),
            'query_id': result.get('query_id', ''),
            'score': score
        })

    mean_score = sum(scores) / len(scores) if scores else 0

    output = {
        'input_file': config['input_path'],
        'num_examples': len(scores),
        'mean_score': mean_score,
        'details': details
    }

    output_path = script_dir / config['output_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults: {mean_score:.2%} ({sum(scores)}/{len(scores)})")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
