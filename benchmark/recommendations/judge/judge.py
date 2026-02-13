#!/usr/bin/env python3
"""
Judge Tool with paired (case_id, prompt_path) filtering from config file.
No CLI filter args — all filtering defined in YAML config.
"""

import json
import os
from pathlib import Path

import yaml
from openai import OpenAI
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config_str = f.read()
    config_str = os.path.expandvars(config_str)
    return yaml.safe_load(config_str)


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, 'r') as f:
        return f.read()


def load_eval_results(input_path: str) -> dict:
    with open(input_path, 'r') as f:
        return json.load(f)


def judge_single(client: OpenAI, model: str, prompt_template: str, result: dict) -> int:
    reference_text = result['reference_answer']['text']
    llm_response = result['llm_response']

    filled_prompt = prompt_template.format(
        reference_text=reference_text,
        llm_response=llm_response
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": filled_prompt}],
        max_tokens=5,
        temperature=0,
        seed=42
    )
    answer = response.choices[0].message.content.strip()
    return 1 if answer == "1" else 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Judge Tool (filters via config)")
    parser.add_argument('--config', required=True, help="Path to judge config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    script_dir = Path(__file__).parent

    # Resolve paths
    judge_prompt_path = script_dir / config['prompt_path']
    eval_input_path = script_dir.parent / config['input_path']

    # Parse paired filter from config
    allowed_pairs = None
    case_ids = config.get('filter_case_id')
    prompt_paths = config.get('filter_prompt_path')

    if case_ids is not None and prompt_paths is not None:
        if len(case_ids) != len(prompt_paths):
            raise ValueError("Error: filter_case_id and filter_prompt_path must have equal length.")
        allowed_pairs = set(zip(case_ids, prompt_paths))
    elif case_ids is not None or prompt_paths is not None:
        raise ValueError("Error: both filter_case_id and filter_prompt_path must be defined together.")

    # Load resources
    judge_prompt = load_prompt(judge_prompt_path)
    eval_data = load_eval_results(eval_input_path)
    client = OpenAI(api_key=config['api_key'])

    # Flatten and filter
    all_results = []
    for case in eval_data['cases']:
        case_id = case.get('case_id', '')
        prompt_path = case.get('prompt_path', '')
        
        if allowed_pairs is not None and (case_id, prompt_path) not in allowed_pairs:
            continue

        for result in case['results']:
            enriched = {
                'case_id': case_id,
                'prompt_path': prompt_path,
                'query_id': result.get('query_id', ''),
                'question': result.get('query') or result.get('question', ''),
                'llm_response': result['llm_response'],
                'reference_answer': result['reference_answer']
            }
            all_results.append(enriched)

    if not all_results:
        print("⚠️ No examples matched the filter pairs (or no filter → but no data).")
        return

    scores = []
    details = []

    print(f"Judging {len(all_results)} examples...")

    for res in tqdm(all_results, desc="Judging"):
        score = judge_single(client, config['model'], judge_prompt, res)
        scores.append(score)
        details.append({
            'case_id': res['case_id'],
            'prompt_path': res['prompt_path'],
            'query_id': res['query_id'],
            'question': res['question'],
            'llm_response': res['llm_response'],
            'reference_answer': res['reference_answer']['text'],
            'score': score
        })

    mean_score = sum(scores) / len(scores)

    output = {
        'input_file': str(config['input_path']),
        'filters': {
            'paired_case_prompt': [
                {'case_id': cid, 'prompt_path': pp}
                for cid, pp in (allowed_pairs if allowed_pairs else [])
            ] if allowed_pairs else None
        },
        'num_examples': len(scores),
        'mean_score': mean_score,
        'details': details
    }

    output_path = script_dir / config['output_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results: {mean_score:.2%} ({sum(scores)}/{len(scores)})")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()