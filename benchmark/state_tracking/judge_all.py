#!/usr/bin/env python3
"""Run judge on all state_machine_location results."""

import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / '.env')

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

prompt_template = open(Path(__file__).parent / 'judge/prompt.txt').read()

# Get all state_machine_location results - only new benchmark files
results_files = []
for f in os.listdir(Path(__file__).parent / 'eval_results'):
    if (f.startswith('results_static_') or f.startswith('results_transition_')) and \
       ('top10' not in f and 'top15' not in f):
        results_files.append(f)
results_files = sorted(results_files)

print(f'Found {len(results_files)} result files to judge')

summary = {}

for result_file in tqdm(results_files, desc='Judging files'):
    input_path = Path(__file__).parent / f'eval_results/{result_file}'
    output_path = Path(__file__).parent / f'judge/results/judge_{result_file.replace("results_", "")}'

    with open(input_path) as f:
        eval_data = json.load(f)

    all_results = []
    for case in eval_data['cases']:
        for result in case['results']:
            all_results.append({'case_id': case['case_id'], **result})

    scores = []
    details = []

    for result in all_results:
        reference = result['reference_answer']
        prompt = prompt_template.format(
            reference_text=reference['text'],
            llm_response=result['llm_response']
        )

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=5,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        score = 1 if answer == '1' else 0
        scores.append(score)
        details.append({
            'case_id': result.get('case_id', ''),
            'score': score
        })

    mean_score = sum(scores) / len(scores) if scores else 0

    output = {
        'input_file': str(input_path),
        'num_examples': len(scores),
        'mean_score': mean_score,
        'details': details
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    summary[result_file] = mean_score

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
for k, v in sorted(summary.items()):
    print(f'{k}: {v:.0%}')
