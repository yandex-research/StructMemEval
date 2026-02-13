#!/usr/bin/env python3
"""Test judge behavior on specific cases."""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

cases = [
    {
        "name": "transition_001 (should match - both mention Battery Park)",
        "reference": "Since you're in NYC, Lower Manhattan now, how about taking a walk to Battery Park to see the Statue of Liberty?",
        "response": "You should walk to Battery Park this Saturday morning, as it's part of your regular activities!"
    },
    {
        "name": "transition_002 (should NOT match - wrong activity)",
        "reference": "Since you're in Tokyo, Shibuya now, why not enjoy a morning jog through Yoyogi Park followed by matcha at your favorite small shop nearby?",
        "response": "This Saturday morning, you can visit the flea market at Assistens Cemetery, as it's part of your regular activities."
    }
]

prompt_template = open(Path(__file__).parent / 'judge/prompt.txt').read()

for case in cases:
    prompt = prompt_template.format(
        reference_text=case["reference"],
        llm_response=case["response"]
    )

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=5,
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    print(f"{case['name']}")
    print(f"  Raw response: {repr(response.choices[0].message.content)}")
    print(f"  Judge: {answer} ({'PASS' if answer == '1' else 'FAIL'})")
    print()
