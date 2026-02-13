#!/usr/bin/env python3
"""Simple test of judge behavior."""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

prompt = """Does the word "Battery Park" appear in BOTH of these texts?

Text A: "Since you're in NYC, Lower Manhattan now, how about taking a walk to Battery Park to see the Statue of Liberty?"

Text B: "You should walk to Battery Park this Saturday morning, as it's part of your regular activities!"

Answer "1" if yes, "0" if no."""

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': prompt}],
    max_tokens=5,
    temperature=0
)
print('Answer:', response.choices[0].message.content)
