## Benchmark

python benchmark.py

Config (`config.yaml`):
```yaml
benchmark:
  cases:
    - data_path: data/health_tracking.json
      prompt_path: prompts/system_prompt_health_huge_hint.txt
```

Output: `eval_results/results_mem_agent.json`, `eval_results/results_mem0.json`

## Judge

python judge/judge.py --config judge/config.yaml

Config (`judge/config.yaml`):
- input_path: path to eval results from the previous step
- model: the LLM to use as a judge

Output: {"num_examples": ..., "mean_score": ..., "details": [...]}

## Adding / editing examples

1. Create / edit files in `data/`:
```json
{
  "case_id": "my_case",
  "sessions": [
    {"session_id": "session_01", "messages": [{"role": "user", "content": "..."}, ...]}
  ],
  "queries": [
    {"question": "...", "reference_answer": {"text": "..."}}
  ]
}
```

2. Create a prompt (or hint) in `prompts/` (we recommend looking at existing prompts first)

3. Add `config.yaml`:
```yaml
benchmark:
  cases:
    - data_path: data/my_case.json
      prompt_path: prompts/my_prompt.txt
```

4. For debugging, you can comment-out other cases like this:
```yaml
benchmark:
  cases:
    # - data_path: data/health_tracking.json
    #   prompt_path: prompts/system_prompt_health_huge_hint.txt
    - data_path: data/my_case.json
      prompt_path: prompts/my_prompt.txt
```

5. Run benchmark & judge as usual:
```bash
python benchmark.py
python judge/judge.py --config judge/config.yaml
```
