## Benchmark

python benchmark.py

Config (`config.yaml`):
```yaml
benchmark:
  cases:
    - data_path: data/debt_tracker_10percent.json
      prompt_path: prompts/system_prompt_health_big_hint.txt
```

Output: `eval_results/`

## Judge

python judge/judge.py --config judge/config.yaml

Config (`judge/config.yaml`):
- input_path: путь к JSON результатам eval
- model: модель для judge

Output: {"num_examples": 1, "mean_score": 0.5, "details": [...]}

## Добавление своих примеров

1. Создать файл данных в `data/`:
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

2. Создать промпт в `prompts/` (можно скопировать существующий и модифицировать)

3. Добавить case в `config.yaml`:
```yaml
benchmark:
  cases:
    - data_path: data/my_case.json
      prompt_path: prompts/my_prompt.txt
```

4. Для отладки — закомментировать остальные cases:
```yaml
benchmark:
  cases:
    # - data_path: data/health_tracking.json
    #   prompt_path: prompts/system_prompt_health_huge_hint.txt
    - data_path: data/my_case.json
      prompt_path: prompts/my_prompt.txt
```

5. Запустить:
```bash
python benchmark.py
python judge/judge.py --config judge/config.yaml
```
