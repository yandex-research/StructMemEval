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

5. Запустить оценку

1. **Убедитесь, что judge-конфиг ссылается на нужный файл с результатами.**  
   В `judge/config.yaml` должно быть указано:
   ```yaml
   input_path: eval_results/results_mem_agent.json   # или results_mem0.json
   prompt_path: judge/judge_prompt.txt               # шаблон для judge-модели
   output_path: judge/results/my_case_judge.json
   model: gpt-4o-mini
   api_key: ${OPENAI_API_KEY}
   
   # Опционально: фильтрация по конкретной паре (case_id, prompt_path)
   filter_case_id:
     - my_case
   filter_prompt_path:
     - prompts/my_prompt.txt
   ```

2. **Запустите judge:**
   ```bash
   python judge/judge.py --config judge/config.yaml
   ```

3. **Результаты сохранятся в указанный `output_path`.**  
   Файл будет содержать:
   - Общее количество оценённых примеров
   - Средний бинарный скор (`mean_score`)
   - Детали по каждому запросу: вопрос, ответ модели, эталон, оценка и метаданные

> 💡 Если вы не используете фильтрацию в конфиге, judge оценит **все примеры** из `input_path`. Чтобы оценить только свой кейс — либо добавьте `filter_case_id`/`filter_prompt_path`, либо временно закомментируйте другие кейсы в `benchmark/cases` перед запуском `benchmark.py`.

