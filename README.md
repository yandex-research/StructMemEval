# Evaluating Memory Structure in LLM Agents

<a href='https://arxiv.org/abs/2602.11243'><img src='https://img.shields.io/badge/ArXiv-PDF-red' height="25"></a> &nbsp;

Supplementary code for the working paper **Asynchronous Reasoning: Training-Free Interactive Thinking LLMs**.

# Raw benchmark data:
- Accounting (count-based): [`./benchmark/accounting/data`](./benchmark/accounting/data)
- Tree-based: [`./benchmark/tree_based/graph_configs`](./benchmark/tree_based/graph_configs)
- State tracking: [`./benchmark/data/state_machine_location`](./benchmark/data/state_machine_location)
- Recsys: [`./benchmark/recommendations/data`](./benchmark/recommendations/data)

# Running evaluation:

## Install mem-agent

We're using a slightly modified [mem-agent codebase](https://github.com/firstbatchxyz/mem-agent). Here's how to install it:
```bash
cd mem-agent
# 1. install dependeincies
make check-uv
make install
.venv/bin/python -m ensurepip --default-pip

# 2. set up API keys and endpoints
cp .env.example .env
nano .env # !!! ACTION REQUIRED: !!! manually edit the copied .env to use your API keys there. Optionally change base urls if needed.

# 3. (optional)for jupyter exps
pip install ipykernel
python -m ipykernel install --user --name=mem-agent --display-name="Python (mem-agent)

cd ..
cp mem-agent/.env .env
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...          # OpenAI (mem0 embedder + default LLM)

# For non-OpenAI models via OpenRouter or other proxy
LLM_PROVIDER_API_KEY=...       # API key for the LLM provider
LLM_PROVIDER_BASE_URL=...      # OpenAI-compatible base URL
```

## Run benchmark

```bash
# Full benchmark
python benchmark.py --config config.yaml

# Quick test (6 cases)
python benchmark.py --config config_test.yaml

# Clean memory before run
python benchmark.py --config config_test.yaml --clean-memory

# Judge evaluation
cd judge && python run_all_judge.py
```

### Memory Systems

| System | Description |
|--------|-------------|
| **mem0 RAG** (top-k) | Vector DB retrieval with configurable top-k |
| **mem0 Agent** | Tool-calling agent with add/search/update/delete memory operations |
| **mem-agent** | Structured markdown file-based memory (user.md + entities/) |

### Config Structure

```yaml
# Memory system definitions (infrastructure + defaults)
mem0:
  llm: { provider: openai, model: gpt-4o-mini, api_key: ${OPENAI_API_KEY} }
  embedder: { provider: openai, model: text-embedding-3-large, ... }
  vector_db: { provider: qdrant, path: ./qdrant_data, ... }

mem_agent:
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  memory_path: memory_mem_agent

# Experiments (which systems to run, with what model, on what data)
experiments:
  - name: gpt-4o-mini
    data_dirs: [benchmark/data/state_machine_location/]
    run:
      mem0_rag: { limits: [5, 20], infer: [false] }
      mem0_agent: { iterations: 5, search_limit: 50 }
      mem_agent: {}

  - name: gemini-2.5-pro           # override model for this experiment
    model: gemini-2.5-pro
    api_key: ${GEMINI_API_KEY}
    base_url: https://generativelanguage.googleapis.com/v1beta/openai/
    data_dirs: [benchmark/data/state_machine_location/]
    run:
      mem0_rag: { limits: [5, 20], infer: [false] }
      mem0_agent: { iterations: 5, search_limit: 50 }
      mem_agent: {}

# Execution settings
max_cases: null
parallel_workers: 2
```

If an experiment defines `model`/`api_key`/`base_url`, they override the defaults from `mem0.llm`. If not, defaults are used. Embedder is never overridden (always OpenAI).

---

## How to Add a New Experiment (Model/Provider)

Add a new entry to the `experiments` list in `config.yaml`:

```yaml
experiments:
  - name: gpt-4o-mini          # uses defaults from mem0.llm
    data_dirs: [benchmark/data/state_machine_location/]
    run:
      mem0_rag: { limits: [5, 20], infer: [false] }
      mem0_agent: { iterations: 5, search_limit: 50 }
      mem_agent: {}

  - name: my-new-model          # override model/key/url
    model: my-model-name
    api_key: ${MY_API_KEY}
    base_url: https://my-provider.com/v1/
    data_dirs: [benchmark/data/state_machine_location/]
    run:
      mem0_rag: { limits: [5, 20], infer: [false] }
      mem0_agent: { iterations: 5, search_limit: 50 }
      mem_agent: {}
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Experiment name (used in output dir and memory isolation) |
| `model` | no | LLM model name (default: `mem0.llm.model`) |
| `api_key` | no | API key (default: `mem0.llm.api_key`) |
| `base_url` | no | OpenAI-compatible base URL (default: `mem0.llm.base_url`) |
| `data_dirs` | yes | List of dataset directories to run |
| `run` | yes | Which memory systems to run and their runtime params |

**Runtime params in `run`:**

| System | Params |
|--------|--------|
| `mem0_rag` | `limits` (list of top-k values), `infer` (list of bool) |
| `mem0_agent` | `iterations` (max tool-call rounds), `search_limit` (max memories) |
| `mem_agent` | `{}` (no extra params) |

Omit a system from `run` to skip it for that experiment.

**Memory isolation** is automatic: Qdrant collection becomes `{collection_name}_{experiment_name}`, mem-agent path becomes `{memory_path}_{experiment_name}/`.

**Output** goes to `{output_dir}/{experiment_name}/`.

---

## How to Add a New Dataset (Scenario)

1. **Create a data directory:**

```
benchmark/data/my_new_scenario/
  dataset.yaml
  prompts/
    mem0_agent_loading.txt
    mem0_agent_query.txt
    system_prompt.txt
  case_001.json
  case_002.json
  ...
```

2. **Write `dataset.yaml`:**

```yaml
collection_name: benchmark_my_scenario   # Qdrant collection prefix
user_id: benchmark_user_my_scenario      # mem0 user ID

prompts:
  mem0_agent:
    loading: prompts/mem0_agent_loading.txt   # system prompt for loading phase
    query: prompts/mem0_agent_query.txt       # system prompt for query phase
  mem_agent:
    system_prompt: prompts/system_prompt.txt  # mem-agent system prompt
```

All paths are relative to the dataset directory. Only include prompt sections for memory systems you want to run. If `mem0_agent` prompts are missing, that phase is skipped.

3. **Create JSON case files** (one per test case):

```json
{
  "case_id": "case_001",
  "sessions": [
    {
      "session_id": "s1",
      "messages": [
        {"role": "user", "content": "I just moved to Berlin."},
        {"role": "assistant", "content": "That's exciting! How do you like it?"}
      ]
    }
  ],
  "queries": [
    {
      "question": "Where does the user currently live?",
      "reference_answer": {"text": "Berlin"}
    }
  ]
}
```

4. **Add to config.yaml:**

```yaml
experiments:
  - name: gpt-4o-mini
    data_dirs:
      - benchmark/data/state_machine_location/
      - benchmark/data/my_new_scenario/          # <-- add here
    run:
      mem0_rag: { limits: [5, 20], infer: [false] }
      mem0_agent: { iterations: 5, search_limit: 50 }
      mem_agent: {}
```
