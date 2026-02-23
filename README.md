# Evaluating Memory Structure in LLM Agents

<a href='https://arxiv.org/abs/2602.11243'><img src='https://img.shields.io/badge/ArXiv-PDF-red' height="25"></a> &nbsp; 

Supplementary code for the working paper **Asynchronous Reasoning: Training-Free Interactive Thinking LLMs**.

**🚧Work in progress!🛠️** The benchmark will be streamlined within the next week (by end of Mar.02 AOE). The initial version (v0.1) is available, but please expect that the code will change soon.


# Raw benchmark data:
- Accounting (count-based): [`./benchmark/accounting/data`](https://github.com/yandex-research/StructMemEval/tree/773b71cfc3d94958f7631fe053de81364db7ff8d/benchmark/accounting/data)
- Tree-based: [`./benchmark/tree_based/graph_configs`](https://github.com/yandex-research/StructMemEval/tree/773b71cfc3d94958f7631fe053de81364db7ff8d/benchmark/tree_based/graph_configs)
- State tracking: [`./benchmark/state_tracking/data`](https://github.com/yandex-research/StructMemEval/tree/773b71cfc3d94958f7631fe053de81364db7ff8d/benchmark/state_tracking/data)
- Recsys: [`./benchmark/recommendations/data`](https://github.com/yandex-research/StructMemEval/tree/773b71cfc3d94958f7631fe053de81364db7ff8d/benchmark/recommendations/data)


# Running evaluation:

- **Install mem-agent**

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


- **Run benchmark**
- Accounting (count-based): [`./benchmark/accounting/benchmark.py`](./benchmark/accounting/benchmark.py)
- Tree-based: [`./benchmark/tree_based/benchmark.py`](./benchmark/tree_based/benchmark.py)
- State tracking: [`./benchmark/state_tracking/benchmark.py`](./benchmark/state_tracking/benchmark.py)

Each one runs via `python benchmark.py --config config.yaml`, when you can change the model, the evaluation mode (retrieval / hint / no hint) and others.

A more streamlined runner will be released soon.
