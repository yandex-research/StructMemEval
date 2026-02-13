#!/usr/bin/env python3
"""
Unified Benchmark: Location Context

Tests mem0 (RAG with different top-K) and mem-agent (with/without hints)
on location-based memory tasks.

Outputs separate JSON files for each configuration for LLM judge evaluation.
"""

import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from httpx import Client
from openai._base_client import DEFAULT_TIMEOUT, DEFAULT_CONNECTION_LIMITS

import yaml
from tqdm import tqdm
from openai import OpenAI

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# mem0 imports
from mem0.memory.main import Memory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

# mem-agent
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "mem-agent"))
os.environ["PYTHONPATH"] = str(project_root / "mem-agent")

from agent.agent import Agent


# ============================================================================
# Configuration & Data Loading
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config with environment variable substitution"""
    with open(config_path, 'r') as f:
        config_str = f.read()
    config_str = os.path.expandvars(config_str)
    return yaml.safe_load(config_str)


def load_benchmark_data(data_path: str) -> dict:
    """Load benchmark data JSON (single case)"""
    with open(data_path, 'r') as f:
        return json.load(f)


# ============================================================================
# Memory Initialization
# ============================================================================

def initialize_mem0(config: dict) -> Memory:
    """Initialize mem0 Memory instance"""
    memory = Memory(
        MemoryConfig(
            llm=LlmConfig(
                provider=config['llm']['provider'],
                config={
                    "model": config['llm']['model'],
                    "api_key": config['llm']['api_key'],
                },
            ),
            embedder=EmbedderConfig(
                provider=config['embedder']['provider'],
                config={
                    "model": config['embedder']['model'],
                    "api_key": config['embedder']['api_key'],
                },
            ),
            vector_db=VectorStoreConfig(
                provider=config['vector_db']['provider'],
                config={
                    "collection_name": config['vector_db']['collection_name'],
                    "path": config['vector_db']['path'],
                    "embedding_model_dims": config['vector_db']['embedding_model_dims'],
                },
            ),
        )
    )
    memory.reset()
    return memory


def initialize_mem_agent(config: dict, prompt_path: str, memory_path: str) -> Agent:
    """Initialize Agent instance with specific prompt and memory path"""
    os.environ['OPENAI_API_KEY'] = config['api_key']
    path = Path(memory_path)
    if path.exists():
        shutil.rmtree(path)
    agent = Agent(
        model=config['model'],
        memory_path=memory_path,
        use_vllm=False,
        system_prompt_path=prompt_path,
    )
    agent._client._client = Client(
        base_url=agent._client._client.base_url, verify=False,
        timeout=DEFAULT_TIMEOUT, limits=DEFAULT_CONNECTION_LIMITS, follow_redirects=True,
    )
    return agent


# ============================================================================
# Session Loading
# ============================================================================

def load_user_messages_to_mem0(memory: Memory, sessions: list, config: dict):
    """Load user messages into mem0"""
    user_messages = []
    for session in sessions:
        for msg in session['messages']:
            if msg['role'] == 'user':
                user_messages.append({'role': 'user', 'content': msg['content']})

    user_id = config['benchmark']['user_id']
    infer = config['benchmark']['infer']

    print(f"\nLoading {len(user_messages)} user messages into mem0...")
    for msg in tqdm(user_messages, desc="mem0 loading"):
        memory.add([msg], user_id=user_id, infer=infer)

    print(f"✓ Loaded {len(user_messages)} messages")


def load_user_messages_to_agent(agent: Agent, sessions: list, verbose: bool = False):
    """Load user messages into mem-agent"""
    user_messages = []
    for session in sessions:
        for msg in session['messages']:
            if msg['role'] == 'user':
                user_messages.append(msg['content'])

    print(f"\nLoading {len(user_messages)} user messages into mem-agent...")
    for content in tqdm(user_messages, desc="mem-agent loading"):
        if verbose:
            print(f"USER: {content}")
        reply = agent.chat(content)
        if verbose:
            print(f"AGENT: {reply}\n")

        # Reset conversation history after each message to avoid accumulation
        agent.messages = agent.messages[:1]

    print(f"✓ Loaded {len(user_messages)} messages")


# ============================================================================
# Query Execution - mem0
# ============================================================================

def run_mem0_query(memory: Memory, query_obj: dict, config: dict, limit: int) -> dict:
    """Run mem0 query with specific retrieve limit"""
    question = query_obj['question']
    user_id = config['benchmark']['user_id']

    # Search with specified limit
    response = memory.search(question, user_id=user_id, limit=limit)
    results = response.get('results', [])

    # Get retrieved memories
    retrieved_memories = [r['memory'] for r in results]

    # Build system prompt
    memory_context = "\n".join(f"- {mem}" for mem in retrieved_memories) if retrieved_memories else "No relevant memories."
    system_prompt = f"""You are a helpful assistant.

Use this context about the user when answering:
{memory_context}

Answer concisely and take the user's preferences into account."""

    # Get LLM response
    client = OpenAI(api_key=config['llm']['api_key'])
    llm_response = client.chat.completions.create(
        model=config['llm']['model'],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    answer = llm_response.choices[0].message.content

    # Build result
    result = {
        "query": question,
        "llm_response": answer,
        "memory_state": {
            "retrieved_memories": [
                {"score": r.get('score', 0), "text": r['memory']}
                for r in results
            ],
            "total_memories": len(retrieved_memories)
        },
        "reference_answer": query_obj['reference_answer'],
        "metadata": {
            "system_prompt": system_prompt,
            "retrieved_count": len(results),
            "retrieve_limit": limit
        }
    }

    return result


# ============================================================================
# Query Execution - mem-agent
# ============================================================================

def get_memory_files(memory_path: str) -> list[str]:
    """Get list of memory files created by agent"""
    memory_dir = Path(memory_path)
    if not memory_dir.exists():
        return []

    files = []
    if (memory_dir / "user.md").exists():
        files.append(str(memory_dir / "user.md"))

    entities_dir = memory_dir / "entities"
    if entities_dir.exists():
        for entity_file in sorted(entities_dir.glob("*.md")):
            files.append(str(entity_file))

    return files


def read_memory_content(memory_path: str) -> dict:
    """Read memory file contents"""
    files = get_memory_files(memory_path)
    content = {}

    for file_path in files:
        rel_path = str(Path(file_path).relative_to(Path(memory_path).parent))
        with open(file_path, 'r') as f:
            content[rel_path] = f.read()

    return content


def run_mem_agent_query(agent: Agent, query_obj: dict, config: dict) -> dict:
    """Run mem-agent query and return result dict"""
    question = query_obj['question']

    # Reset agent conversation to only system prompt before query
    agent.messages = agent.messages[:1]

    # Get response
    response = agent.chat(question)

    # Get memory state
    memory_path = config['memory_path']
    memory_files = get_memory_files(memory_path)
    memory_content = read_memory_content(memory_path)

    # Build result
    result = {
        "query": question,
        "llm_response": response.reply,
        "memory_state": {
            "memory_files": memory_files,
            "memory_content": memory_content
        },
        "reference_answer": query_obj['reference_answer'],
        "metadata": {
            "agent_thoughts": response.thoughts,
            "python_block": response.python_block
        }
    }

    return result


# ============================================================================
# Output Generation
# ============================================================================

def save_results(data: dict, output_path: str):
    """Save results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# Case Processing
# ============================================================================

def process_case_mem0(mem0: Memory, case_data: dict, config: dict, limits: list) -> dict:
    """Process a single case with mem0 at different limits. Returns {limit: results}"""
    results_by_limit = {}

    for limit in limits:
        print(f"    mem0 top-{limit}...")
        case_results = []
        for query_obj in case_data['queries']:
            result = run_mem0_query(mem0, query_obj, config, limit)
            case_results.append(result)
        results_by_limit[limit] = case_results

    return results_by_limit


def process_case_agent(agent: Agent, case_data: dict, config: dict, memory_path: str) -> list:
    """Process a single case with mem-agent"""
    agent_config = {**config, 'memory_path': memory_path}
    case_results = []

    for query_obj in case_data['queries']:
        result = run_mem_agent_query(agent, query_obj, agent_config)
        case_results.append(result)

    return case_results


# ============================================================================
# Main
# ============================================================================

def run_benchmark(script_dir: Path, config: dict, data_path: str, prompt_configs: list):
    """Run benchmark for a single data file (one case per file)."""
    case_data = load_benchmark_data(str(script_dir / data_path))
    case_id = case_data.get('case_id', Path(data_path).stem)
    mem0_limits = config['benchmark'].get('mem0_limits', [10])

    print(f"\n{'='*80}")
    print(f"CASE: {case_id} ({data_path})")
    print(f"  mem0 limits: {mem0_limits}")
    print(f"  Agent prompts: {[p['name'] for p in prompt_configs]}")
    print("="*80)

    # Results storage: {config_name: case_result}
    all_results = {}

    # Initialize mem0 (fresh for each case)
    mem0 = initialize_mem0(config['mem0'])
    load_user_messages_to_mem0(mem0, case_data['sessions'], config['mem0'])

    # Run mem0 queries at all limits
    mem0_results = process_case_mem0(mem0, case_data, config['mem0'], mem0_limits)
    for limit, results in mem0_results.items():
        all_results[f"mem0_top{limit}"] = {
            "case_id": case_id,
            "results": results
        }

    # Run agent with each prompt config
    for prompt_cfg in prompt_configs:
        print(f"  {prompt_cfg['name']}...")
        memory_path = f"{config['mem_agent']['memory_path']}/{case_id}_{prompt_cfg['name']}"
        agent = initialize_mem_agent(
            config['mem_agent'],
            str(script_dir / prompt_cfg['path']),
            memory_path
        )
        load_user_messages_to_agent(agent, case_data['sessions'], config['benchmark']['verbose'])
        agent_results = process_case_agent(agent, case_data, config['mem_agent'], memory_path)
        all_results[prompt_cfg['name']] = {
            "case_id": case_id,
            "prompt_path": prompt_cfg['path'],
            "results": agent_results
        }

    return all_results


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    config = load_config(str(config_path))

    # Get configurations
    data_paths = config['benchmark'].get('data_paths', [])
    prompt_configs = config['benchmark'].get('agent_prompts', [
        {"name": "mem_agent", "path": "prompts/system_prompt.txt"}
    ])

    if not data_paths:
        # Fallback to old format
        data_paths = [c['data_path'] for c in config['benchmark'].get('cases', [])]

    print(f"Running benchmark on {len(data_paths)} data file(s)")

    output_dir = script_dir / config['benchmark']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    # Run benchmarks
    for data_path in data_paths:
        all_results = run_benchmark(script_dir, config, data_path, prompt_configs)

        # Determine group name from path
        group_name = Path(data_path).stem

        # Save results for each configuration
        print(f"\n{'='*80}")
        print(f"SAVING RESULTS for {group_name}")
        print("="*80)

        for config_name, case_result in all_results.items():
            output = {
                "benchmark_timestamp": timestamp,
                "data_path": data_path,
                "memory_type": config_name,
                "config": {
                    "model": config['mem0']['llm']['model'] if config_name.startswith('mem0') else config['mem_agent']['model'],
                },
                "cases": [case_result]  # wrap single case in list for consistency
            }

            output_path = output_dir / f"results_{group_name}_{config_name}.json"
            save_results(output, str(output_path))
            print(f"✓ Saved {output_path}")

    print("\n" + "="*80)
    print("✓ BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
