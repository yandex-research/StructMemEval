#!/usr/bin/env python3
"""
Unified Benchmark: Indirect State Interactions

Tests both mem0 (RAG) and mem-agent (agentic) memory systems
on health symptoms embedded in casual conversations.

Outputs standardized JSON for LLM judge evaluation.
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

# mem0 imports
from mem0.memory.main import Memory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

# mem-ganet
project_root = Path.cwd().parent.parent
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
    
    load_dotenv()

    config_str = os.path.expandvars(config_str)
    return yaml.safe_load(config_str)


def load_benchmark_data(data_path: str) -> dict:
    """Load enhanced benchmark data JSON"""
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
                    "openrouter_base_url": config['llm']['openrouter_base_url']
                },
            ),
            embedder=EmbedderConfig(
                provider=config['embedder']['provider'],
                config={
                    "model": config['embedder']['model'],
                    "api_key": config['embedder']['api_key'],
                    "openai_base_url": config['embedder']['openai_base_url']
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
    http_client = Client(
        verify=False,
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True
    )

    memory.llm.client = OpenAI(
        api_key=config['llm']['api_key'],
        base_url=config['llm']['openrouter_base_url'],
        http_client=http_client
    )

    memory.embedding_model.client = OpenAI(
        api_key=config['embedder']['api_key'],
        base_url=config['embedder']['openai_base_url'],
        http_client=http_client
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
        model=config["model"],
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

def run_mem0_query(memory: Memory, query_obj: dict, config: dict) -> dict:
    """Run mem0 query and return result dict"""
    question = query_obj['question']
    user_id = config['benchmark']['user_id']
    limit = config['benchmark']['retrieve_limit']

    # Search
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
    llm_response = memory.llm.client.chat.completions.create(
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
            "retrieved_count": len(results)
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

def save_results_incremental(data: dict, output_path: str, memory_type: str):
    """Save results to JSON file incrementally, preserving existing entries."""
    output_file = Path(output_path)
    
    # Read existing data if file exists
    existing_data = {}
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = {}
    else:
        # Initialize empty structure if file doesn't exist
        existing_data = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "memory_type": memory_type,
            "total_cases": 0,
            "config": {},
            "cases": []
        }

    # Update the config from new data if not already set
    if not existing_data.get("config"):
        existing_data["config"] = data.get("config", {})

    # Process new cases and add only unique ones
    new_cases = data.get("cases", [])
    existing_cases = existing_data.get("cases", [])
    
    # Create a set of existing (case_id, prompt_path) pairs for quick lookup
    existing_keys = set()
    for case in existing_cases:
        key = (case.get("case_id"), case.get("prompt_path"))
        if key[0] is not None and key[1] is not None:  # Only add valid keys
            existing_keys.add(key)

    # Add new cases that aren't already present
    added_count = 0
    for new_case in new_cases:
        new_key = (new_case.get("case_id"), new_case.get("prompt_path"))
        
        if new_key[0] is not None and new_key[1] is not None and new_key not in existing_keys:
            existing_cases.append(new_case)
            existing_keys.add(new_key)
            added_count += 1
    
    # Update total cases count
    existing_data["total_cases"] = len(existing_cases)
    
    # Write updated data back to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated {output_file} - Added {added_count} new case(s), Total: {existing_data['total_cases']} case(s)")


# ============================================================================
# Main
# ============================================================================

def run_case(script_dir: Path, config: dict, case_config: dict, case_idx: int) -> tuple[dict, dict]:
    """Run benchmark for a single case. Returns (mem0_case_result, agent_case_result)."""
    data_path = script_dir / case_config['data_path']
    prompt_path = script_dir / case_config['prompt_path']
    prompt_file = ".".join(case_config['prompt_path'].split(".")[:-1])
    data = load_benchmark_data(str(data_path))

    case_id = data['case_id']
    print(f"\n{'='*80}")
    print(f"CASE {case_idx + 1}: {case_id}")
    print(f"  Data: {case_config['data_path']}")
    print(f"  Prompt: {case_config['prompt_path']}")
    print(f"  Sessions: {len(data['sessions'])}, Queries: {len(data['queries'])}")
    print("="*80)

    # Initialize memory systems (fresh for each case)
    mem0 = initialize_mem0(config['mem0'])
    memory_path = f"{config['mem_agent']['memory_path']}/{case_id}/{prompt_file}"
    agent = initialize_mem_agent(config['mem_agent'], str(prompt_path), memory_path)

    # Load sessions
    load_user_messages_to_mem0(mem0, data['sessions'], config['mem0'])
    load_user_messages_to_agent(agent, data['sessions'], config['benchmark']['verbose'])

    # Run queries
    mem0_results = []
    agent_results = []

    for i, query_obj in enumerate(data['queries']):
        print(f"\n  Query {i+1}/{len(data['queries'])}: {query_obj['question'][:50]}...")

        mem0_result = run_mem0_query(mem0, query_obj, config['mem0'])
        mem0_results.append(mem0_result)

        agent_config_with_path = {**config['mem_agent'], 'memory_path': memory_path}
        agent_result = run_mem_agent_query(agent, query_obj, agent_config_with_path)
        agent_results.append(agent_result)

    mem0_case = {
        "case_id": case_id,
        "prompt_path": case_config['prompt_path'],
        "results": mem0_results
    }
    agent_case = {
        "case_id": case_id,
        "prompt_path": case_config['prompt_path'],
        "results": agent_results
    }

    # Save results incrementally after each case
    output_dir = script_dir / config['benchmark']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    mem0_output_path = output_dir / "results_mem0.json"
    mem0_data = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "memory_type": "mem0",
        "config": {
            "model": config['mem0']['llm']['model'],
            "embedder": config['mem0']['embedder']['model'],
        },
        "cases": [mem0_case]
    }
    save_results_incremental(mem0_data, str(mem0_output_path), "mem0")

    agent_output_path = output_dir / "results_mem_agent.json"
    agent_data = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "memory_type": "mem_agent",
        "config": {
            "model": config['mem_agent']['model'],
        },
        "cases": [agent_case]
    }
    save_results_incremental(agent_data, str(agent_output_path), "mem_agent")

    return mem0_case, agent_case


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    config = load_config(str(config_path))
    
    cases = config['benchmark']['cases']
    print(f"Found {len(cases)} case(s) to run")

    all_mem0_cases = []
    all_agent_cases = []

    for idx, case_config in enumerate(cases):
        mem0_case, agent_case = run_case(script_dir, config, case_config, idx)
        all_mem0_cases.append(mem0_case)
        all_agent_cases.append(agent_case)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
