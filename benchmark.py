#!/usr/bin/env python3
"""
StructMemEval Benchmark Runner

Tests memory systems — mem0 RAG (top-K retrieval), mem0 Agent (tool-calling),
and mem-agent (structured markdown) — on long-term memory tasks.

Outputs separate JSON files per case/config for LLM judge evaluation.
"""

import json
import os
import sys
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from qdrant_client import QdrantClient

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

from dotenv import load_dotenv
from httpx import Client
from openai._base_client import DEFAULT_TIMEOUT, DEFAULT_CONNECTION_LIMITS

import yaml
from tqdm import tqdm
from openai import OpenAI

from config_loader import load_config, resolve_dataset, resolve_experiments, Experiment, Dataset

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

# mem0 auto-detects OPENROUTER_API_KEY and switches provider — prevent this
os.environ.pop('OPENROUTER_API_KEY', None)

# mem0 imports
from mem0.memory.main import Memory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

# mem-agent
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "mem-agent"))
os.environ["PYTHONPATH"] = str(project_root / "mem-agent")

from agent.agent import Agent


# ============================================================================
# LLM Client Helpers
# ============================================================================

def create_llm_client(llm_config: dict) -> OpenAI:
    """Create OpenAI-compatible client from LLM config dict.

    Args:
        llm_config: Dict with 'api_key', optional 'base_url'
    """
    kwargs = {'api_key': llm_config['api_key'], 'max_retries': 5}
    if llm_config.get('base_url'):
        kwargs['base_url'] = llm_config['base_url']
        # Disable SSL verification and use longer timeout for custom/proxy endpoints
        kwargs['http_client'] = Client(
            verify=False, timeout=120.0,
            limits=DEFAULT_CONNECTION_LIMITS, follow_redirects=True,
        )
    return OpenAI(**kwargs)


def create_client_from_experiment(experiment: Experiment) -> OpenAI:
    """Create OpenAI client from Experiment dataclass."""
    cfg = {'api_key': experiment.api_key}
    if experiment.base_url:
        cfg['base_url'] = experiment.base_url
    return create_llm_client(cfg)


def normalize_tool_calls(message):
    """Normalize Gemini quirks in tool call responses.

    Gemini via OpenAI compat layer may return empty tool_call.id
    or arguments as dict instead of JSON string.
    """
    if not message.tool_calls:
        return message
    for tc in message.tool_calls:
        if isinstance(tc.function.arguments, dict):
            tc.function.arguments = json.dumps(tc.function.arguments)
        if not tc.id:
            tc.id = f"call_{uuid.uuid4().hex[:8]}"
    return message


# ============================================================================
# Memory Cleanup
# ============================================================================

def clean_memory(config: dict, script_dir: Path, experiments: list[Experiment]):
    """Clean Qdrant collections and mem-agent directories before benchmark run."""
    print("\n" + "="*60)
    print("MEMORY CLEANUP")
    print("="*60)

    # 1. Clean Qdrant
    qdrant_path_str = config['mem0']['vector_db'].get('path', './qdrant_data')
    qdrant_path = script_dir / qdrant_path_str

    if qdrant_path.exists():
        client = None
        try:
            client = QdrantClient(path=str(qdrant_path))
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_names:
                print(f"  Deleting {len(collection_names)} Qdrant collection(s):")
                for name in collection_names:
                    try:
                        client.delete_collection(name)
                        print(f"    ✓ {name}")
                    except Exception as e:
                        print(f"    ✗ Failed to delete {name}: {e}")
            else:
                print(f"  No Qdrant collections found")
        except Exception as e:
            print(f"  ✗ Error cleaning Qdrant: {e}")
            print(f"  Fallback: deleting {qdrant_path}")
            shutil.rmtree(qdrant_path, ignore_errors=True)
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass
    else:
        print(f"  Qdrant path {qdrant_path} does not exist, skipping")

    # 2. Clean mem-agent memory directories
    base_memory_path_str = config['mem_agent'].get('memory_path', 'memory_mem_agent')
    paths_to_clean = [script_dir / base_memory_path_str]
    for exp in experiments:
        paths_to_clean.append(script_dir / f"{base_memory_path_str}_{exp.name}")

    for mem_path in paths_to_clean:
        if mem_path.exists():
            shutil.rmtree(mem_path, ignore_errors=True)
            print(f"  ✓ Deleted mem-agent directory: {mem_path}")

    print("✓ Memory cleanup complete\n")


# ============================================================================
# Data Loading
# ============================================================================

def load_benchmark_data(data_path: str) -> dict:
    """Load benchmark data JSON (single case)"""
    with open(data_path, 'r') as f:
        return json.load(f)


# ============================================================================
# Memory Initialization
# ============================================================================

def initialize_mem0(mem0_config: dict, experiment: Experiment,
                    collection_name: str) -> Memory:
    """Initialize mem0 Memory instance.

    Args:
        mem0_config: config['mem0'] dict (embedder, vector_db sections)
        experiment: Experiment with resolved model/api_key/base_url for the LLM
        collection_name: Full collection name (already includes experiment suffix)
    """
    os.environ.pop('OPENROUTER_API_KEY', None)

    llm_config = {
        "model": experiment.model,
        "api_key": experiment.api_key,
    }
    if experiment.base_url:
        llm_config["openai_base_url"] = experiment.base_url

    print(llm_config)
    memory = Memory(
        MemoryConfig(
            llm=LlmConfig(
                provider=mem0_config['llm']['provider'],
                config=llm_config,
            ),
            embedder=EmbedderConfig(
                provider=mem0_config['embedder']['provider'],
                config={
                    "model": mem0_config['embedder']['model'],
                    "api_key": mem0_config['embedder']['api_key'],
                    "embedding_dims": mem0_config['embedder'].get('embedding_dims', 3072),
                    "openai_base_url": mem0_config['embedder']['openai_base_url']
                },
            ),
            vector_store=VectorStoreConfig(
                provider=mem0_config['vector_db']['provider'],
                config={
                    "collection_name": collection_name,
                    "path": mem0_config['vector_db']['path'],
                    "embedding_model_dims": mem0_config['vector_db']['embedding_model_dims'],
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
        api_key=mem0_config['llm']['api_key'],
        base_url=mem0_config['llm']['openrouter_base_url'],
        http_client=http_client
    )

    memory.embedding_model.client = OpenAI(
        api_key=mem0_config['embedder']['api_key'],
        base_url=mem0_config['embedder']['openai_base_url'],
        http_client=http_client
    )
    memory.reset()
    return memory


def initialize_mem_agent(experiment: Experiment, prompt_path: str,
                         memory_path: str) -> Agent:
    """Initialize Agent instance.

    Args:
        experiment: Experiment with resolved model/api_key/base_url
        prompt_path: Absolute path to system prompt file
        memory_path: Path for agent memory storage
    """
    os.environ['OPENAI_API_KEY'] = experiment.api_key
    path = Path(memory_path)
    if path.exists():
        shutil.rmtree(path)
    agent = Agent(
        model=experiment.model,
        memory_path=memory_path,
        use_vllm=False,
        system_prompt_path=prompt_path,
        api_key=experiment.api_key,
        base_url=experiment.base_url,
    )
    agent._client._client = Client(
        base_url=agent._client._client.base_url, verify=False,
        timeout=120.0, limits=DEFAULT_CONNECTION_LIMITS, follow_redirects=True,
    )
    return agent


# ============================================================================
# Session Loading
# ============================================================================

def load_user_messages_to_mem0(memory: Memory, sessions: list, user_id: str,
                                start: int = 0, end: int = -1, infer: bool = False):
    """Load user messages into mem0.

    Args:
        memory: mem0 Memory instance
        sessions: List of conversation sessions
        user_id: User ID from dataset.yaml
        infer: Whether to use LLM inference for fact extraction
    """
    user_messages = []
    for session in sessions:
        for msg in session['messages'][start: end]:
            if msg['role'] == 'user':
                user_messages.append({'role': 'user', 'content': msg['content']})

    infer_label = "infer" if infer else "raw"
    print(f"\nLoading {len(user_messages)} user messages into mem0 ({infer_label})...")
    for msg in tqdm(user_messages, desc=f"mem0 {infer_label}"):
        memory.add([msg], user_id=user_id, infer=infer)

    print(f"✓ Loaded {len(user_messages)} messages")


def load_user_messages_to_agent(agent: Agent, sessions: list, start: int = 0, end: int = -1, verbose: bool = False):
    """Load user messages into mem-agent"""
    user_messages = []
    for session in sessions:
        for msg in session['messages'][start: end]:
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

def run_mem0_query(memory: Memory, query_obj: dict, user_id: str, limit: int,
                   experiment: Experiment, answer_idx: int) -> dict:
    """Run mem0 query with specific retrieve limit."""
    question = query_obj['question']

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
    client = create_client_from_experiment(experiment)
    llm_response = client.chat.completions.create(
        model=experiment.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    answer = llm_response.choices[0].message.content
    if isinstance(query_obj['reference_answer'], list):
        ref_answer = query_obj['reference_answer'][answer_idx]
    else:
        ref_answer = query_obj['reference_answer']

    return {
        "query": question,
        "llm_response": answer,
        "memory_state": {
            "retrieved_memories": [
                {"score": r.get('score', 0), "text": r['memory']}
                for r in results
            ],
            "total_memories": len(retrieved_memories)
        },
        "reference_answer": ref_answer,
        "metadata": {
            "system_prompt": system_prompt,
            "retrieved_count": len(results),
            "retrieve_limit": limit
        },
        "message_checkpoint": answer_idx
    }


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

    # Scan cities/ directory
    cities_dir = memory_dir / "cities"
    if cities_dir.exists():
        for city_file in sorted(cities_dir.glob("*.md")):
            files.append(str(city_file))

    # Scan entities/ directory
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


def run_mem_agent_query(agent: Agent, query_obj: dict, memory_path: str, answer_idx: int) -> dict:
    """Run mem-agent query and return result dict."""
    question = query_obj['question']

    # Reset agent conversation to only system prompt before query
    agent.messages = agent.messages[:1]

    # Get response
    response = agent.chat(question)

    # Get memory state
    memory_files = get_memory_files(memory_path)
    memory_content = read_memory_content(memory_path)
    if isinstance(query_obj['reference_answer'], list):
        ref_answer = query_obj['reference_answer'][answer_idx]
    else:
        ref_answer = query_obj['reference_answer']

    return {
        "query": question,
        "llm_response": response.reply,
        "memory_state": {
            "memory_files": memory_files,
            "memory_content": memory_content
        },
        "reference_answer": ref_answer,
        "metadata": {
            "agent_thoughts": response.thoughts,
            "python_block": response.python_block
        },
        "message_checkpoint": answer_idx
    }


# ============================================================================
# Query Execution - mem0 agent (tool-calling)
# ============================================================================

MEM0_AGENT_TOOLS_ALL = [
    {
        "type": "function",
        "function": {
            "name": "add_memory",
            "description": "Save a fact about the user to memory. Store concise, self-contained facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The fact to remember about the user"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": "Search user memories by query. Returns results with IDs (use IDs for update/delete).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Update an existing memory by its ID. Use after search_memories to get the ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory to update"},
                    "text": {"type": "string", "description": "New text for this memory"}
                },
                "required": ["memory_id", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "Delete an existing memory by its ID. Use after search_memories to get the ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory to delete"}
                },
                "required": ["memory_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_memories",
            "description": "List all stored memories about the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results (default 50)", "default": 50}
                }
            }
        }
    }
]

# Subsets for different phases
MEM0_AGENT_TOOLS_LOADING = [t for t in MEM0_AGENT_TOOLS_ALL
                             if t["function"]["name"] in ("add_memory", "search_memories", "update_memory", "delete_memory")]
MEM0_AGENT_TOOLS_QUERY = [t for t in MEM0_AGENT_TOOLS_ALL
                           if t["function"]["name"] in ("search_memories", "get_all_memories")]

# Default values
DEFAULT_AGENT_ITERATIONS = 5
DEFAULT_AGENT_SEARCH_LIMIT = 50


def message_to_dict(message):
    """Convert OpenAI ChatCompletionMessage to dict for messages list."""
    d = {"role": message.role, "content": message.content}
    if message.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return d


def execute_mem0_tool_call(memory: Memory, tool_call, user_id: str,
                            default_search_limit: int = DEFAULT_AGENT_SEARCH_LIMIT) -> str:
    """Execute a single mem0 tool call and return result string."""
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if func_name == "add_memory":
        text = args["text"]
        memory.add([{"role": "user", "content": text}], user_id=user_id, infer=False)
        return f"Saved: {text}"

    elif func_name == "search_memories":
        query = args["query"]
        limit = args.get("limit", 5)
        response = memory.search(query, user_id=user_id, limit=limit)
        results = response.get("results", [])
        if results:
            lines = [f"- [id={r['id']}] {r['memory']}" for r in results]
            return "\n".join(lines)
        return "No memories found."

    elif func_name == "update_memory":
        memory_id = args["memory_id"]
        text = args["text"]
        try:
            memory.update(memory_id, text)
            return f"Updated [{memory_id}]: {text}"
        except Exception as e:
            return f"Error updating [{memory_id}]: {e}"

    elif func_name == "delete_memory":
        memory_id = args["memory_id"]
        try:
            memory.delete(memory_id)
            return f"Deleted [{memory_id}]"
        except Exception as e:
            return f"Error deleting [{memory_id}]: {e}"

    elif func_name == "get_all_memories":
        limit = args.get("limit", default_search_limit)
        response = memory.get_all(user_id=user_id, limit=limit)
        results = response.get("results", [])
        if results:
            lines = [f"- [id={r['id']}] {r['memory']}" for r in results]
            return "\n".join(lines)
        return "No memories stored."

    return f"Unknown tool: {func_name}"


def load_user_messages_to_mem0_agent(memory: Memory, sessions: list, user_id: str,
                                      loading_prompt: str, experiment: Experiment,
                                      run_config: dict, start: int = 0, end: int = -1):
    """Load user messages into mem0 via agent with add_memory tool.

    Args:
        user_id: From dataset.yaml
        loading_prompt: System prompt content for loading phase
        experiment: Experiment with model/api_key/base_url
        run_config: Runtime params (iterations, search_limit)
    """
    client = create_client_from_experiment(experiment)
    model_name = experiment.model

    user_messages = []
    for session in sessions:
        for msg in session['messages'][start: end]:
            if msg['role'] == 'user':
                user_messages.append(msg['content'])

    max_iterations = run_config.get('iterations', DEFAULT_AGENT_ITERATIONS)
    search_limit = run_config.get('search_limit', DEFAULT_AGENT_SEARCH_LIMIT)

    print(f"\nLoading {len(user_messages)} user messages into mem0 agent...")
    for content in tqdm(user_messages, desc=f"[{experiment.name}] mem0 agent loading"):
        messages = [
            {"role": "system", "content": loading_prompt},
            {"role": "user", "content": content},
        ]
        for _ in range(max_iterations):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=MEM0_AGENT_TOOLS_LOADING,
                tool_choice="auto",
            )
            assistant_msg = normalize_tool_calls(response.choices[0].message)
            messages.append(message_to_dict(assistant_msg))

            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    result = execute_mem0_tool_call(memory, tc, user_id, search_limit)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                break

    print(f"✓ Loaded {len(user_messages)} messages via mem0 agent")


def run_mem0_agent_query(memory: Memory, query_obj: dict, user_id: str,
                          query_prompt: str, experiment: Experiment,
                          run_config: dict, answer_idx: int) -> dict:
    """Run mem0 agent query with tool-calling loop.

    Args:
        user_id: From dataset.yaml
        query_prompt: System prompt content for query phase
        experiment: Experiment with model/api_key/base_url
        run_config: Runtime params (iterations, search_limit)
    """
    question = query_obj['question']

    client = create_client_from_experiment(experiment)
    model_name = experiment.model

    max_iterations = run_config.get('iterations', DEFAULT_AGENT_ITERATIONS)
    search_limit = run_config.get('search_limit', DEFAULT_AGENT_SEARCH_LIMIT)

    messages = [
        {"role": "system", "content": query_prompt},
        {"role": "user", "content": question},
    ]
    tool_calls_log = []

    assistant_msg = None
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=MEM0_AGENT_TOOLS_QUERY,
            tool_choice="auto",
        )
        assistant_msg = normalize_tool_calls(response.choices[0].message)
        messages.append(message_to_dict(assistant_msg))

        if assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                result = execute_mem0_tool_call(memory, tc, user_id, search_limit)
                tool_calls_log.append({
                    "iteration": iteration,
                    "tool": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            break

    if isinstance(query_obj['reference_answer'], list):
        ref_answer = query_obj['reference_answer'][answer_idx]
    else:
        ref_answer = query_obj['reference_answer']
    

    return {
        "query": question,
        "llm_response": assistant_msg.content if assistant_msg else "",
        "memory_state": {"tool_calls": tool_calls_log},
        "reference_answer": ref_answer,
        "metadata": {
            "system_prompt": query_prompt,
            "tool_calls_count": len(tool_calls_log),
        },
        "message_checkpoint": answer_idx
    }


# ============================================================================
# Output Generation
# ============================================================================

def save_results(data: dict, output_path: str):
    """Save results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_result_file(output_dir: Path, case_id: str, config_name: str,
                     results: list, timestamp: str, experiment: Experiment,
                     case_file: Path):
    """Save a single result file."""
    output = {
        "benchmark_timestamp": timestamp,
        "data_path": str(case_file),
        "memory_type": config_name,
        "config": {"model": experiment.model},
        "cases": [{"case_id": case_id, "results": results}],
    }
    output_path = output_dir / f"results_{case_id}_{config_name}.json"
    save_results(output, str(output_path))


# ============================================================================
# Parallel Execution
# ============================================================================

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=30, max=180),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: print(
        f"  Rate limit hit for {retry_state.args[0][0].get('case_id', 'unknown')}, "
        f"retrying in {retry_state.next_action.sleep:.0f}s (attempt {retry_state.attempt_number}/6)..."
    )
)
def run_agent_case(args) -> dict:
    """Run a single mem-agent case - can be called in parallel.

    Args:
        args: Tuple of (case_data, case_file, experiment, system_prompt_path,
                         mem_agent_config, script_dir, verbose)
    """
    case_data, case_file, experiment, system_prompt_path, mem_agent_config, script_dir, verbose, mem_checkpoints = args

    case_id = case_data.get('case_id', 'unknown')
    base_memory_path = mem_agent_config.get('memory_path', 'memory_mem_agent')
    memory_path = f"{base_memory_path}_{experiment.name}/{case_id}_mem_agent"

    print(f"  [{experiment.name}] Starting {case_id}...")

    agent = initialize_mem_agent(experiment, system_prompt_path, memory_path)
    case_results = []
    for i in range(len(mem_checkpoints) - 1):
        load_user_messages_to_agent(agent, case_data['sessions'], mem_checkpoints[i], mem_checkpoints[i + 1], verbose)
        for query_obj in case_data['queries']:
            result = run_mem_agent_query(agent, query_obj, memory_path, i)
            case_results.append(result)

    print(f"  [{experiment.name}] ✓ Completed {case_id}")

    return {
        'case_id': case_id,
        'case_file': str(case_file),
        'config_name': 'mem_agent',
        'results': case_results,
    }


def run_mem_agent_parallel(case_files: list[Path], experiment: Experiment,
                            system_prompt_path: str, mem_agent_config: dict,
                            script_dir: Path, max_workers: int = 3,
                            verbose: bool = False, mem_checkpoints: list[int] = [0, -1]) -> dict:
    """Run mem-agent benchmarks in parallel across cases."""
    tasks = []
    for case_file in case_files:
        case_data = load_benchmark_data(str(case_file))
        tasks.append((case_data, case_file, experiment, system_prompt_path,
                       mem_agent_config, script_dir, verbose, mem_checkpoints))

    print(f"\n[{experiment.name}] Running {len(tasks)} mem-agent tasks with {max_workers} parallel workers...")

    results = {}
    failed_cases = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_agent_case, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"[{experiment.name}] mem-agent"):
            task = futures[future]
            case_id = task[0].get('case_id', 'unknown')

            try:
                result = future.result()
                key = (result['case_id'], result['config_name'])
                results[key] = result
            except Exception as e:
                print(f"  [{experiment.name}] ✗ Error in {case_id}: {e}")
                failed_cases.append((case_id, str(e)))

    if failed_cases:
        print(f"\n[{experiment.name}] {len(failed_cases)} cases failed:")
        for case_id, error in failed_cases:
            print(f"  - {case_id}: {error}")

    return results


# ============================================================================
# Main
# ============================================================================

def run_experiment(experiment: Experiment, config: dict, script_dir: Path):
    """Run the full benchmark for a single experiment.

    Args:
        experiment: Experiment with resolved model/api_key/base_url and run config
        config: Full config dict (for mem0, mem_agent sections)
        script_dir: Path to benchmark directory
    """
    max_cases = config.get('max_cases')
    parallel_workers = config.get('parallel_workers', 1)
    verbose = config.get('verbose', False)
    mem_checkpoints = config.get('mem_checkpoints', [0, -1])
    if mem_checkpoints[0] != 0:
        mem_checkpoints = [0] + mem_checkpoints

    output_dir = script_dir / config.get('output_dir', 'eval_results') / experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    print(f"\n[{experiment.name}] LLM: {experiment.model}")
    print(f"[{experiment.name}] Output: {output_dir}")

    for data_dir_rel in experiment.data_dirs:
        data_dir = script_dir / data_dir_rel
        dataset = resolve_dataset(data_dir, max_cases)
        collection_name = f"{dataset.collection_name}_{experiment.name}"

        print(f"\n[{experiment.name}] Dataset: {data_dir_rel} ({len(dataset.case_files)} cases)")

        # ==================================================================
        # Phase 1: mem0 RAG
        # ==================================================================
        if 'mem0_rag' in experiment.run:
            run_cfg = experiment.run['mem0_rag']
            mem0_limits = run_cfg.get('limits', [10])
            mem0_infer_modes = run_cfg.get('infer', [False])

            print(f"\n[{experiment.name}] " + "="*60)
            print(f"[{experiment.name}] PHASE 1: mem0 RAG")
            print(f"[{experiment.name}]   infer modes: {mem0_infer_modes}, limits: {mem0_limits}")
            print(f"[{experiment.name}] " + "="*60)

            mem0_all_results = {}

            for infer_mode in mem0_infer_modes:
                infer_suffix = "infer" if infer_mode else "top"

                for case_file in dataset.case_files:
                    case_data = load_benchmark_data(str(case_file))
                    case_id = case_data.get('case_id', case_file.stem)
                    group_name = case_file.stem

                    print(f"\n  [{experiment.name}] Processing {case_id} (infer={infer_mode})...")

                    mem0 = initialize_mem0(config['mem0'], experiment, collection_name)
                    case_results = dict()
                    for i in range(len(mem_checkpoints) - 1):
                        load_user_messages_to_mem0(mem0, case_data['sessions'], dataset.user_id, mem_checkpoints[i], mem_checkpoints[i + 1], infer=infer_mode)

                        for limit in mem0_limits:
                            print(f"    [{experiment.name}] mem0 {infer_suffix}{limit}...")
                            
                            for query_obj in case_data['queries']:
                                result = run_mem0_query(mem0, query_obj, dataset.user_id, limit, experiment, i)
                                if limit in case_results:
                                    case_results[limit].append(result)
                                else:
                                    case_results[limit] = result
                    for limit in mem0_limits:
                        config_name = f"mem0_{infer_suffix}{limit}"
                        mem0_all_results[(group_name, config_name)] = {
                            "case_id": case_id,
                            "case_file": str(case_file),
                            "results": case_results[limit]
                        }

            # Save mem0 results
            print(f"\n  [{experiment.name}] Saving mem0 results...")
            for (group_name, config_name), case_result in mem0_all_results.items():
                save_result_file(
                    output_dir, group_name, config_name,
                    case_result['results'], timestamp, experiment,
                    Path(case_result['case_file']),
                )
            print(f"  [{experiment.name}] ✓ Saved {len(mem0_all_results)} mem0 result files")

        # ==================================================================
        # Phase 2: mem0 Agent (tool-calling)
        # ==================================================================
        if 'mem0_agent' in experiment.run and dataset.mem0_agent_loading_prompt and dataset.mem0_agent_query_prompt:
            run_cfg = experiment.run['mem0_agent']

            print(f"\n[{experiment.name}] " + "="*60)
            print(f"[{experiment.name}] PHASE 2: mem0 Agent (tool-calling)")
            print(f"[{experiment.name}] " + "="*60)

            mem0_agent_results = {}

            for case_file in dataset.case_files:
                case_data = load_benchmark_data(str(case_file))
                case_id = case_data.get('case_id', case_file.stem)
                group_name = case_file.stem

                print(f"\n  [{experiment.name}] Processing {case_id}...")

                try:
                    mem0 = initialize_mem0(config['mem0'], experiment, collection_name)
                    case_results = []
                    for i in range(len(mem_checkpoints) - 1):
                        load_user_messages_to_mem0_agent(
                            mem0, case_data['sessions'], dataset.user_id,
                            dataset.mem0_agent_loading_prompt, experiment, run_cfg, 
                            mem_checkpoints[i], mem_checkpoints[i + 1]
                        )

                        print(f"    [{experiment.name}] Querying...")
                        
                        for query_obj in case_data['queries']:
                            result = run_mem0_agent_query(
                                mem0, query_obj, dataset.user_id,
                                dataset.mem0_agent_query_prompt, experiment, run_cfg,
                            )
                            result["message_idx"] = mem_checkpoints[i + 1]
                            case_results.append(result)

                    mem0_agent_results[(group_name, 'mem0_agent')] = {
                        "case_id": case_id,
                        "case_file": str(case_file),
                        "results": case_results,
                    }
                except Exception as e:
                    print(f"  [{experiment.name}] ✗ FAILED {case_id}: {e}")
                    continue

            # Save mem0 agent results
            print(f"\n  [{experiment.name}] Saving mem0 agent results...")
            for (group_name, config_name), case_result in mem0_agent_results.items():
                save_result_file(
                    output_dir, group_name, config_name,
                    case_result['results'], timestamp, experiment,
                    Path(case_result['case_file']),
                )
            print(f"  [{experiment.name}] ✓ Saved {len(mem0_agent_results)} mem0 agent result files")
        elif 'mem0_agent' in experiment.run:
            print(f"\n[{experiment.name}] PHASE 2: SKIPPED (no mem0_agent prompts in dataset)")

        # ==================================================================
        # Phase 3: mem-agent (parallel)
        # ==================================================================
        if 'mem_agent' in experiment.run and dataset.mem_agent_system_prompt:
            print(f"\n[{experiment.name}] " + "="*60)
            print(f"[{experiment.name}] PHASE 3: mem-agent (parallel_workers={parallel_workers})")
            print(f"[{experiment.name}] " + "="*60)

            agent_count = 0
            if parallel_workers > 1:
                agent_results = run_mem_agent_parallel(
                    dataset.case_files, experiment, dataset.mem_agent_system_prompt,
                    config['mem_agent'], script_dir, parallel_workers, verbose, mem_checkpoints,
                )
                if agent_results:
                    print(f"\n  [{experiment.name}] Saving mem-agent results...")
                    for (case_id, config_name), result in agent_results.items():
                        save_result_file(
                            output_dir, case_id, config_name,
                            result['results'], timestamp, experiment,
                            Path(result['case_file']),
                        )
                    agent_count = len(agent_results)
            else:
                for case_file in dataset.case_files:
                    case_data = load_benchmark_data(str(case_file))
                    try:
                        result = run_agent_case((
                            case_data, case_file, experiment,
                            dataset.mem_agent_system_prompt,
                            config['mem_agent'], script_dir, verbose, mem_checkpoints,
                        ))
                        save_result_file(
                            output_dir, result['case_id'], result['config_name'],
                            result['results'], timestamp, experiment, case_file,
                        )
                        agent_count += 1
                        print(f"    [{experiment.name}] ✓ Saved results_{result['case_id']}_{result['config_name']}.json")
                    except Exception as e:
                        group_name = case_file.stem
                        print(f"    [{experiment.name}] ✗ FAILED {group_name}: {e}")

            if agent_count:
                print(f"  [{experiment.name}] ✓ Saved {agent_count} mem-agent result files")
        elif 'mem_agent' in experiment.run:
            print(f"\n[{experiment.name}] PHASE 3: SKIPPED (no mem_agent prompt in dataset)")

    print(f"\n[{experiment.name}] " + "="*60)
    print(f"[{experiment.name}] ✓ EXPERIMENT COMPLETE!")
    print(f"[{experiment.name}] " + "="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run memory benchmark')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Config file path (default: config.yaml)')
    parser.add_argument('--clean-memory', action='store_true',
                        help='Clean Qdrant collections and mem-agent directories before run')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    config = load_config(str(config_path))
    print(f"Using config: {args.config}")

    # Resolve experiments
    experiments = resolve_experiments(config)

    print(f"Experiments: {[e.name for e in experiments]}")

    # Clean memory if requested or configured
    should_clean = args.clean_memory or config.get('auto_clean_memory', False)
    if should_clean:
        clean_memory(config, script_dir, experiments)

    # Run experiments sequentially
    for i, experiment in enumerate(experiments):
        print(f"\n{'#'*80}")
        print(f"  EXPERIMENT {i+1}/{len(experiments)}: {experiment.name}")
        print(f"{'#'*80}")
        run_experiment(experiment, config, script_dir)

    print("\n" + "="*80)
    print("ALL BENCHMARKS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
