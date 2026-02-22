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

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# mem0 auto-detects OPENROUTER_API_KEY and switches provider — prevent this
os.environ.pop('OPENROUTER_API_KEY', None)

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

def clean_memory(config: dict, script_dir: Path, model_profiles: list = None):
    """Clean Qdrant collections and mem-agent directories before benchmark run.

    Args:
        config: Full config dict
        script_dir: Path to benchmark directory
        model_profiles: List of model profiles (to clean their specific collections/paths)
    """
    print("\n" + "="*60)
    print("MEMORY CLEANUP")
    print("="*60)

    # 1. Clean Qdrant collections
    mem0_config = config.get('mem0', {})
    vector_db_config = mem0_config.get('vector_db', {})
    qdrant_path_str = vector_db_config.get('path', './qdrant_data')
    base_collection = vector_db_config.get('collection_name', 'benchmark_location')

    qdrant_path = script_dir / qdrant_path_str

    if qdrant_path.exists():
        client = None
        try:
            client = QdrantClient(path=str(qdrant_path))
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            # Find collections matching our base name
            to_delete = [name for name in collection_names if name.startswith(base_collection)]

            if to_delete:
                print(f"  Deleting {len(to_delete)} Qdrant collection(s):")
                for name in to_delete:
                    try:
                        client.delete_collection(name)
                        print(f"    ✓ {name}")
                    except Exception as e:
                        print(f"    ⚠ Failed to delete {name}: {e}")
            else:
                print(f"  No Qdrant collections matching '{base_collection}*' found")
        except Exception as e:
            print(f"  ⚠ Error cleaning Qdrant: {e}")
            # Fallback: delete entire qdrant_data directory
            print(f"  Fallback: deleting {qdrant_path}")
            shutil.rmtree(qdrant_path, ignore_errors=True)
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass  # Ignore close errors
    else:
        print(f"  Qdrant path {qdrant_path} does not exist, skipping")

    # 2. Clean mem-agent memory directories
    mem_agent_config = config.get('mem_agent', {})
    base_memory_path_str = mem_agent_config.get('memory_path', 'memory_mem_agent')
    base_memory_path = script_dir / base_memory_path_str

    # Also clean model-specific paths if profiles provided
    memory_paths_to_clean = [base_memory_path]
    if model_profiles:
        for profile in model_profiles:
            suffix = profile.get('memory_path_suffix', '')
            if suffix:
                memory_paths_to_clean.append(script_dir / (base_memory_path_str + suffix))

    for mem_path in memory_paths_to_clean:
        if mem_path.exists():
            shutil.rmtree(mem_path, ignore_errors=True)
            print(f"  ✓ Deleted mem-agent directory: {mem_path}")
        else:
            print(f"  mem-agent path {mem_path} does not exist, skipping")

    print("✓ Memory cleanup complete\n")


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

def initialize_mem0(config: dict, collection_suffix: str = "") -> Memory:
    """Initialize mem0 Memory instance

    Args:
        config: mem0 config dict (with llm, embedder, vector_db sections)
        collection_suffix: Optional suffix for qdrant collection name (e.g. '_gemini')
    """
    # mem0 auto-detects OPENROUTER_API_KEY and switches provider — prevent this
    os.environ.pop('OPENROUTER_API_KEY', None)

    collection_name = config['vector_db']['collection_name'] + collection_suffix

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
                    "embedding_dims": config['embedder'].get('embedding_dims', 3072),
                },
            ),
            vector_store=VectorStoreConfig(
                provider=config['vector_db']['provider'],
                config={
                    "collection_name": collection_name,
                    "path": config['vector_db']['path'],
                    "embedding_model_dims": config['vector_db']['embedding_model_dims'],
                },
            ),
        )
    )
    memory.reset()
    return memory


def initialize_mem_agent(config: dict, prompt_path: str, memory_path: str,
                         model_profile: dict = None) -> Agent:
    """Initialize Agent instance with specific prompt and memory path

    Args:
        config: mem_agent config dict
        prompt_path: Path to system prompt file
        memory_path: Path for agent memory storage
        model_profile: Optional model profile dict with 'llm' section
    """
    # Determine model/api_key/base_url from profile or fallback to config
    if model_profile:
        model_name = model_profile['llm']['model']
        api_key = model_profile['llm']['api_key']
        base_url = model_profile['llm'].get('base_url')
    else:
        model_name = config['model']
        api_key = config['api_key']
        base_url = None

    os.environ['OPENAI_API_KEY'] = api_key
    path = Path(memory_path)
    if path.exists():
        shutil.rmtree(path)
    agent = Agent(
        model=model_name,
        memory_path=memory_path,
        use_vllm=False,
        system_prompt_path=prompt_path,
        api_key=api_key,
        base_url=base_url,
    )
    agent._client._client = Client(
        base_url=agent._client._client.base_url, verify=False,
        timeout=120.0, limits=DEFAULT_CONNECTION_LIMITS, follow_redirects=True,
    )
    return agent


# ============================================================================
# Session Loading
# ============================================================================

def load_user_messages_to_mem0(memory: Memory, sessions: list, config: dict, infer: bool = False):
    """Load user messages into mem0

    Args:
        memory: mem0 Memory instance
        sessions: List of conversation sessions
        config: mem0 config dict
        infer: Whether to use LLM inference for fact extraction (slower but smarter)
    """
    user_messages = []
    for session in sessions:
        for msg in session['messages']:
            if msg['role'] == 'user':
                user_messages.append({'role': 'user', 'content': msg['content']})

    user_id = config['benchmark']['user_id']

    infer_label = "infer" if infer else "raw"
    print(f"\nLoading {len(user_messages)} user messages into mem0 ({infer_label})...")
    for msg in tqdm(user_messages, desc=f"mem0 {infer_label}"):
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

def run_mem0_query(memory: Memory, query_obj: dict, config: dict, limit: int,
                   model_profile: dict = None) -> dict:
    """Run mem0 query with specific retrieve limit

    Args:
        model_profile: Optional model profile dict. If provided, uses its 'llm'
                       section for the answer generation LLM.
    """
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

    # Get LLM response — use model_profile if available
    if model_profile:
        client = create_llm_client(model_profile['llm'])
        model_name = model_profile['llm']['model']
    else:
        client = OpenAI(api_key=config['llm']['api_key'])
        model_name = config['llm']['model']

    llm_response = client.chat.completions.create(
        model=model_name,
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

# Default values (can be overridden via config)
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
    """Execute a single mem0 tool call and return result string.

    Args:
        memory: mem0 Memory instance
        tool_call: OpenAI tool call object
        user_id: User ID for memory operations
        default_search_limit: Default limit for get_all_memories (from config)
    """
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


def load_user_messages_to_mem0_agent(memory: Memory, sessions: list, config: dict,
                                      loading_prompt: str, model_profile: dict = None):
    """Load user messages into mem0 via agent with add_memory tool.

    The model decides what facts to extract and how to store them.

    Args:
        model_profile: Optional model profile. Uses 'llm_tool_calling' for tool-calling endpoint.
    """
    user_id = config['benchmark']['user_id']

    if model_profile:
        client = create_llm_client(model_profile['llm_tool_calling'])
        model_name = model_profile['llm_tool_calling']['model']
        desc_prefix = f"[{model_profile['name']}] "
    else:
        client = OpenAI(api_key=config['llm']['api_key'])
        model_name = config['llm']['model']
        desc_prefix = ""

    user_messages = []
    for session in sessions:
        for msg in session['messages']:
            if msg['role'] == 'user':
                user_messages.append(msg['content'])

    max_iterations = config['benchmark'].get('mem0_agent_iterations', DEFAULT_AGENT_ITERATIONS)
    search_limit = config['benchmark'].get('mem0_agent_search_limit', DEFAULT_AGENT_SEARCH_LIMIT)

    print(f"\nLoading {len(user_messages)} user messages into mem0 agent...")
    for content in tqdm(user_messages, desc=f"{desc_prefix}mem0 agent loading"):
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


def run_mem0_agent_query(memory: Memory, query_obj: dict, config: dict,
                          query_prompt: str, model_profile: dict = None) -> dict:
    """Run mem0 agent query with tool-calling loop.

    Args:
        model_profile: Optional model profile. Uses 'llm_tool_calling' for tool-calling endpoint.
    """
    question = query_obj['question']
    user_id = config['benchmark']['user_id']

    if model_profile:
        client = create_llm_client(model_profile['llm_tool_calling'])
        model_name = model_profile['llm_tool_calling']['model']
    else:
        client = OpenAI(api_key=config['llm']['api_key'])
        model_name = config['llm']['model']

    max_iterations = config['benchmark'].get('mem0_agent_iterations', DEFAULT_AGENT_ITERATIONS)
    search_limit = config['benchmark'].get('mem0_agent_search_limit', DEFAULT_AGENT_SEARCH_LIMIT)

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

    return {
        "query": question,
        "llm_response": assistant_msg.content if assistant_msg else "",
        "memory_state": {"tool_calls": tool_calls_log},
        "reference_answer": query_obj['reference_answer'],
        "metadata": {
            "system_prompt": query_prompt,
            "tool_calls_count": len(tool_calls_log),
        },
    }


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

def process_case_mem0(mem0: Memory, case_data: dict, config: dict, limits: list,
                      model_profile: dict = None) -> dict:
    """Process a single case with mem0 at different limits. Returns {limit: results}"""
    results_by_limit = {}

    for limit in limits:
        print(f"    mem0 top-{limit}...")
        case_results = []
        for query_obj in case_data['queries']:
            result = run_mem0_query(mem0, query_obj, config, limit, model_profile=model_profile)
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
# Parallel Execution
# ============================================================================

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=30, max=180),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: print(
        f"  ⏳ Rate limit hit for {retry_state.args[0][0].get('case_id', 'unknown')}, "
        f"retrying in {retry_state.next_action.sleep:.0f}s (attempt {retry_state.attempt_number}/6)..."
    )
)
def run_agent_case(args) -> dict:
    """Run a single mem-agent case - can be called in parallel.

    Args:
        args: Tuple of (case_data, config, prompt_cfg, script_dir[, model_profile])

    Returns:
        Dict with case_id, config_name, prompt_path, and results
    """
    if len(args) == 6:
        case_data, data_path, config, prompt_cfg, script_dir, model_profile = args
    elif len(args) == 5:
        case_data, data_path, config, prompt_cfg, script_dir = args
        model_profile = None
    else:
        case_data, config, prompt_cfg, script_dir = args
        data_path = None
        model_profile = None

    case_id = case_data.get('case_id', 'unknown')

    # Memory path with optional suffix for model isolation
    mem_suffix = model_profile.get('memory_path_suffix', '') if model_profile else ''
    base_memory_path = config['mem_agent']['memory_path'] + mem_suffix
    memory_path = f"{base_memory_path}/{case_id}_{prompt_cfg['name']}"

    profile_name = model_profile['name'] if model_profile else 'default'
    print(f"  [{profile_name}] Starting {case_id} with {prompt_cfg['name']}...")

    agent = initialize_mem_agent(
        config['mem_agent'],
        str(script_dir / prompt_cfg['path']),
        memory_path,
        model_profile=model_profile,
    )
    load_user_messages_to_agent(agent, case_data['sessions'], config['benchmark'].get('verbose', False))
    results = process_case_agent(agent, case_data, config['mem_agent'], memory_path)

    print(f"  [{profile_name}] ✓ Completed {case_id} with {prompt_cfg['name']}")

    return {
        'case_id': case_id,
        'data_path': data_path,
        'config_name': prompt_cfg['name'],
        'prompt_path': prompt_cfg['path'],
        'results': results
    }


def run_mem_agent_parallel(data_paths: list, config: dict, prompt_configs: list,
                           script_dir: Path, max_workers: int = 3,
                           model_profile: dict = None) -> dict:
    """Run mem-agent benchmarks in parallel across cases.

    Args:
        data_paths: List of data file paths
        config: Full config dict
        prompt_configs: List of prompt configurations
        script_dir: Script directory path
        max_workers: Maximum parallel workers
        model_profile: Optional model profile dict

    Returns:
        Dict mapping (case_id, config_name) to result dict
    """
    # Prepare all tasks
    tasks = []
    for data_path in data_paths:
        case_data = load_benchmark_data(str(script_dir / data_path))
        for prompt_cfg in prompt_configs:
            tasks.append((case_data, data_path, config, prompt_cfg, script_dir, model_profile))

    profile_name = model_profile['name'] if model_profile else 'default'
    print(f"\n[{profile_name}] Running {len(tasks)} mem-agent tasks with {max_workers} parallel workers...")

    # Execute in parallel
    results = {}
    failed_cases = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_agent_case, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"[{profile_name}] mem-agent"):
            task = futures[future]
            case_id = task[0].get('case_id', 'unknown')
            config_name = task[2]['name']

            try:
                result = future.result()
                key = (result['case_id'], result['config_name'])
                results[key] = result
            except Exception as e:
                print(f"  [{profile_name}] ✗ Error in {case_id}/{config_name}: {e}")
                failed_cases.append((case_id, config_name, str(e)))

    if failed_cases:
        print(f"\n[{profile_name}] ⚠ {len(failed_cases)} cases failed:")
        for case_id, config_name, error in failed_cases:
            print(f"  - {case_id}/{config_name}: {error}")

    return results


# ============================================================================
# Main
# ============================================================================

def run_benchmark_for_model(model_profile: dict, config: dict, script_dir: Path):
    """Run the full benchmark (all 3 phases) for a single model profile.

    Args:
        model_profile: Dict with 'name', 'llm', 'llm_tool_calling', 'output_dir', etc.
        config: Full shared config dict
        script_dir: Path to benchmark directory
    """
    profile_name = model_profile['name']
    collection_suffix = model_profile.get('qdrant_suffix', '')
    mem_suffix = model_profile.get('memory_path_suffix', '')

    data_paths = config['benchmark'].get('data_paths', [])
    prompt_configs = config['benchmark'].get('agent_prompts', [])
    mem0_limits = config['benchmark'].get('mem0_limits', [10])
    mem0_infer_modes = config['benchmark'].get('mem0_infer', [False])
    parallel_workers = config['benchmark'].get('parallel_workers', 1)
    mem0_agent_prompts = config['benchmark'].get('mem0_agent_prompts', [])

    output_dir = script_dir / model_profile.get('output_dir', 'eval_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    model_display = model_profile['llm']['model']

    print(f"\n[{profile_name}] Running benchmark on {len(data_paths)} data file(s)")
    print(f"[{profile_name}] LLM: {model_display}")
    print(f"[{profile_name}] Output: {output_dir}")

    # ========================================================================
    # Phase 1: Run mem0 benchmarks (fast, sequential)
    # ========================================================================
    if mem0_limits:
        print(f"\n[{profile_name}] " + "="*60)
        print(f"[{profile_name}] PHASE 1: mem0 benchmarks")
        print(f"[{profile_name}]   infer modes: {mem0_infer_modes}, limits: {mem0_limits}")
        print(f"[{profile_name}] " + "="*60)

        mem0_all_results = {}

        for infer_mode in mem0_infer_modes:
            infer_suffix = "infer" if infer_mode else "top"

            for data_path in data_paths:
                case_data = load_benchmark_data(str(script_dir / data_path))
                case_id = case_data.get('case_id', Path(data_path).stem)
                group_name = Path(data_path).stem

                print(f"\n  [{profile_name}] Processing {case_id} (infer={infer_mode})...")

                mem0 = initialize_mem0(config['mem0'], collection_suffix=collection_suffix)
                load_user_messages_to_mem0(mem0, case_data['sessions'], config['mem0'], infer=infer_mode)

                for limit in mem0_limits:
                    print(f"    [{profile_name}] mem0 {infer_suffix}{limit}...")
                    case_results = []
                    for query_obj in case_data['queries']:
                        result = run_mem0_query(mem0, query_obj, config['mem0'], limit,
                                                model_profile=model_profile)
                        case_results.append(result)

                    config_name = f"mem0_{infer_suffix}{limit}"
                    mem0_all_results[(group_name, config_name)] = {
                        "case_id": case_id,
                        "data_path": data_path,
                        "results": case_results
                    }

        # Save mem0 results
        print(f"\n  [{profile_name}] Saving mem0 results...")
        for (group_name, config_name), case_result in mem0_all_results.items():
            output = {
                "benchmark_timestamp": timestamp,
                "data_path": case_result.pop("data_path"),
                "memory_type": config_name,
                "config": {"model": model_display},
                "cases": [case_result]
            }
            output_path = output_dir / f"results_{group_name}_{config_name}.json"
            save_results(output, str(output_path))
        print(f"  [{profile_name}] ✓ Saved {len(mem0_all_results)} mem0 result files")

    # ========================================================================
    # Phase 1.5: Run mem0 agent benchmarks (tool-calling, sequential)
    # ========================================================================
    if mem0_agent_prompts:
        tc_model = model_profile['llm_tool_calling']['model']
        print(f"\n[{profile_name}] " + "="*60)
        print(f"[{profile_name}] PHASE 1.5: mem0 agent benchmarks (tool-calling)")
        print(f"[{profile_name}]   tool-calling model: {tc_model}")
        print(f"[{profile_name}] " + "="*60)

        mem0_agent_results = {}

        for prompt_cfg in mem0_agent_prompts:
            loading_prompt_path = script_dir / prompt_cfg['loading_prompt']
            query_prompt_path = script_dir / prompt_cfg['query_prompt']
            with open(loading_prompt_path, 'r') as f:
                loading_prompt = f.read()
            with open(query_prompt_path, 'r') as f:
                query_prompt = f.read()

            for data_path in data_paths:
                case_data = load_benchmark_data(str(script_dir / data_path))
                case_id = case_data.get('case_id', Path(data_path).stem)
                group_name = Path(data_path).stem

                print(f"\n  [{profile_name}] Processing {case_id} with {prompt_cfg['name']}...")

                try:
                    mem0 = initialize_mem0(config['mem0'], collection_suffix=collection_suffix)
                    load_user_messages_to_mem0_agent(
                        mem0, case_data['sessions'], config['mem0'], loading_prompt,
                        model_profile=model_profile,
                    )

                    print(f"    [{profile_name}] Querying...")
                    case_results = []
                    for query_obj in case_data['queries']:
                        result = run_mem0_agent_query(
                            mem0, query_obj, config['mem0'], query_prompt,
                            model_profile=model_profile,
                        )
                        case_results.append(result)

                    mem0_agent_results[(group_name, prompt_cfg['name'])] = {
                        "case_id": case_id,
                        "data_path": data_path,
                        "results": case_results,
                    }
                except Exception as e:
                    print(f"  [{profile_name}] ✗ FAILED {case_id}: {e}")
                    continue

        # Save mem0 agent results
        print(f"\n  [{profile_name}] Saving mem0 agent results...")
        for (group_name, config_name), case_result in mem0_agent_results.items():
            output = {
                "benchmark_timestamp": timestamp,
                "data_path": case_result.pop("data_path"),
                "memory_type": config_name,
                "config": {"model": tc_model},
                "cases": [case_result],
            }
            output_path = output_dir / f"results_{group_name}_{config_name}.json"
            save_results(output, str(output_path))
        print(f"  [{profile_name}] ✓ Saved {len(mem0_agent_results)} mem0 agent result files")
    else:
        print(f"\n[{profile_name}] PHASE 1.5: SKIPPED (no mem0_agent_prompts)")

    # ========================================================================
    # Phase 2: Run mem-agent benchmarks (slow, PARALLEL)
    # ========================================================================
    if not prompt_configs:
        print(f"\n[{profile_name}] PHASE 2: SKIPPED (no agent_prompts)")
    else:
        print(f"\n[{profile_name}] " + "="*60)
        print(f"[{profile_name}] PHASE 2: mem-agent benchmarks (parallel_workers={parallel_workers})")
        print(f"[{profile_name}] " + "="*60)

    agent_count = 0
    if prompt_configs and parallel_workers > 1:
        agent_results = run_mem_agent_parallel(
            data_paths, config, prompt_configs, script_dir, parallel_workers,
            model_profile=model_profile,
        )
        if agent_results:
            print(f"\n  [{profile_name}] Saving mem-agent results...")
            for (case_id, config_name), result in agent_results.items():
                output = {
                    "benchmark_timestamp": timestamp,
                    "data_path": result.get('data_path', ''),
                    "memory_type": config_name,
                    "config": {"model": model_display},
                    "cases": [{
                        "case_id": case_id,
                        "prompt_path": result['prompt_path'],
                        "results": result['results']
                    }]
                }
                output_path = output_dir / f"results_{case_id}_{config_name}.json"
                save_results(output, str(output_path))
            agent_count = len(agent_results)
    elif prompt_configs:
        for data_path in data_paths:
            case_data = load_benchmark_data(str(script_dir / data_path))
            for prompt_cfg in prompt_configs:
                try:
                    result = run_agent_case((case_data, data_path, config, prompt_cfg, script_dir, model_profile))
                    case_id = result['case_id']
                    config_name = result['config_name']
                    output = {
                        "benchmark_timestamp": timestamp,
                        "data_path": result.get('data_path', data_path),
                        "memory_type": config_name,
                        "config": {"model": model_display},
                        "cases": [{
                            "case_id": case_id,
                            "prompt_path": result['prompt_path'],
                            "results": result['results']
                        }]
                    }
                    output_path = output_dir / f"results_{case_id}_{config_name}.json"
                    save_results(output, str(output_path))
                    agent_count += 1
                    print(f"    [{profile_name}] ✓ Saved {output_path.name}")
                except Exception as e:
                    group_name = Path(data_path).stem
                    print(f"    [{profile_name}] ✗ FAILED {group_name}/{prompt_cfg['name']}: {e}")

    if agent_count:
        print(f"  [{profile_name}] ✓ Saved {agent_count} mem-agent result files")

    print(f"\n[{profile_name}] " + "="*60)
    print(f"[{profile_name}] ✓ MODEL COMPLETE!")
    print(f"[{profile_name}] " + "="*60)


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

    data_paths = config['benchmark'].get('data_paths', [])
    if not data_paths:
        data_paths = [c['data_path'] for c in config['benchmark'].get('cases', [])]

    # Get model profiles (new format) or build a default one (backward compat)
    model_profiles = config.get('model_profiles', None)

    if model_profiles is None:
        # Backward compatibility: build a single profile from old config format
        model_profiles = [{
            'name': config['mem0']['llm']['model'],
            'llm': {
                'model': config['mem0']['llm']['model'],
                'api_key': config['mem0']['llm']['api_key'],
            },
            'llm_tool_calling': {
                'model': config['mem0']['llm']['model'],
                'api_key': config['mem0']['llm']['api_key'],
            },
            'output_dir': config['benchmark'].get('output_dir', 'eval_results'),
            'qdrant_suffix': '',
            'memory_path_suffix': '',
        }]

    print(f"Models: {[p['name'] for p in model_profiles]}")
    print(f"Data files: {len(data_paths)}")

    # Clean memory if requested or configured
    should_clean = args.clean_memory or config['benchmark'].get('auto_clean_memory', False)
    if should_clean:
        clean_memory(config, script_dir, model_profiles)

    # Run benchmark for each model profile sequentially
    for i, profile in enumerate(model_profiles):
        print(f"\n{'#'*80}")
        print(f"  MODEL {i+1}/{len(model_profiles)}: {profile['name']}")
        print(f"{'#'*80}")
        run_benchmark_for_model(profile, config, script_dir)

    print("\n" + "="*80)
    print("✓ ALL BENCHMARKS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
