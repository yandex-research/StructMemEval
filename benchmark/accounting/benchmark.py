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
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from httpx import Client
from openai._base_client import DEFAULT_TIMEOUT, DEFAULT_CONNECTION_LIMITS

import yaml
from tqdm import tqdm
from openai import OpenAI, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


QUESTION = """Based on the memories from our previous dialogs, calculate for each person their total spending and determine who owes money to whom for settlement. Provide a final settlement plan showing all necessary transactions to clear all debts. Format your answer as a concise list of transactions only, using the format: [Person A] pays [amount] to [Person B], etc."""
MEMORY_LIMITS = {0: 10, 1: 20, 2: 50, 3: 100, 4: 200}
CUR_LIMIT = 0
MESSAGE_NUM = [0, 10, 20, 50]

# ============================================================================
# mem0 Agent Tool Definitions (for tool-calling approach)
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

DEFAULT_AGENT_ITERATIONS = 5
DEFAULT_AGENT_SEARCH_LIMIT = 50


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
# mem0 Agent Helpers (tool-calling)
# ============================================================================

def create_llm_client(llm_config: dict) -> OpenAI:
    """Create OpenAI-compatible client from LLM config dict."""
    kwargs = {'api_key': llm_config['api_key'], 'max_retries': 5}
    if llm_config.get('base_url'):
        kwargs['base_url'] = llm_config['base_url']
        kwargs['http_client'] = Client(
            verify=False, timeout=DEFAULT_TIMEOUT,
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


def message_to_dict(message):
    """Convert OpenAI ChatCompletionMessage to dict for messages list."""
    d = {"role": message.role, "content": message.content or ""}
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


@retry(
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, ConnectionError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    before_sleep=lambda rs: print(f"  ⏳ Retry {rs.attempt_number}/5 after timeout..."),
    reraise=True,
)
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


# ============================================================================
# Session Loading
# ============================================================================

def load_user_messages_to_mem0(memory: Memory, session: list, config: dict, start=0, end=10):
    """Load user messages into mem0"""
    user_messages = []
    for msg in session['messages'][start:end]:
        if msg['role'] == 'user':
            user_messages.append({'role': 'user', 'content': msg['content']})

    user_id = config['benchmark']['user_id']
    infer = config['benchmark']['infer']

    print(f"\nLoading {len(user_messages)} user messages into mem0...")
    for msg in tqdm(user_messages, desc="mem0 loading"):
        memory.add([msg], user_id=user_id, infer=infer)

    print(f"✓ Loaded {len(user_messages)} messages")


def load_user_messages_to_agent(agent: Agent, session: list, verbose: bool = False, start=0, end=10):
    """Load user messages into mem-agent"""
    user_messages = []
    for msg in session['messages'][start:end]:
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


def load_user_messages_to_mem0_agent(memory: Memory, session: dict, config: dict,
                                      loading_prompt: str, start=0, end=10):
    """Load user messages into mem0 via agent with tool-calling.

    Args:
        memory: mem0 Memory instance
        session: Single session dict with 'messages' key
        config: mem0 config section (config['mem0'])
        loading_prompt: System prompt for loading phase
        start: Start index in messages list
        end: End index in messages list
    """
    user_id = config['benchmark']['user_id']
    llm_config = config['llm']

    client = create_llm_client({
        'api_key': llm_config['api_key'],
        'base_url': llm_config.get('openrouter_base_url'),
    })
    model_name = llm_config['model']

    user_messages = []
    for msg in session['messages'][start:end]:
        if msg['role'] == 'user':
            user_messages.append(msg['content'])

    max_iterations = config['benchmark'].get('mem0_agent_iterations', DEFAULT_AGENT_ITERATIONS)
    search_limit = config['benchmark'].get('mem0_agent_search_limit', DEFAULT_AGENT_SEARCH_LIMIT)

    @retry(
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        before_sleep=lambda rs: print(f"  ⏳ LLM retry {rs.attempt_number}/3..."),
        reraise=True,
    )
    def _call_llm(client, model_name, messages, tools):
        return client.chat.completions.create(
            model=model_name, messages=messages, tools=tools, tool_choice="auto",
        )

    print(f"\nLoading {len(user_messages)} user messages into mem0 agent...")
    for content in tqdm(user_messages, desc="mem0 agent loading"):
        messages = [
            {"role": "system", "content": loading_prompt},
            {"role": "user", "content": content},
        ]
        for _ in range(max_iterations):
            response = _call_llm(client, model_name, messages, MEM0_AGENT_TOOLS_LOADING)
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


# ============================================================================
# Query Execution - mem0
# ============================================================================

def run_mem0_query(memory: Memory, answer_gt: str, config: dict, limit: int) -> dict:
    """Run mem0 query with specific retrieve limit"""
    #question = query_obj['question']
    user_id = config['benchmark']['user_id']

    # Search with specified limit
    response = memory.search(QUESTION, user_id=user_id, limit=limit)
    results = response.get('results', [])

    # Get retrieved memories
    retrieved_memories = [r['memory'] for r in results]

    # Build system prompt
    memory_context = "\n".join(f"- {mem}" for mem in retrieved_memories) if retrieved_memories else "No relevant memories."
    system_prompt = f"""You are a travel expense tracking assistant.

Memory Context:
{memory_context}

Save expenses as: "[Payer] paid [amount] for [description] covering [people] on [date]".

Hint: For settlement, track pairwise debts (e.g., Alice->Bob X euro, Alice->Charlie Y euro, etc.).

Use saved expenses to answer questions."""

    # Get LLM response
    llm_response = memory.llm.client.chat.completions.create(
        model=config['llm']['model'],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUESTION}
        ]
    )
    answer_llm = llm_response.choices[0].message.content

    # Build result
    result = {
        "query": QUESTION,
        "llm_response": answer_llm,
        "memory_state": {
            "retrieved_memories": [
                {"score": r.get('score', 0), "text": r['memory']}
                for r in results
            ],
            "total_memories": len(retrieved_memories)
        },
        "reference_answer": answer_gt,
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


def run_mem_agent_query(agent: Agent, answer: str, config: dict) -> dict:
    """Run mem-agent query and return result dict"""
    #question = query_obj['question']

    # Reset agent conversation to only system prompt before query
    agent.messages = agent.messages[:1]

    # Get response
    response = agent.chat(QUESTION)

    # Get memory state
    memory_path = config['memory_path']
    memory_files = get_memory_files(memory_path)
    memory_content = read_memory_content(memory_path)

    # Build result
    result = {
        "query": QUESTION,
        "llm_response": response.reply,
        "memory_state": {
            "memory_files": memory_files,
            "memory_content": memory_content
        },
        "reference_answer": answer,
        "metadata": {
            "agent_thoughts": response.thoughts,
            "python_block": response.python_block
        }
    }

    return result


# ============================================================================
# Query Execution - mem0 agent (tool-calling)
# ============================================================================

def run_mem0_agent_query(memory: Memory, answer_gt: str, config: dict,
                          query_prompt: str) -> dict:
    """Run mem0 agent query with tool-calling loop.

    Args:
        memory: mem0 Memory instance
        answer_gt: Ground truth answer string
        config: mem0 config section (config['mem0'])
        query_prompt: System prompt for query phase
    """
    user_id = config['benchmark']['user_id']
    llm_config = config['llm']

    client = create_llm_client({
        'api_key': llm_config['api_key'],
        'base_url': llm_config.get('openrouter_base_url'),
    })
    model_name = llm_config['model']

    max_iterations = config['benchmark'].get('mem0_agent_iterations', DEFAULT_AGENT_ITERATIONS)
    search_limit = config['benchmark'].get('mem0_agent_search_limit', DEFAULT_AGENT_SEARCH_LIMIT)

    @retry(
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        before_sleep=lambda rs: print(f"  ⏳ Query LLM retry {rs.attempt_number}/3..."),
        reraise=True,
    )
    def _call_llm_query(client, model_name, messages, tools):
        return client.chat.completions.create(
            model=model_name, messages=messages, tools=tools, tool_choice="auto",
        )

    messages = [
        {"role": "system", "content": query_prompt},
        {"role": "user", "content": QUESTION},
    ]
    tool_calls_log = []

    assistant_msg = None
    for iteration in range(max_iterations):
        response = _call_llm_query(client, model_name, messages, MEM0_AGENT_TOOLS_QUERY)
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
        "query": QUESTION,
        "llm_response": (assistant_msg.content or "") if assistant_msg else "",
        "memory_state": {"tool_calls": tool_calls_log},
        "reference_answer": answer_gt,
        "metadata": {
            "system_prompt": query_prompt,
            "tool_calls_count": len(tool_calls_log),
        },
    }


def process_case_mem0_agent(mem0: Memory, session: dict, config: dict,
                             query_prompt: str, start_idx=0) -> list:
    """Process a single checkpoint with mem0 agent query. Returns list of results."""
    case_results = []
    result = run_mem0_agent_query(mem0, session['answers'][start_idx], config, query_prompt)
    case_results.append(result)
    return case_results


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

def process_case_mem0(mem0: Memory, session: dict, config: dict, limits: list, start_idx=0) -> dict:
    """Process a single case with mem0 at different limits. Returns {limit: results}"""
    results_by_limit = {}

    for limit in limits:
        print(f"    mem0 top-{limit}...")
        case_results = []
        result = run_mem0_query(mem0, session['answers'][start_idx], config, limit)
        case_results.append(result)
        results_by_limit[limit] = case_results

    return results_by_limit


def process_case_agent(agent: Agent, session: dict, config: dict, memory_path: str, start_idx=0) -> list:
    """Process a single case with mem-agent"""
    agent_config = {**config, 'memory_path': memory_path}
    case_results = []
    result = run_mem_agent_query(agent, session['answers'][start_idx], agent_config)
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

    # Load mem0 agent prompts (if any)
    mem0_agent_prompts = config['benchmark'].get('mem0_agent_prompts', [])
    mem0_agent_prompt_data = []
    for prompt_cfg in mem0_agent_prompts:
        with open(script_dir / prompt_cfg['loading_prompt'], 'r') as f:
            loading_prompt = f.read()
        with open(script_dir / prompt_cfg['query_prompt'], 'r') as f:
            query_prompt = f.read()
        mem0_agent_prompt_data.append({
            'name': prompt_cfg['name'],
            'loading_prompt': loading_prompt,
            'query_prompt': query_prompt,
        })

    # Allow limiting sessions and message checkpoints via config
    max_sessions = config['benchmark'].get('max_sessions', None)
    message_checkpoints = config['benchmark'].get('message_checkpoints', MESSAGE_NUM)

    sessions = case_data['sessions']
    if max_sessions:
        sessions = sessions[:max_sessions]

    session_id = 1
    all_results = [dict() for _ in range(len(message_checkpoints) - 1)]
    for session in sessions:
        print(len(session), session.keys())
        print(f"\n{'='*80}")
        print(f"CASE: {case_id} {session_id} ({data_path})")
        print(f"  mem0 limits: {mem0_limits}")
        print(f"  Agent prompts: {[p['name'] for p in prompt_configs]}")
        print(f"  mem0 agent prompts: {[p['name'] for p in mem0_agent_prompt_data]}")
        print("="*80)

        # Initialize mem0 (fresh for each case)
        mem0 = initialize_mem0(config['mem0'])
        for i in range(1, len(message_checkpoints)):
            load_user_messages_to_mem0(mem0, session, config['mem0'], message_checkpoints[i - 1], message_checkpoints[i])

            # Run mem0 queries at all limits
            mem0_results = process_case_mem0(mem0, session, config['mem0'], mem0_limits, i - 1)
            for limit, results in mem0_results.items():
                if f"mem0_top{limit}" in all_results[i - 1]:
                    all_results[i - 1][f"mem0_top{limit}"].append({
                        "session_id": session_id,
                        "results": results,
                        "message_num": message_checkpoints[i]
                    })
                else:
                    all_results[i - 1][f"mem0_top{limit}"] = [{
                        "session_id": session_id,
                        "results": results,
                        "message_num": message_checkpoints[i]
                    }]

        # Run mem0 agent with each prompt config (tool-calling approach)
        for prompt_data in mem0_agent_prompt_data:
            print(f"  mem0 agent: {prompt_data['name']}...")
            try:
                mem0_agent = initialize_mem0(config['mem0'])
                for i in range(1, len(message_checkpoints)):
                    load_user_messages_to_mem0_agent(
                        mem0_agent, session, config['mem0'],
                        prompt_data['loading_prompt'],
                        message_checkpoints[i - 1], message_checkpoints[i]
                    )
                    print(f"    mem0 agent query at M={message_checkpoints[i]}...")
                    agent_results = process_case_mem0_agent(
                        mem0_agent, session, config['mem0'],
                        prompt_data['query_prompt'], i - 1
                    )
                    config_name = prompt_data['name']
                    if config_name in all_results[i - 1]:
                        all_results[i - 1][config_name].append({
                            "session_id": session_id,
                            "results": agent_results,
                            "message_num": message_checkpoints[i]
                        })
                    else:
                        all_results[i - 1][config_name] = [{
                            "session_id": session_id,
                            "results": agent_results,
                            "message_num": message_checkpoints[i]
                        }]
            except Exception as e:
                print(f"  ✗ FAILED mem0 agent {prompt_data['name']} session {session_id}: {e}")
                continue

        # Run mem-agent with each prompt config
        for prompt_cfg in prompt_configs:
            print(f"  {prompt_cfg['name']}...")
            memory_path = f"{config['mem_agent']['memory_path']}/{session_id}_{prompt_cfg['name']}"
            agent = initialize_mem_agent(
                config['mem_agent'],
                str(script_dir / prompt_cfg['path']),
                memory_path
            )
            for i in range(1, len(message_checkpoints)):
                load_user_messages_to_agent(agent, session, config['benchmark']['verbose'], message_checkpoints[i - 1], message_checkpoints[i])
                agent_results = process_case_agent(agent, session, config['mem_agent'], memory_path, i - 1)
                if prompt_cfg['name'] in all_results[i - 1]:
                    all_results[i - 1][prompt_cfg['name']].append({
                        "session_id": session_id,
                        "prompt_path": prompt_cfg['path'],
                        "results": agent_results,
                        "message_num": message_checkpoints[i]
                    })
                else:
                    all_results[i - 1][prompt_cfg['name']] = [{
                        "session_id": session_id,
                        "prompt_path": prompt_cfg['path'],
                        "results": agent_results,
                        "message_num": message_checkpoints[i]
                    }]
        session_id += 1

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run memory benchmark')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Config file path (default: config.yaml)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    config = load_config(str(config_path))

    # Get configurations
    data_paths = config['benchmark'].get('data_paths', [])
    prompt_configs = config['benchmark'].get('agent_prompts', [
        {"name": "mem_agent", "path": "prompts/system_prompt.txt"}
    ])

    if not data_paths:
        # Fallback to old format
        data_paths = [c['data_path'] for c in config['benchmark'].get('cases', [])]

    message_checkpoints = config['benchmark'].get('message_checkpoints', MESSAGE_NUM)
    print(f"Running benchmark on {len(data_paths)} data file(s)")
    print(f"Message checkpoints: {message_checkpoints}")

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
        for i in range(len(message_checkpoints) - 1):
            for config_name, case_result in all_results[i].items():
                output = {
                    "benchmark_timestamp": timestamp,
                    "data_path": data_path,
                    "memory_type": config_name,
                    "message_num": message_checkpoints[i + 1],
                    "config": {
                        "model": config['mem0']['llm']['model'] if config_name.startswith('mem0') else config['mem_agent']['model'],
                    },
                    "cases": case_result
                }

                output_path = output_dir / f"results_{group_name}_{config_name}_M_{message_checkpoints[i + 1]}.json"
                save_results(output, str(output_path))
                print(f"✓ Saved {output_path}")

    print("\n" + "="*80)
    print("✓ BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
