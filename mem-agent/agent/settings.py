import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Agent settings
MAX_TOOL_TURNS = 8

# LLM defaults (overridden by api_key/base_url params when called from benchmark)
OPENROUTER_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_STRONG_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# vLLM
VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))

# Memory
MEMORY_PATH = "memory_dir"
FILE_SIZE_LIMIT = 1024 * 1024  # 1MB
DIR_SIZE_LIMIT = 1024 * 1024 * 10  # 10MB
MEMORY_SIZE_LIMIT = 1024 * 1024 * 100  # 100MB

# Engine
SANDBOX_TIMEOUT = 20

# Path settings
#SYSTEM_PROMPT_PATH = "agent/system_prompt.txt"
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.txt"
SAVE_CONVERSATION_PATH = "output/conversations/"