"""Config loading and dataset resolution for the benchmark runner."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Dataset:
    """Resolved dataset from a data directory's dataset.yaml."""
    dir_path: Path
    collection_name: str
    user_id: str
    # Resolved prompts (None if not defined in dataset.yaml):
    mem0_agent_loading_prompt: Optional[str] = None   # file content
    mem0_agent_query_prompt: Optional[str] = None     # file content
    mem_agent_system_prompt: Optional[str] = None     # absolute path (Agent needs path, not content)
    case_files: list[Path] = field(default_factory=list)


@dataclass
class Experiment:
    """Resolved experiment configuration."""
    name: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    data_dirs: list[str] = field(default_factory=list)
    run: dict = field(default_factory=dict)


def load_config(config_path: str) -> dict:
    """Load YAML config with environment variable substitution."""
    with open(config_path, 'r') as f:
        config_str = f.read()
    config_str = os.path.expandvars(config_str)
    return yaml.safe_load(config_str)


def resolve_dataset(data_dir: Path, max_cases: Optional[int] = None) -> Dataset:
    """Load dataset.yaml from a data directory and resolve all paths.

    Args:
        data_dir: Absolute path to data directory containing dataset.yaml
        max_cases: If set, only take the first N JSON files (sorted alphabetically)

    Returns:
        Resolved Dataset with absolute prompt paths and case file list
    """
    dataset_yaml = data_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"No dataset.yaml found in {data_dir}")

    with open(dataset_yaml, 'r') as f:
        raw = yaml.safe_load(f.read())

    # Resolve prompts
    prompts = raw.get('prompts', {})

    mem0_agent_loading_prompt = None
    mem0_agent_query_prompt = None
    if 'mem0_agent' in prompts:
        loading_path = data_dir / prompts['mem0_agent']['loading']
        query_path = data_dir / prompts['mem0_agent']['query']
        with open(loading_path, 'r') as f:
            mem0_agent_loading_prompt = f.read()
        with open(query_path, 'r') as f:
            mem0_agent_query_prompt = f.read()

    mem_agent_system_prompt = None
    if 'mem_agent' in prompts:
        mem_agent_system_prompt = str(data_dir / prompts['mem_agent']['system_prompt'])

    # Discover case JSON files (sorted alphabetically)
    case_files = sorted(data_dir.glob("*.json"))
    if max_cases is not None:
        case_files = case_files[:max_cases]

    return Dataset(
        dir_path=data_dir,
        collection_name=raw['collection_name'],
        user_id=raw['user_id'],
        mem0_agent_loading_prompt=mem0_agent_loading_prompt,
        mem0_agent_query_prompt=mem0_agent_query_prompt,
        mem_agent_system_prompt=mem_agent_system_prompt,
        case_files=case_files,
    )


def resolve_experiments(config: dict) -> list[Experiment]:
    """Parse experiments from config, merging with memory system defaults.

    If an experiment defines model/api_key/base_url, those override the defaults
    from mem0.llm / mem_agent. If not, the defaults are used.
    """
    # Defaults from memory system configs
    default_model = config['mem0']['llm']['model']
    default_api_key = config['mem0']['llm']['api_key']
    default_base_url = config['mem0']['llm'].get('base_url')

    experiments = []
    for exp_raw in config.get('experiments', []):
        experiments.append(Experiment(
            name=exp_raw['name'],
            model=exp_raw.get('model', default_model),
            api_key=exp_raw.get('api_key', default_api_key),
            base_url=exp_raw.get('base_url', default_base_url),
            data_dirs=exp_raw.get('data_dirs', []),
            run=exp_raw.get('run', {}),
        ))

    return experiments
