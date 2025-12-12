"""
Harbor RL Recipe - Train agents on Harbor terminal bench tasks.

This recipe uses Harbor's Trial infrastructure directly, avoiding reimplementation
of the agent harness. It follows the verifiers_rl pattern of custom_do_group_rollout.

Usage:
    python -m tinker_cookbook.recipes.harbor_rl.train \
        model_name=Qwen/Qwen3-4B \
        tasks_dir=./tasks
"""

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborEnvGroupBuilder,
    HarborRLDataset,
    HarborRLDatasetBuilder,
)
from tinker_cookbook.recipes.harbor_rl.tinker_llm import TinkerLLM

__all__ = [
    "HarborEnvGroupBuilder",
    "HarborRLDataset",
    "HarborRLDatasetBuilder",
    "TinkerLLM",
]

