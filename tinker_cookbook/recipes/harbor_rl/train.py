from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train

from .harbor_env import (
    HarborRLDatasetBuilder,
    create_harbor_rollout_handler,
)

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    tasks_dir: str
    log_path: str = "/tmp/tinker-examples/harbor_rl"

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32

    group_size: int = 4
    tasks_per_batch: int = 8
    learning_rate: float = 4e-5
    num_substeps: int = 1
    kl_penalty_coef: float = 0.0

    # Generation
    max_tokens: int = 1024
    temperature: float = 0.7
    context_limit: int = 32000

    # Parallelism
    max_concurrent_trials: int = 8

    # Logging
    save_every: int = 20
    eval_every: int = 0
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Main entry point for Harbor RL training."""
    cli_utils.check_log_dir(
        cli_config.log_path,
        behavior_if_exists=cli_config.behavior_if_log_dir_exists,
    )

    # Suppress verbose Harbor logs
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("harbor"):
            logging.getLogger(name).setLevel(logging.WARNING)

    # Setup trials directory
    trials_dir = Path(cli_config.log_path) / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Create Harbor-specific rollout handler and override the default
    train.do_group_rollout = create_harbor_rollout_handler(
        model_name=cli_config.model_name,
        group_size=cli_config.group_size,
        trials_dir=trials_dir,
        max_concurrent_trials=cli_config.max_concurrent_trials,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        context_limit=cli_config.context_limit,
    )

    # Configure and run training
    config = train.Config(
        model_name=cli_config.model_name,
        log_path=cli_config.log_path,
        dataset_builder=HarborRLDatasetBuilder(
            tasks_dir=cli_config.tasks_dir,
            tasks_per_batch=cli_config.tasks_per_batch,
        ),
        learning_rate=cli_config.learning_rate,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        num_substeps=cli_config.num_substeps,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        remove_constant_reward_groups=True,
    )

    await train.main(config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
