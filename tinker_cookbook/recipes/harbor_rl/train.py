from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import cast

import chz
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborEnvGroupBuilder,
    HarborRLDatasetBuilder,
    run_harbor_group,
)
from tinker_cookbook.recipes.harbor_rl.tinker_llm import TinkerLLM
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32

    tasks_dir: str = "./tasks"

    group_size: int = 4
    groups_per_batch: int = 2
    learning_rate: float = 4e-5
    max_tokens: int = 1024

    log_path: str | None = None
    save_every: int = 20
    wandb_project: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    # Build log path
    model_tag = cli_config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"harbor_{model_tag}_{timestamp}"
    log_path = cli_config.log_path or f"/tmp/tinker-examples/harbor_rl/{run_name}"

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Shared state (initialized lazily in rollout function)
    shared_llm: TinkerLLM | None = None
    shared_tokenizer: Tokenizer | None = None
    shared_renderer: renderers.Renderer | None = None

    async def custom_do_group_rollout(
        builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> TrajectoryGroup:
        """Run Harbor trials for a task group."""
        nonlocal shared_llm, shared_tokenizer, shared_renderer

        # Lazy init
        if shared_tokenizer is None:
            shared_tokenizer = get_tokenizer(cli_config.model_name)
        if shared_renderer is None:
            renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
            shared_renderer = renderers.get_renderer(renderer_name, shared_tokenizer)

        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client

        if shared_llm is None:
            shared_llm = TinkerLLM(
                sampling_client=sampling_client,
                tokenizer=shared_tokenizer,
                renderer=shared_renderer,
                model_name=cli_config.model_name,
                max_tokens=cli_config.max_tokens,
            )
        else:
            shared_llm.update_sampling_client(sampling_client)

        harbor_builder = cast(HarborEnvGroupBuilder, builder)
        return await run_harbor_group(
            task=harbor_builder.task,
            tinker_llm=shared_llm,
            group_size=cli_config.group_size,
            log_path=log_path,
        )

    # Override rollout function
    train.do_group_rollout = custom_do_group_rollout

    # Build config and run
    cfg = train.Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        dataset_builder=HarborRLDatasetBuilder(
            tasks_dir=cli_config.tasks_dir,
            groups_per_batch=cli_config.groups_per_batch,
        ),
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        lora_rank=cli_config.lora_rank,
        wandb_project=cli_config.wandb_project,
        wandb_name=run_name,
        save_every=cli_config.save_every,
    )

    logger.info(f"Starting Harbor RL: {cli_config.model_name} on {cli_config.tasks_dir}")
    await train.main(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("harbor").setLevel(logging.WARNING)

    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
