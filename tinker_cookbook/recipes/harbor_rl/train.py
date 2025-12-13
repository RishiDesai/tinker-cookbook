from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import cast

import chz

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .harbor_env import (
    HarborEnvGroupBuilder,
    HarborRLDatasetBuilder,
    convert_results_to_trajectory_group,
    run_harbor_trials,
)
from .tinker_llm import TinkerLLM

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

    # Initialize tokenizer/renderer once at startup
    tokenizer = get_tokenizer(cli_config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    trials_dir = Path(cli_config.log_path) / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # TinkerLLM instance - sampling_client updated each rollout
    tinker_llm: TinkerLLM | None = None

    async def custom_do_group_rollout(
        builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> TrajectoryGroup | None:
        """Run Harbor trials instead of standard Env.step() rollouts."""
        nonlocal tinker_llm

        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
        harbor_builder = cast(HarborEnvGroupBuilder, builder)

        # Create or update TinkerLLM with current sampling client
        if tinker_llm is None:
            tinker_llm = TinkerLLM(
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                renderer=renderer,
                model_name=cli_config.model_name,
                max_tokens=cli_config.max_tokens,
                temperature=cli_config.temperature,
                context_limit=cli_config.context_limit,
            )
        else:
            tinker_llm.update_sampling_client(sampling_client)

        results = await run_harbor_trials(
            task=harbor_builder.task,
            tinker_llm=tinker_llm,
            group_size=cli_config.group_size,
            trials_dir=trials_dir,
            max_concurrent=cli_config.max_concurrent_trials,
        )

        if results:
            rewards = [
                r.verifier_result.rewards.get("reward", 0)
                if r.verifier_result and r.verifier_result.rewards else 0
                for r in results
            ]
            logger.info(f"Task {harbor_builder.task.task_id}: {len(results)} trials, avg={sum(rewards)/len(rewards):.3f}")
        else:
            logger.warning(f"Task {harbor_builder.task.task_id}: all trials failed")
            return TrajectoryGroup(trajectories_G=[], final_rewards_G=[], metrics_G=[])

        traj_group = convert_results_to_trajectory_group(results)

        # Skip if no valid transitions were collected
        if not any(t.transitions for t in traj_group.trajectories_G):
            logger.warning(f"Task {harbor_builder.task.task_id}: no transitions collected, skipping")
            return TrajectoryGroup(trajectories_G=[], final_rewards_G=[], metrics_G=[])

        return traj_group

    # Override cookbook's rollout with Harbor implementation
    train.do_group_rollout = custom_do_group_rollout

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
