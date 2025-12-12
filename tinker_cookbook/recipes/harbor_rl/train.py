from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import chz
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborEnvGroupBuilder,
    HarborRLDatasetBuilder,
    convert_trial_results_to_trajectory_group,
)
from tinker_cookbook.recipes.harbor_rl.tinker_llm import TinkerLLM
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI configuration for Harbor RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Task configuration
    tasks_dir: str = "./tasks"
    num_epochs: int = 1

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 32
    learning_rate: float = 4e-5
    num_substeps: int = 1
    kl_penalty_coef: float = 0.0

    # Generation parameters
    max_tokens: int = 1024
    temperature: float = 0.7
    context_limit: int = 32000

    # Agent configuration
    max_turns: int | None = None
    enable_summarize: bool = True
    proactive_summarization_threshold: int = 8000

    # Environment configuration
    environment_type: Literal["docker", "daytona", "modal", "e2b", "runloop"] = "docker"
    environment_kwargs: dict[str, Any] = chz.field(default_factory=dict)
    n_parallel_envs: int = 8
    trial_timeout_sec: float | None = None

    # Logging configuration
    eval_every: int = 0
    save_every: int = 20
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    seed: int = 67

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _suppress_harbor_logs() -> None:
    """Suppress verbose Harbor DEBUG logs."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("harbor"):
            harbor_logger = logging.getLogger(name)
            harbor_logger.setLevel(logging.WARNING)
            harbor_logger.propagate = False


async def cli_main(cli_config: CLIConfig):
    """Main entry point for Harbor RL training."""

    # Build run name
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"harbor_rl_{model_name_short}_gp{cli_config.groups_per_batch}_"
        f"gs{cli_config.group_size}_lr{cli_config.learning_rate}_"
        f"rank{cli_config.lora_rank}_{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/harbor_rl/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Shared state captured by closure (initialized lazily)
    shared_tinker_llm: TinkerLLM | None = None
    shared_tokenizer: Tokenizer | None = None
    shared_renderer: renderers.Renderer | None = None
    semaphore = asyncio.Semaphore(cli_config.n_parallel_envs)

    # Per-task locks to serialize Docker image builds for the same task.
    # Different tasks still run fully in parallel.
    # After first trial builds the image, subsequent trials reuse it without waiting.
    task_build_locks: dict[str, asyncio.Lock] = {}
    task_image_ready: set[str] = set()  # Tasks whose images are already built

    def create_trial_config(task: Task, tinker_llm: TinkerLLM) -> TrialConfig:
        """Create TrialConfig for a task, injecting TinkerLLM."""
        env_type = EnvironmentType(cli_config.environment_type)

        trials_dir = Path(log_path) / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)

        return TrialConfig(
            task=TaskConfig(path=task.task_dir),
            trials_dir=trials_dir,
            agent=AgentConfig(
                name="terminus-2",
                model_name=cli_config.model_name,
                kwargs={
                    "llm": tinker_llm,
                    "collect_rollout_details": True,
                    "enable_summarize": cli_config.enable_summarize,
                    "proactive_summarization_threshold": cli_config.proactive_summarization_threshold,
                    "max_turns": cli_config.max_turns,
                },
            ),
            environment=EnvironmentConfig(
                type=env_type,
                delete=True,
                kwargs=cli_config.environment_kwargs,
            ),
        )

    async def run_single_trial(task: Task, tinker_llm: TinkerLLM) -> TrialResult | None:
        """Run a single Harbor trial with semaphore limiting and per-task build serialization."""
        task_id = task.task_id

        # Get or create lock for this task
        if task_id not in task_build_locks:
            task_build_locks[task_id] = asyncio.Lock()
        task_lock = task_build_locks[task_id]

        async with semaphore:
            # If image not yet built for this task, serialize to avoid Docker race
            need_lock = task_id not in task_image_ready

            if need_lock:
                await task_lock.acquire()

            try:
                trial = Trial(create_trial_config(task, tinker_llm))

                if cli_config.trial_timeout_sec is not None:
                    result = await asyncio.wait_for(
                        trial.run(), timeout=cli_config.trial_timeout_sec
                    )
                else:
                    result = await trial.run()

                # Mark image as ready after successful trial
                task_image_ready.add(task_id)
                return result

            except asyncio.TimeoutError:
                logger.warning(f"Trial timed out for {task.task_id}")
                return None
            except Exception as e:
                logger.warning(f"Trial failed for {task.task_id}: {e}")
                # Still mark as ready - the image was likely built even if trial failed
                task_image_ready.add(task_id)
                return None
            finally:
                if need_lock and task_lock.locked():
                    task_lock.release()

    async def custom_do_group_rollout(
        builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> TrajectoryGroup:
        """
        Custom rollout function that runs Harbor Trials.

        This replaces the default do_group_rollout to use Harbor's Trial
        infrastructure instead of step-by-step Env instances.
        """
        nonlocal shared_tinker_llm, shared_tokenizer, shared_renderer

        # Initialize tokenizer and renderer lazily
        if shared_tokenizer is None:
            shared_tokenizer = get_tokenizer(cli_config.model_name)

        if shared_renderer is None:
            renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
                cli_config.model_name
            )
            shared_renderer = renderers.get_renderer(renderer_name, shared_tokenizer)

        # Get the Tinker sampling client from the policy
        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client

        # Create or update TinkerLLM
        if shared_tinker_llm is None:
            shared_tinker_llm = TinkerLLM(
                sampling_client=sampling_client,
                tokenizer=shared_tokenizer,
                renderer=shared_renderer,
                model_name=cli_config.model_name,
                max_tokens=cli_config.max_tokens,
                temperature=cli_config.temperature,
                context_limit=cli_config.context_limit,
            )
        else:
            shared_tinker_llm.update_sampling_client(sampling_client)

        # Get the task from the builder
        harbor_builder = cast(HarborEnvGroupBuilder, builder)
        task = harbor_builder.task

        logger.info(f"Running {cli_config.group_size} trials for task: {task.task_id}")

        # Suppress verbose Harbor logs
        _suppress_harbor_logs()

        # Run group_size trials in parallel
        results = await asyncio.gather(
            *[run_single_trial(task, shared_tinker_llm) for _ in range(cli_config.group_size)],
            return_exceptions=True,
        )

        # Filter valid results
        valid_results: list[TrialResult] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Trial exception for {task.task_id}: {r}")
            elif r is None:
                pass  # Already logged
            elif not r.agent_result:
                logger.warning(f"No agent_result for {task.task_id}")
            elif not r.agent_result.rollout_details:
                logger.warning(
                    f"No rollout_details for {task.task_id}. "
                    f"agent_result keys: {list(r.agent_result.__dict__.keys()) if hasattr(r.agent_result, '__dict__') else 'N/A'}"
                )
            else:
                valid_results.append(r)

        # Log summary
        if valid_results:
            rewards = [
                r.verifier_result.rewards.get("reward", 0.0)
                if r.verifier_result and r.verifier_result.rewards
                else 0.0
                for r in valid_results
            ]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            logger.info(
                f"Task {task.task_id}: {len(valid_results)}/{cli_config.group_size} valid, "
                f"avg_reward={avg_reward:.3f}"
            )

        # Convert to TrajectoryGroup
        trajectory_group = convert_trial_results_to_trajectory_group(valid_results)

        # Warn if no valid trajectories (can cause downstream errors)
        if not trajectory_group.trajectories_G or all(
            len(t.transitions) == 0 for t in trajectory_group.trajectories_G
        ):
            logger.warning(
                f"No valid trajectories for task {task.task_id}. "
                "This group will produce no training data."
            )

        return trajectory_group

    # Create dataset builder
    dataset_builder = HarborRLDatasetBuilder(
        tasks_dir=cli_config.tasks_dir,
        groups_per_batch=cli_config.groups_per_batch,
        shuffle=True,
        seed=cli_config.seed,
        num_epochs=cli_config.num_epochs,
    )

    # Override do_group_rollout with our custom Harbor implementation
    train.do_group_rollout = custom_do_group_rollout

    # Build training config
    cfg = train.Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        lora_rank=cli_config.lora_rank,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
    )

    logger.info("=" * 60)
    logger.info("Starting Harbor RL Training")
    logger.info("=" * 60)
    logger.info(f"Model: {cli_config.model_name}")
    logger.info(f"Tasks: {cli_config.tasks_dir}")
    logger.info(f"Environment: {cli_config.environment_type}")
    logger.info(f"Groups per batch: {cli_config.groups_per_batch}")
    logger.info(f"Group size: {cli_config.group_size}")
    logger.info(f"Learning rate: {cli_config.learning_rate}")
    logger.info("=" * 60)

    # Run training using cookbook's main loop
    await train.main(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress verbose library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("harbor").setLevel(logging.WARNING)

    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
