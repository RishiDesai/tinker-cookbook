from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import chz
import tinker
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .tinker_llm import TinkerLLM

logger = logging.getLogger(__name__)


async def run_harbor_trials(
    task: Task,
    tinker_llm: TinkerLLM,
    group_size: int,
    trials_dir: Path,
    max_concurrent: int = 8,
) -> list[TrialResult]:
    """Run multiple Harbor trials for a single task."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(idx: int) -> TrialResult | None:
        async with semaphore:
            try:
                config = TrialConfig(
                    task=TaskConfig(path=task.task_dir),
                    trials_dir=trials_dir,
                    agent=AgentConfig(
                        name="terminus-2",
                        model_name=tinker_llm.model_name,
                        kwargs={"llm": tinker_llm, "collect_rollout_details": True},
                    ),
                    environment=EnvironmentConfig(type="docker", delete=True),
                )
                return await Trial(config).run()
            except Exception as e:
                logger.warning(f"Trial {idx} failed for {task.task_id}: {e}")
                return None

    results = await asyncio.gather(*[run_one(i) for i in range(group_size)])
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Trajectory Conversion
# ---------------------------------------------------------------------------


def extract_reward(verifier_result: Any) -> float:
    """Extract reward from Harbor's VerifierResult."""
    if not verifier_result or not verifier_result.rewards:
        return 0.0
    reward = verifier_result.rewards.get("reward", 0.0)
    if isinstance(reward, (int, float)):
        return float(reward)
    return 1.0 if verifier_result.rewards.get("pass", False) else 0.0


def convert_results_to_trajectory_group(results: list[TrialResult]) -> TrajectoryGroup:
    """Convert Harbor TrialResults to tinker TrajectoryGroup."""
    if not results:
        return TrajectoryGroup(trajectories_G=[], final_rewards_G=[], metrics_G=[])

    trajectories_G: list[Trajectory] = []
    final_rewards_G: list[float] = []
    metrics_G: list[dict[str, float | int]] = []

    for result in results:
        transitions: list[Transition] = []

        if result.agent_result and result.agent_result.rollout_details:
            rd = result.agent_result.rollout_details[0]  # Main agent only
            prompt_tokens = rd.get("prompt_token_ids", [])
            completion_tokens = rd.get("completion_token_ids", [])
            logprobs = rd.get("logprobs", [])

            for turn_idx in range(len(completion_tokens)):
                if turn_idx >= len(prompt_tokens) or turn_idx >= len(logprobs):
                    continue
                if not completion_tokens[turn_idx] or not logprobs[turn_idx]:
                    continue

                ob = tinker.ModelInput.from_ints(prompt_tokens[turn_idx])
                ac = TokensWithLogprobs(
                    tokens=completion_tokens[turn_idx],
                    maybe_logprobs=logprobs[turn_idx],
                )
                transitions.append(Transition(
                    ob=ob,
                    ac=ac,
                    reward=0.0,  # Reward at episode end
                    episode_done=(turn_idx == len(completion_tokens) - 1),
                    metrics={},
                ))

        trajectory = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
        trajectories_G.append(trajectory)

        reward = extract_reward(result.verifier_result)
        final_rewards_G.append(reward)

        n_turns = len(transitions)
        metrics_G.append({"success": int(reward > 0.5), "n_turns": n_turns})

    return TrajectoryGroup(
        trajectories_G=trajectories_G,
        final_rewards_G=final_rewards_G,
        metrics_G=metrics_G,
    )


# ---------------------------------------------------------------------------
# Harbor Rollout Handler Factory
# ---------------------------------------------------------------------------


def create_harbor_rollout_handler(
    model_name: str,
    group_size: int,
    trials_dir: Path,
    max_concurrent_trials: int,
    max_tokens: int,
    temperature: float,
    context_limit: int,
) -> Callable[[EnvGroupBuilder, TokenCompleter], TrajectoryGroup | None]:
    """Factory that creates a Harbor-specific rollout function."""
    # Lazy initialization - will be set on first rollout
    tokenizer = None
    renderer = None
    tinker_llm: TinkerLLM | None = None

    async def do_harbor_rollout(
        builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> TrajectoryGroup | None:
        """Run Harbor trials instead of standard Env.step() rollouts."""
        nonlocal tokenizer, renderer, tinker_llm

        # Lazy initialize tokenizer and renderer on first call
        if tokenizer is None:
            tokenizer = get_tokenizer(model_name)
            renderer_name = model_info.get_recommended_renderer_name(model_name)
            renderer = renderers.get_renderer(renderer_name, tokenizer)
            logger.info(f"Initialized tokenizer and renderer: {renderer_name}")

        # Get current sampling client from policy
        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
        harbor_builder = cast(HarborEnvGroupBuilder, builder)

        # Create or update TinkerLLM with current sampling client
        if tinker_llm is None:
            tinker_llm = TinkerLLM(
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                renderer=renderer,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                context_limit=context_limit,
            )
            logger.info("Created TinkerLLM instance")
        else:
            tinker_llm.update_sampling_client(sampling_client)

        # Run Harbor trials
        results = await run_harbor_trials(
            task=harbor_builder.task,
            tinker_llm=tinker_llm,
            group_size=group_size,
            trials_dir=trials_dir,
            max_concurrent=max_concurrent_trials,
        )

        # Log results
        if results:
            rewards = [
                r.verifier_result.rewards.get("reward", 0)
                if r.verifier_result and r.verifier_result.rewards else 0
                for r in results
            ]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            logger.info(
                f"Task {harbor_builder.task.task_id}: "
                f"{len(results)} trials, avg_reward={avg_reward:.3f}"
            )
        else:
            logger.warning(f"Task {harbor_builder.task.task_id}: all trials failed")
            return TrajectoryGroup(trajectories_G=[], final_rewards_G=[], metrics_G=[])

        # Convert Harbor results to Tinker trajectory format
        traj_group = convert_results_to_trajectory_group(results)

        # Skip if no valid transitions were collected
        if not any(t.transitions for t in traj_group.trajectories_G):
            logger.warning(
                f"Task {harbor_builder.task.task_id}: no transitions collected, skipping"
            )
            return TrajectoryGroup(trajectories_G=[], final_rewards_G=[], metrics_G=[])

        return traj_group

    return do_harbor_rollout


# ---------------------------------------------------------------------------
# Dataset / EnvGroupBuilder
# ---------------------------------------------------------------------------


class HarborEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder for a Harbor task."""

    def __init__(self, task: Task):
        self.task = task

    async def make_envs(self) -> Sequence[Env]:
        return []  # Rollouts via custom_do_group_rollout

    def logging_tags(self) -> list[str]:
        return ["harbor", self.task.task_id]


class HarborRLDataset(RLDataset):
    """Dataset of Harbor tasks."""

    def __init__(self, tasks: list[Task], tasks_per_batch: int):
        self.tasks = tasks
        self.tasks_per_batch = tasks_per_batch

    def __len__(self) -> int:
        return (len(self.tasks) + self.tasks_per_batch - 1) // self.tasks_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.tasks_per_batch
        end = min(start + self.tasks_per_batch, len(self.tasks))
        return [HarborEnvGroupBuilder(self.tasks[i]) for i in range(start, end)]


@chz.chz
class HarborRLDatasetBuilder(RLDatasetBuilder):
    """Builder for Harbor RL datasets."""

    tasks_dir: str
    tasks_per_batch: int = 8

    async def __call__(self) -> tuple[HarborRLDataset, None]:
        tasks = _load_tasks(self.tasks_dir)
        if not tasks:
            raise ValueError(f"No tasks found in {self.tasks_dir}")
        logger.info(f"Loaded {len(tasks)} Harbor tasks from {self.tasks_dir}")
        return HarborRLDataset(tasks, self.tasks_per_batch), None


def _load_tasks(tasks_dir: str) -> list[Task]:
    """Load all Harbor tasks from a directory."""
    tasks_path = Path(tasks_dir)
    tasks = []
    for task_dir in sorted(tasks_path.iterdir()):
        if task_dir.is_dir() and (task_dir / "task.toml").exists():
            try:
                tasks.append(Task(task_dir=task_dir))
            except Exception as e:
                logger.warning(f"Failed to load task from {task_dir}: {e}")
    return tasks
