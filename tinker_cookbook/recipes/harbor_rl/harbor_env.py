from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Sequence

import chz
import tinker
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

from tinker_cookbook.recipes.harbor_rl.tinker_llm import TinkerLLM

logger = logging.getLogger(__name__)

# Per-task lock to serialize Docker image builds (first trial builds, rest reuse)
_task_locks: dict[str, asyncio.Lock] = {}


async def run_harbor_group(
    task: Task,
    tinker_llm: "TinkerLLM",
    group_size: int,
    log_path: str,
    max_concurrent: int = 8,
) -> TrajectoryGroup:
    """Run a group of Harbor trials for a single task."""
    trials_dir = Path(log_path) / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)
    task_id = task.task_id

    # Get or create lock for this task (serializes Docker image build)
    if task_id not in _task_locks:
        _task_locks[task_id] = asyncio.Lock()
    task_lock = _task_locks[task_id]
    image_built = asyncio.Event()

    async def run_one(trial_idx: int, is_first: bool) -> TrialResult | None:
        async with semaphore:
            # First trial holds lock while building image, others wait
            if is_first:
                await task_lock.acquire()
                logger.info(f"Task {task_id}: starting first trial (building Docker image)...")
            else:
                await image_built.wait()

            try:
                trial_config = TrialConfig(
                    task=TaskConfig(path=task.task_dir),
                    trials_dir=trials_dir,
                    agent=AgentConfig(
                        name="terminus-2",
                        model_name=tinker_llm.model_name,
                        kwargs={
                            "llm": tinker_llm,
                            "collect_rollout_details": True,
                        },
                    ),
                    environment=EnvironmentConfig(type="docker", delete=True),
                )
                logger.debug(f"Task {task_id} trial {trial_idx}: running...")
                result = await Trial(trial_config).run()
                logger.debug(f"Task {task_id} trial {trial_idx}: completed")
                return result
            except Exception as e:
                logger.warning(f"Trial failed for {task_id}: {e}", exc_info=True)
                return None
            finally:
                if is_first:
                    image_built.set()  # Signal others that image is ready
                    task_lock.release()

    # Suppress verbose Harbor logs during trials
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("harbor"):
            logging.getLogger(name).setLevel(logging.WARNING)

    results = await asyncio.gather(*[run_one(trial_idx=i, is_first=(i == 0)) for i in range(group_size)])

    # Filter to valid results with detailed logging
    valid: list[TrialResult] = []
    for i, r in enumerate(results):
        if r is None:
            logger.warning(f"Task {task_id} trial {i}: returned None (exception)")
            continue
        if not r.agent_result:
            logger.warning(f"Task {task_id} trial {i}: no agent_result")
            continue
        if not r.agent_result.rollout_details:
            logger.warning(
                f"Task {task_id} trial {i}: no rollout_details. "
                f"agent_result keys: {list(vars(r.agent_result).keys())}"
            )
            continue
        # Check if rollout_details has actual data
        rd = r.agent_result.rollout_details[0]
        n_completions = len(rd.get("completion_token_ids", []))
        if n_completions == 0:
            logger.warning(f"Task {task_id} trial {i}: rollout_details has 0 completions")
            continue
        valid.append(r)

    if valid:
        rewards = [_extract_reward(r.verifier_result) for r in valid]
        logger.info(
            f"Task {task_id}: {len(valid)}/{group_size} valid, "
            f"avg_reward={sum(rewards)/len(rewards):.3f}"
        )
    else:
        logger.error(f"Task {task_id}: ALL {group_size} trials produced no valid training data!")

    return convert_trial_results_to_trajectory_group(valid)


def _extract_reward(verifier_result) -> float:
    """Extract reward from a VerifierResult."""
    if not verifier_result or not verifier_result.rewards:
        return 0.0
    reward = verifier_result.rewards.get("reward", 0.0)
    if isinstance(reward, (int, float)):
        return float(reward)
    return 1.0 if verifier_result.rewards.get("pass", False) else 0.0


# ---------------------------------------------------------------------------
# Trajectory conversion
# ---------------------------------------------------------------------------


def convert_trial_results_to_trajectory_group(results: list[TrialResult]) -> TrajectoryGroup:
    """Convert Harbor TrialResults to a TrajectoryGroup for training."""
    trajectories_G: list[Trajectory] = []
    final_rewards_G: list[float] = []
    metrics_G: list[dict[str, float | int]] = []

    for result in results:
        traj = _convert_trial_to_trajectory(result)
        if traj is None:
            traj = Trajectory(transitions=[], final_ob=tinker.ModelInput.empty())

        trajectories_G.append(traj)

        reward = _extract_reward(result.verifier_result)
        final_rewards_G.append(reward)

        metrics: dict[str, float | int] = {"success": 1 if reward > 0.5 else 0}
        if result.agent_result and result.agent_result.rollout_details:
            n_turns = len(result.agent_result.rollout_details[0].get("completion_token_ids", []))
            metrics["n_turns"] = n_turns
        metrics_G.append(metrics)

    return TrajectoryGroup(
        trajectories_G=trajectories_G,
        final_rewards_G=final_rewards_G,
        metrics_G=metrics_G,
    )


def _convert_trial_to_trajectory(result: TrialResult) -> Trajectory | None:
    """Convert a single TrialResult to a Trajectory."""
    if not result.agent_result or not result.agent_result.rollout_details:
        return None

    rd = result.agent_result.rollout_details[0] # Main agent rollout only
    prompt_tokens = rd.get("prompt_token_ids", [])
    completion_tokens = rd.get("completion_token_ids", [])
    logprobs = rd.get("logprobs", [])

    n_turns = len(completion_tokens)
    if n_turns == 0:
        return None

    transitions: list[Transition] = []
    for i in range(n_turns):
        if i >= len(prompt_tokens) or i >= len(logprobs):
            continue
        if not completion_tokens[i]:
            continue

        ob = tinker.ModelInput.from_ints(prompt_tokens[i])
        ac = TokensWithLogprobs(
            tokens=completion_tokens[i],
            maybe_logprobs=logprobs[i] if logprobs[i] else None,
        )
        transitions.append(
            Transition(ob=ob, ac=ac, reward=0.0, episode_done=(i == n_turns - 1), metrics={})
        )

    if not transitions:
        return None

    return Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())


# ---------------------------------------------------------------------------
# Dataset / EnvGroupBuilder
# ---------------------------------------------------------------------------


class HarborEnvGroupBuilder(EnvGroupBuilder):
    """Builder for a Harbor task group. Envs are created via run_harbor_group()."""

    def __init__(self, task: Task):
        self.task = task

    async def make_envs(self) -> Sequence[Env]:
        return []  # Rollouts handled by custom_do_group_rollout

    def logging_tags(self) -> list[str]:
        return ["harbor", self.task.task_id]


class HarborRLDataset(RLDataset):
    """Dataset of Harbor tasks."""

    def __init__(self, tasks: list[Task], groups_per_batch: int, shuffle: bool = True, seed: int = 42):
        self.tasks = tasks
        self.groups_per_batch = groups_per_batch
        self._indices = list(range(len(tasks)))
        if shuffle:
            random.Random(seed).shuffle(self._indices)

    def __len__(self) -> int:
        return (len(self._indices) + self.groups_per_batch - 1) // self.groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        end = min(start + self.groups_per_batch, len(self._indices))
        if start >= len(self._indices):
            raise IndexError(f"Batch index {index} out of range")
        return [HarborEnvGroupBuilder(self.tasks[self._indices[i]]) for i in range(start, end)]


@chz.chz
class HarborRLDatasetBuilder(RLDatasetBuilder):
    """Builder for Harbor RL datasets."""

    tasks_dir: str
    groups_per_batch: int = 32
    shuffle: bool = True
    seed: int = 42

    async def __call__(self) -> tuple[HarborRLDataset, None]:
        tasks_path = Path(self.tasks_dir)
        tasks: list[Task] = []

        for task_dir in sorted(tasks_path.iterdir()):
            if task_dir.is_dir() and (task_dir / "task.toml").exists():
                try:
                    tasks.append(Task(task_dir=task_dir))
                except Exception as e:
                    logger.warning(f"Failed to load task from {task_dir}: {e}")

        if not tasks:
            raise ValueError(f"No tasks found in {self.tasks_dir}")

        logger.info(f"Loaded {len(tasks)} tasks from {self.tasks_dir}")
        return HarborRLDataset(tasks, self.groups_per_batch, self.shuffle, self.seed), None
