from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Sequence

import chz
import tinker
from harbor.models.task.task import Task
from harbor.models.trial.result import TrialResult
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

logger = logging.getLogger(__name__)


def extract_reward(verifier_result) -> float:
    """Extract reward from a VerifierResult."""
    if not verifier_result or not verifier_result.rewards:
        return 0.0
    reward_value = verifier_result.rewards.get("reward", 0.0)
    if isinstance(reward_value, (int, float)):
        return float(reward_value)
    pass_value = verifier_result.rewards.get("pass", False)
    return 1.0 if pass_value else 0.0


def convert_trial_result_to_trajectory(result: TrialResult) -> Trajectory | None:
    """Convert a single TrialResult to a Trajectory."""
    if not result.agent_result or not result.agent_result.rollout_details:
        return None

    rd = result.agent_result.rollout_details[0]  # Main agent rollout only
    prompt_tokens = rd.get("prompt_token_ids", [])
    completion_tokens = rd.get("completion_token_ids", [])
    logprobs = rd.get("logprobs", [])

    n_turns = len(completion_tokens)
    if n_turns == 0:
        return None

    transitions: list[Transition] = []

    for turn_idx in range(n_turns):
        if turn_idx >= len(prompt_tokens) or turn_idx >= len(logprobs):
            continue

        turn_prompt = prompt_tokens[turn_idx]
        turn_completion = completion_tokens[turn_idx]
        turn_logprobs = logprobs[turn_idx]

        if not turn_completion:
            continue

        # Create observation from prompt tokens
        ob = tinker.ModelInput.from_ints(turn_prompt)

        # Create action with completion tokens and logprobs
        ac = TokensWithLogprobs(
            tokens=turn_completion,
            maybe_logprobs=turn_logprobs if turn_logprobs else None,
        )

        is_last = turn_idx == n_turns - 1
        transition = Transition(
            ob=ob,
            ac=ac,
            reward=0.0,  # Reward comes from final group reward
            episode_done=is_last,
            metrics={},
        )
        transitions.append(transition)

    if not transitions:
        return None

    return Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.empty(),
    )


def convert_trial_results_to_trajectory_group(
    results: list[TrialResult],
) -> TrajectoryGroup:
    """
    Convert a group of TrialResults to a TrajectoryGroup.

    This is the main conversion function used by custom_do_group_rollout.
    """
    trajectories_G: list[Trajectory] = []
    final_rewards_G: list[float] = []
    metrics_G: list[dict[str, float | int]] = []

    for result in results:
        trajectory = convert_trial_result_to_trajectory(result)
        if trajectory is None:
            # Create empty trajectory for failed/invalid results
            trajectory = Trajectory(transitions=[], final_ob=tinker.ModelInput.empty())

        trajectories_G.append(trajectory)

        # Extract reward and metrics
        reward = extract_reward(result.verifier_result)
        final_rewards_G.append(reward)

        metrics: dict[str, float | int] = {
            "success": 1 if reward > 0.5 else 0,
        }
        if result.agent_result and result.agent_result.rollout_details:
            n_turns = len(result.agent_result.rollout_details[0].get("completion_token_ids", []))
            metrics["n_turns"] = n_turns
        metrics_G.append(metrics)

    return TrajectoryGroup(
        trajectories_G=trajectories_G,
        final_rewards_G=final_rewards_G,
        metrics_G=metrics_G,
    )


class HarborEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for Harbor task environment groups.

    Note: make_envs() returns [] because we use custom_do_group_rollout
    to run Harbor Trials instead of step-by-step Env instances.
    """

    def __init__(self, task: Task):
        self.task = task

    async def make_envs(self) -> Sequence[Env]:
        # Not used - rollouts are handled by custom_do_group_rollout
        return []

    def logging_tags(self) -> list[str]:
        return ["harbor", self.task.task_id]


class HarborRLDataset(RLDataset):
    """Dataset of Harbor tasks for RL training."""

    def __init__(
        self,
        tasks: list[Task],
        groups_per_batch: int,
        shuffle: bool = True,
        seed: int = 42,
        num_epochs: int = 1,
    ):
        self.tasks = tasks
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.num_epochs = num_epochs

        self._task_indices = self._build_task_indices()

    def _build_task_indices(self) -> list[int]:
        indices = list(range(len(self.tasks))) * self.num_epochs

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        return indices

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return (len(self._task_indices) + self.groups_per_batch - 1) // self.groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        start = index * self.groups_per_batch
        end = min(start + self.groups_per_batch, len(self._task_indices))

        if start >= len(self._task_indices):
            raise IndexError(f"Batch index {index} out of range")

        builders: list[EnvGroupBuilder] = []
        for i in range(start, end):
            task_idx = self._task_indices[i]
            task = self.tasks[task_idx]
            builders.append(HarborEnvGroupBuilder(task=task))

        return builders


@chz.chz
class HarborRLDatasetBuilder(RLDatasetBuilder):
    """Builder for Harbor RL datasets."""

    tasks_dir: str
    groups_per_batch: int = 32
    shuffle: bool = True
    seed: int = 42
    num_epochs: int = 1

    async def __call__(self) -> tuple[HarborRLDataset, None]:
        """Build the training dataset."""
        tasks_path = Path(self.tasks_dir)

        all_tasks: list[Task] = []
        for task_dir in sorted(tasks_path.iterdir()):
            if task_dir.is_dir() and (task_dir / "task.toml").exists():
                try:
                    all_tasks.append(Task(task_dir=task_dir))
                except Exception as e:
                    logger.warning(f"Failed to load task from {task_dir}: {e}")

        if not all_tasks:
            raise ValueError(f"No tasks found in {self.tasks_dir}")

        logger.info(f"Loaded {len(all_tasks)} tasks from {self.tasks_dir}")

        dataset = HarborRLDataset(
            tasks=all_tasks,
            groups_per_batch=self.groups_per_batch,
            shuffle=self.shuffle,
            seed=self.seed,
            num_epochs=self.num_epochs,
        )

        return dataset, None

