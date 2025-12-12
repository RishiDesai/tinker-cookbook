# RL Training with Harbor Terminal Bench

[Harbor](https://github.com/abundant-ai/harbor) is a framework for evaluating and training AI agents on terminal-based tasks. It's the official format for TerminalBench 2.0.

## What is Terminal Bench?

Terminal Bench tasks present agents with coding challenges in sandboxed Docker environments. The agent interacts via a terminal (tmux session), executing shell commands to navigate codebases, edit files, run tests, and fix bugs. 


## Installation

1. **Install Harbor** (provides the Trial infrastructure and Terminus2 agent):

```bash
uv pip install https://github.com/laude-institute/harbor
```

2. **Ensure Docker is running**:

```bash
docker info
docker login
```

## Usage

Full run:

```bash
python -m tinker_cookbook.recipes.harbor_rl.train \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    tasks_dir=./terminal-bench-2/ \
    group_size=4 \
    groups_per_batch=8 \
    learning_rate=4e-5 \
    max_tokens=1024 \
    temperature=0.7 \
    environment_type=docker \
    n_parallel_envs=8 \
    wandb_project=my-project
```

## How It Works

This recipe uses the `custom_do_group_rollout` pattern (like `verifiers_rl`) to integrate Harbor's Trial infrastructure with Tinker's training loop:

1. **HarborEnvGroupBuilder** - Returns task metadata; actual rollouts happen via Harbor Trials
2. **TinkerLLM** - Adapts Tinker's SamplingClient to Harbor's LLM interface
3. **custom_do_group_rollout** - Runs Harbor Trials, extracts rollout_details (tokens + logprobs), converts to TrajectoryGroup
4. **Cookbook training loop** - Computes advantages, runs forward/backward, updates weights

The agent (Terminus2) handles:
- Terminal interaction via tmux
- JSON-formatted command execution
- Context summarization when context limit is reached
- Double-confirmation for task completion


<!-- ## Potential Footguns

- **Docker race conditions**: When running multiple trials for the same task in parallel, Docker image builds can conflict. The recipe uses per-task locks to serialize the first build, then subsequent trials run in parallel.

- **Empty trajectories**: If all trials in a batch fail (e.g., Docker issues), the cookbook's metrics computation may error. Check Harbor logs if you see `ZeroDivisionError`.

- **Slow Docker builds**: First run for each task builds the Docker image. Subsequent runs reuse cached images. Consider pre-building images for large task sets. -->
