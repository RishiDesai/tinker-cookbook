# RL Training with Tinker + Harbor

Train models on real-world agent benchmarks using [Harbor](https://github.com/laude-institute/harbor) for rollouts and Tinker for distributed RL training.

## Why Harbor?

Harbor is the official evaluation framework for [Terminal-Bench 2.0](https://github.com/laude-institute/terminal-bench-2) and supports many other third-party benchmarks and datasets.

1. **Install Harbor**:
```bash
uv pip install harbor
docker info # ensure docking is running
```

2. **Download tasks**:
```bash
git clone https://github.com/laude-institute/terminal-bench-2/
```

## Usage

```bash
python -m tinker_cookbook.recipes.harbor_rl.train \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    tasks_dir=./path/to/tasks/ \
    group_size=4 \
    tasks_per_batch=8
```

## How It Works

1. `HarborRLDatasetBuilder` loads tasks from a directory of Harbor task folders
2. `custom_do_group_rollout` replaces the standard rollout with Harbor's `Trial.run()`
3. `TinkerLLM` bridges Tinker's sampling client to Harbor's LLM interface
4. `convert_results_to_trajectory_group` extracts token IDs and logprobs from Harbor's ATIF trajectories
5. Tinker computes advantages and runs the training step as usual
