# RL Training with Tinker + Harbor

Train models on [Harbor](https://github.com/abundant-ai/harbor)-style terminal tasks using tinker_cookbook's RL infrastructure. Harbor is a framework for evaluating AI agents on terminal-use tasks, and is the official format for TerminalBench 2.0 and many other datasets.

Tinker handles the distributed RL training, while Harbor handles the agent harness, task orchestration, and rollouts.

## Installation

1. **Install Harbor**:
```bash
uv pip install https://github.com/laude-institute/harbor
```

2. **Ensure Docker is running**:
```bash
docker info
docker login
```

3. **Download tasks**:
```bash
git clone https://github.com/laude-institute/terminal-bench-2/
```

## Usage

```bash
python -m tinker_cookbook.recipes.harbor_rl.train \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    tasks_dir=./terminal-bench-2/ \
    group_size=4 \
    groups_per_batch=8 \
    learning_rate=4e-5 \
    max_tokens=1024 \
    temperature=0.7 \
    n_parallel_envs=8 \
```

## How It Works

This recipe uses the `custom_do_group_rollout` pattern (like `verifiers_rl`) to integrate Harbor's Trial infrastructure with the cookbook's training loop.
