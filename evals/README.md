# Toolshed Evaluation Framework

Evaluation framework for benchmarking LLMs with Toolshed vision tools on spatial reasoning datasets. Used to collect empirical results in the SpaceTools paper.

> **Note:** This codebase is research-grade and may contain bugs.

## Quick Start

```bash
# Activate environment
conda activate toolshed

# Evaluate with tools (requires GPU for vision tools)
python -m toolshed.evals.run \
    --dataset data/robospatial_home/test.parquet \
    --dataset-type robospatial \
    --provider bedrock \
    --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --use-tools \
    --output results/my_eval

# Evaluate without tools (baseline)
python -m toolshed.evals.run \
    --dataset data/robospatial_home/test.parquet \
    --dataset-type robospatial \
    --provider openai \
    --model gpt-5 \
    --output results/baseline
```

## Supported Datasets

| Dataset Type | Description |
|--------------|-------------|
| `robospatial` | RoboSpatial home environment spatial reasoning |
| `refspatial` | RefSpatial benchmark (placement, location, unseen) |
| `spatialbench` | SpatialBench positional reasoning |
| `blink` | BLINK relative depth estimation |
| `cvbench` | CVBench 3D depth and 2D relation tasks |
| `bop_ask` | BOP-ASK pose estimation and grasp prediction |

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | Path to parquet dataset file |
| `--dataset-type` | One of the supported dataset types |
| `--provider` | LLM provider: `openai`, `anthropic`, `bedrock`, `llm_gateway_openai` |
| `--model` | Model name (e.g., `gpt-5`, `claude-sonnet-4-20250514`) |
| `--use-tools` | Enable Toolshed vision tools (requires GPU) |
| `--output` | Output directory for results and checkpoints |
| `--save-conversations` | Save full conversation logs to `convos/` |
| `--save-training-data` | Save ShareGPT-format data to `training_data/` |
| `--resume` | Resume from checkpoint if interrupted |
| `--num-workers` | Parallel conversation threads (increases tool utilization) |
| `--enable-tools` | Comma-separated list of tools to enable |
| `--no-system-prompt` | Disable system prompt (for baseline comparisons) |

## Examples

### Basic evaluation with tools
```bash
python -m toolshed.evals.run \
    --dataset data/spatialbench_positional/test.parquet \
    --dataset-type spatialbench \
    --provider openai --model gpt-5 \
    --use-tools \
    --enable-tools roborefer,sam2,depth_estimator,vision_ops \
    --output results/spatialbench-gpt5
```

### Baseline without tools
```bash
python -m toolshed.evals.run \
    --dataset data/blink_relative_depth/val.parquet \
    --dataset-type blink \
    --provider anthropic --model claude-sonnet-4-20250514 \
    --no-system-prompt \
    --output results/blink-baseline
```

### Parallel evaluation for speed
```bash
python -m toolshed.evals.run \
    --dataset data/robospatial_home/test.parquet \
    --dataset-type robospatial \
    --provider bedrock --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --use-tools \
    --num-workers 50 \
    --save-conversations --save-training-data \
    --resume \
    --output results/robospatial-claude
```

## Output Structure

```
results/my_eval/
├── checkpoint.json      # Results and metrics (also serves as checkpoint)
├── convos/              # Full conversation dumps (if --save-conversations)
└── training_data/       # ShareGPT format data (if --save-training-data)
```

## Architecture

- `run.py` — CLI entry point, handles toolkit initialization and evaluation loop
- `models.py` — `UnifiedModel` wrapper around Toolshed's `ToolAgent`
- `robospatial.py`, `blink.py`, etc. — Dataset-specific evaluators with scoring logic

