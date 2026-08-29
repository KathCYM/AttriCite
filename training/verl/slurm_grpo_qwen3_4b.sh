#!/bin/bash
#SBATCH --job-name=citeguard-qwen3-grpo
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160000M
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VERL_VENV="${VERL_VENV:-$PROJECT_ROOT/verl/.venv}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_full_grpo_json_fewshot_kl_n8_max5}"
RUN_LOG="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}-${SLURM_JOB_ID}.log"

cd "$PROJECT_ROOT"
mkdir -p logs

source "$VERL_VENV/bin/activate"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

ray stop --force >/dev/null 2>&1 || true
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

NUM_GPUS=4 \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
bash training/verl/run_grpo_qwen3_4b.sh \
  2>&1 | tee "$RUN_LOG"
