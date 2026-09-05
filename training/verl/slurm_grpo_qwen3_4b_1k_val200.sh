#!/bin/bash
#SBATCH --job-name=citeguard-qwen3-grpo-1k
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160000M
#SBATCH --time=30:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
VERL_VENV="${VERL_VENV:-$PROJECT_ROOT/verl/.venv}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_full_grpo_1k_val200_json_fewshot_kl_n8_max5}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/training/verl/data_1k_val200}"
TRAIN_CSV="${TRAIN_CSV:-$PROJECT_ROOT/training_data_collection/splits/train_2024_rl_1k.csv}"
VAL_CSV="${VAL_CSV:-$PROJECT_ROOT/training_data_collection/splits/validation_2024_rl_200.csv}"
TEST_CSV="${TEST_CSV:-$PROJECT_ROOT/training_data_collection/splits/test_2025_balanced.csv}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.parquet}"
VAL_DATA="${VAL_DATA:-$DATA_DIR/validation.parquet}"
RUN_LOG="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}-${SLURM_JOB_ID}.log"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs" "$DATA_DIR"

source "$VERL_VENV/bin/activate"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Training CSV: $TRAIN_CSV"
echo "Validation CSV: $VAL_CSV"
echo "Training Parquet: $TRAIN_DATA"
echo "Validation Parquet: $VAL_DATA"
echo "Expected optimization steps: 1000 / 2 * 3 = 1500"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

for path in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
  if [[ ! -f "$path" ]]; then
    echo "Required split file not found: $path" >&2
    exit 1
  fi
done

# Generate a separate set of veRL inputs. Existing training/verl/data files are
# never replaced.
if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" || ! -f "$DATA_DIR/test_2025.parquet" ]]; then
  echo "Preparing the 1k/200 veRL Parquet datasets..."
  python -m training.verl.prepare_dataset \
    --train-csv "$TRAIN_CSV" \
    --validation-csv "$VAL_CSV" \
    --test-csv "$TEST_CSV" \
    --output-dir "$DATA_DIR"
fi

ray stop --force >/dev/null 2>&1 || true
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

NUM_GPUS=4 \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$VAL_DATA" \
bash training/verl/run_grpo_qwen3_4b.sh \
  trainer.resume_mode=auto \
  trainer.total_epochs=3 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.max_actor_ckpt_to_keep=10 \
  2>&1 | tee "$RUN_LOG"
