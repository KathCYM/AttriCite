#!/bin/bash
#SBATCH --job-name=citeguard-step475-citeme
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160000M
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VERL_VENV="${VERL_VENV:-$PROJECT_ROOT/verl/.venv}"
DATASET="${DATASET:-$PROJECT_ROOT/DATASET.csv}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/citeme_temperature_sampling}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_full_grpo_json_fewshot_kl_n8_max5}"
STEP="${STEP:-475}"
ACTOR_DIR="${ACTOR_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME/global_step_$STEP/actor}"
MERGED_DIR="${MERGED_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME/global_step_$STEP/actor_hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-citeguard-step${STEP}-citeme-tp2}"

TEMPERATURE="${TEMPERATURE:-0.7}"
NUM_RUNS="${NUM_RUNS:-3}"
MAX_ACTIONS="${MAX_ACTIONS:-5}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
VLLM_LOG="${VLLM_LOG:-$PROJECT_ROOT/logs/vllm-${SERVED_MODEL_NAME}-${SLURM_JOB_ID}.log}"

cd "$PROJECT_ROOT"
mkdir -p "$RESULT_DIR" "$PROJECT_ROOT/logs" "$(dirname "$MERGED_DIR")"

source "$VERL_VENV/bin/activate"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "CiteMe dataset: $DATASET"
echo "Checkpoint actor: $ACTOR_DIR"
echo "Merged checkpoint: $MERGED_DIR"
echo "Temperature: $TEMPERATURE"
echo "Runs: $NUM_RUNS"
nvidia-smi -L

if [[ ! -f "$DATASET" ]]; then
  echo "CiteMe dataset not found: $DATASET" >&2
  exit 1
fi

# A complete Hugging Face model is required by vLLM. Merge step 475 once if
# the corresponding FSDP actor has not already been merged.
if [[ ! -f "$MERGED_DIR/config.json" ]] || \
   { ! compgen -G "$MERGED_DIR/*.safetensors" >/dev/null && \
     ! compgen -G "$MERGED_DIR/*.bin" >/dev/null; }; then
  if [[ ! -f "$ACTOR_DIR/fsdp_config.json" ]]; then
    echo "FSDP actor checkpoint not found: $ACTOR_DIR" >&2
    exit 1
  fi
  echo "Merging step $STEP FSDP actor checkpoint..."
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$MERGED_DIR"
fi

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM process $VLLM_PID..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting step $STEP vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

READY=0
for _ in $(seq 1 120); do
  if curl --fail --silent "$VLLM_BASE_URL/models" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited during startup. Last log lines:" >&2
    tail -200 "$VLLM_LOG" >&2
    exit 1
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  echo "vLLM was not ready after 10 minutes. Last log lines:" >&2
  tail -200 "$VLLM_LOG" >&2
  exit 1
fi

for run in $(seq 1 "$NUM_RUNS"); do
  RESULT_PATH="$RESULT_DIR/qwen3_4b_step${STEP}_citeme_temp0p7_run${run}.json"
  EVAL_LOG="$PROJECT_ROOT/logs/eval-qwen3-step${STEP}-citeme-temp0p7-run${run}-${SLURM_JOB_ID}.log"

  echo "Evaluating step $STEP on CiteMe: run $run/$NUM_RUNS"
  python -m src.main \
    --dataset "$DATASET" \
    --result_path "$RESULT_PATH" \
    --model_name "$SERVED_MODEL_NAME" \
    --use_vllm \
    --vllm_base_url "$VLLM_BASE_URL" \
    --temperature "$TEMPERATURE" \
    --max_actions "$MAX_ACTIONS" \
    --no_interactive_context \
    2>&1 | tee "$EVAL_LOG"

  python - "$RESULT_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    rows = json.load(handle).get("results", [])

total = len(rows)
correct = sum(row.get("is_correct") is True for row in rows)
selected = sum(row.get("selected") is not None for row in rows)
errors = sum(row.get("status") == "error" for row in rows)

print(f"Result:   {path}")
print(f"Examples: {total}")
print(f"Correct:  {correct}/{total} = {correct / total:.2%}" if total else "Correct:  0/0")
print(f"Selected: {selected}/{total} = {selected / total:.2%}" if total else "Selected: 0/0")
print(f"Errors:   {errors}")
PY
done

echo "Completed $NUM_RUNS CiteMe evaluations for step $STEP."
