#!/bin/bash
#SBATCH --job-name=citeguard-biomed-base-step425
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160000M
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VERL_VENV="${VERL_VENV:-$PROJECT_ROOT/verl/.venv}"
TEST_DATA="${TEST_DATA:-$PROJECT_ROOT/training_data_collection/splits/test_2025_biomed.csv}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/biomed_temperature_sampling}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_full_grpo_json_fewshot_kl_n8_max5}"
STEP="${STEP:-425}"
ACTOR_DIR="${ACTOR_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME/global_step_$STEP/actor}"
MERGED_DIR="${MERGED_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME/global_step_$STEP/actor_hf}"

TEMPERATURE="${TEMPERATURE:-0.7}"
NUM_RUNS="${NUM_RUNS:-3}"
MAX_ACTIONS="${MAX_ACTIONS:-5}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

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
echo "Biomedical test data: $TEST_DATA"
echo "Checkpoint actor: $ACTOR_DIR"
echo "Merged checkpoint: $MERGED_DIR"
echo "Temperature: $TEMPERATURE"
echo "Runs per model: $NUM_RUNS"
nvidia-smi -L

if [[ ! -f "$TEST_DATA" ]]; then
  echo "Biomedical test split not found: $TEST_DATA" >&2
  exit 1
fi

VLLM_PID=""

stop_server() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM process $VLLM_PID..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  VLLM_PID=""
}

cleanup() {
  stop_server
}
trap cleanup EXIT INT TERM

start_server() {
  local model_path="$1"
  local served_name="$2"
  local server_log="$3"

  echo "Starting $served_name from $model_path..."
  python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$served_name" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --generation-config vllm \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    >"$server_log" 2>&1 &
  VLLM_PID=$!

  local ready=0
  for _ in $(seq 1 120); do
    if curl --fail --silent "$VLLM_BASE_URL/models" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited during startup. Last log lines:" >&2
      tail -100 "$server_log" >&2
      exit 1
    fi
    sleep 5
  done

  if [[ "$ready" -ne 1 ]]; then
    echo "vLLM was not ready after 10 minutes. Last log lines:" >&2
    tail -100 "$server_log" >&2
    exit 1
  fi
}

summarize_result() {
  local result_path="$1"
  python - "$result_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    rows = json.load(handle).get("results", [])

total = len(rows)
correct = sum(row.get("is_correct") is True for row in rows)
selected = sum(row.get("selected") is not None for row in rows)
errors = sum(row.get("status") == "error" for row in rows)
accuracy = correct / total if total else 0.0
selection_rate = selected / total if total else 0.0

print(f"Result:    {path}")
print(f"Examples:  {total}")
print(f"Correct:   {correct}/{total} = {accuracy:.2%}")
print(f"Selected:  {selected}/{total} = {selection_rate:.2%}")
print(f"Errors:    {errors}")
PY
}

run_three_evaluations() {
  local label="$1"
  local served_name="$2"

  for run in $(seq 1 "$NUM_RUNS"); do
    local result_path="$RESULT_DIR/${label}_test_biomed_temp0p7_run${run}.json"
    local eval_log="$PROJECT_ROOT/logs/eval-${label}-biomed-temp0p7-run${run}-${SLURM_JOB_ID}.log"

    echo "Evaluating $label: run $run/$NUM_RUNS"
    python -m src.main \
      --dataset "$TEST_DATA" \
      --result_path "$result_path" \
      --model_name "$served_name" \
      --use_vllm \
      --vllm_base_url "$VLLM_BASE_URL" \
      --temperature "$TEMPERATURE" \
      --max_actions "$MAX_ACTIONS" \
      --no_interactive_context \
      2>&1 | tee "$eval_log"

    summarize_result "$result_path"
  done
}

BASE_SERVED_NAME="qwen3-4b-base-biomed-tp2"
BASE_VLLM_LOG="$PROJECT_ROOT/logs/vllm-${BASE_SERVED_NAME}-${SLURM_JOB_ID}.log"
start_server "$BASE_MODEL" "$BASE_SERVED_NAME" "$BASE_VLLM_LOG"
run_three_evaluations "qwen3_4b_base" "$BASE_SERVED_NAME"
stop_server

# A complete Hugging Face model is required by vLLM. Merge the four-rank FSDP
# checkpoint once if step 425 has not already been merged.
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

FT_SERVED_NAME="qwen3-4b-citeguard-step${STEP}-biomed-tp2"
FT_VLLM_LOG="$PROJECT_ROOT/logs/vllm-${FT_SERVED_NAME}-${SLURM_JOB_ID}.log"
start_server "$MERGED_DIR" "$FT_SERVED_NAME" "$FT_VLLM_LOG"
run_three_evaluations "qwen3_4b_step${STEP}" "$FT_SERVED_NAME"
stop_server

echo "Completed three biomedical evaluations for the base model and step $STEP."
