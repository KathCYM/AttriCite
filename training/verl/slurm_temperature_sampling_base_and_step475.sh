#!/bin/bash
#SBATCH --job-name=citeguard-temp-sampling
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000M
#SBATCH --time=16:00:00
#SBATCH --array=0-19%4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VERL_VENV="${VERL_VENV:-$PROJECT_ROOT/verl/.venv}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
STEP="${STEP:-475}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_full_grpo_json_fewshot_kl_n8_max5}"
FT_MODEL="${FT_MODEL:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME/global_step_$STEP/actor_hf}"
VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/training_data_collection/splits/validation_2024_rl.csv}"
TEST_DATA="${TEST_DATA:-$PROJECT_ROOT/training_data_collection/splits/test_2025_balanced.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/temperature_sampling_validation_test}"
VLLM_PORT="${VLLM_PORT:-8000}"

TEMPERATURES=(0.0 0.2 0.4 0.7)
RUNS_PER_TEMPERATURE=5
TEMPERATURE_INDEX=$((SLURM_ARRAY_TASK_ID / RUNS_PER_TEMPERATURE))
RUN=$((SLURM_ARRAY_TASK_ID % RUNS_PER_TEMPERATURE + 1))
TEMPERATURE="${TEMPERATURES[$TEMPERATURE_INDEX]}"
TEMPERATURE_LABEL="${TEMPERATURE/./p}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"
source "$VERL_VENV/bin/activate"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Temperature: $TEMPERATURE"
echo "Run: $RUN/$RUNS_PER_TEMPERATURE"
nvidia-smi -L

if [[ ! -f "$FT_MODEL/config.json" ]] || \
   { ! compgen -G "$FT_MODEL/*.safetensors" >/dev/null && \
     ! compgen -G "$FT_MODEL/*.bin" >/dev/null; }; then
  echo "Merged fine-tuned model not found: $FT_MODEL" >&2
  exit 1
fi

VLLM_PID=""
stop_server() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  VLLM_PID=""
}
trap stop_server EXIT INT TERM

start_server() {
  local model_path="$1"
  local served_name="$2"
  local server_log="$3"

  python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$served_name" \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --generation-config vllm \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    >"$server_log" 2>&1 &
  VLLM_PID=$!

  local ready=0
  for _ in $(seq 1 120); do
    if curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
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

run_eval() {
  local dataset="$1"
  local result_path="$2"
  local served_name="$3"
  local eval_log="$4"

  python -m src.main \
    --dataset "$dataset" \
    --result_path "$result_path" \
    --model_name "$served_name" \
    --use_vllm \
    --vllm_base_url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --temperature "$TEMPERATURE" \
    --max_actions 5 \
    --no_interactive_context \
    2>&1 | tee "$eval_log"
}

# Base Qwen3-4B: validation and test.
BASE_SERVED_NAME="qwen3-4b-base-temp${TEMPERATURE_LABEL}-run${RUN}"
start_server \
  "$BASE_MODEL" \
  "$BASE_SERVED_NAME" \
  "$PROJECT_ROOT/logs/vllm-base-temp${TEMPERATURE_LABEL}-run${RUN}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

run_eval \
  "$VAL_DATA" \
  "$OUTPUT_DIR/base_validation_temp${TEMPERATURE_LABEL}_run${RUN}.json" \
  "$BASE_SERVED_NAME" \
  "$OUTPUT_DIR/base_validation_temp${TEMPERATURE_LABEL}_run${RUN}.log"

run_eval \
  "$TEST_DATA" \
  "$OUTPUT_DIR/base_test_temp${TEMPERATURE_LABEL}_run${RUN}.json" \
  "$BASE_SERVED_NAME" \
  "$OUTPUT_DIR/base_test_temp${TEMPERATURE_LABEL}_run${RUN}.log"

stop_server

# Fine-tuned checkpoint: validation was already swept, so run test only.
FT_SERVED_NAME="qwen3-4b-citeguard-step${STEP}-temp${TEMPERATURE_LABEL}-run${RUN}"
start_server \
  "$FT_MODEL" \
  "$FT_SERVED_NAME" \
  "$PROJECT_ROOT/logs/vllm-step${STEP}-temp${TEMPERATURE_LABEL}-run${RUN}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

run_eval \
  "$TEST_DATA" \
  "$OUTPUT_DIR/step${STEP}_test_temp${TEMPERATURE_LABEL}_run${RUN}.json" \
  "$FT_SERVED_NAME" \
  "$OUTPUT_DIR/step${STEP}_test_temp${TEMPERATURE_LABEL}_run${RUN}.log"

stop_server

# OpenAI sweep intentionally disabled. When enabled later, run validation and
# test for the same TEMPERATURE and RUN in a separate CPU-only Slurm job.
# OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.4-mini-2026-03-17}"
# run OpenAI validation -> openai_validation_temp${TEMPERATURE_LABEL}_run${RUN}.json
# run OpenAI test       -> openai_test_temp${TEMPERATURE_LABEL}_run${RUN}.json

echo "Completed temperature=$TEMPERATURE run=$RUN"
