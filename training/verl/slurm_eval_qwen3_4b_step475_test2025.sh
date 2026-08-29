#!/bin/bash
#SBATCH --job-name=citeguard-qwen3-test2025
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
STEP="${STEP:-475}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_full_grpo_json_fewshot_kl_n8_max5}"
ACTOR_DIR="${ACTOR_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME/global_step_$STEP/actor}"
MERGED_DIR="${MERGED_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME/global_step_$STEP/actor_hf}"
TEST_DATA="${TEST_DATA:-$PROJECT_ROOT/training_data_collection/splits/test_2025_balanced.csv}"
RESULT_PATH="${RESULT_PATH:-$PROJECT_ROOT/results/qwen3_4b_citeguard_step${STEP}_test_2025.json}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-citeguard-step${STEP}}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_LOG="${VLLM_LOG:-$PROJECT_ROOT/logs/vllm-${SERVED_MODEL_NAME}-${SLURM_JOB_ID}.log}"
EVAL_LOG="${EVAL_LOG:-$PROJECT_ROOT/logs/eval-${SERVED_MODEL_NAME}-${SLURM_JOB_ID}.log}"

cd "$PROJECT_ROOT"
mkdir -p "$MERGED_DIR" "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"

source "$VERL_VENV/bin/activate"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Checkpoint: $ACTOR_DIR"
echo "Merged model: $MERGED_DIR"
echo "Test data: $TEST_DATA"
echo "Results: $RESULT_PATH"
nvidia-smi -L

if [[ ! -f "$ACTOR_DIR/fsdp_config.json" ]]; then
  echo "FSDP actor checkpoint not found: $ACTOR_DIR" >&2
  exit 1
fi

if [[ ! -f "$MERGED_DIR/config.json" ]] || \
   { ! compgen -G "$MERGED_DIR/*.safetensors" >/dev/null && \
     ! compgen -G "$MERGED_DIR/*.bin" >/dev/null; }; then
  echo "Merging the FSDP actor checkpoint..."
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$MERGED_DIR"
fi

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting vLLM..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM (pid $VLLM_PID)..."
READY=0
for _ in $(seq 1 120); do
  if curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited during startup. Last log lines:" >&2
    tail -100 "$VLLM_LOG" >&2
    exit 1
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  echo "vLLM was not ready after 10 minutes. Last log lines:" >&2
  tail -100 "$VLLM_LOG" >&2
  exit 1
fi

echo "vLLM is ready; starting the 2025 test evaluation."
python -m src.main \
  --dataset "$TEST_DATA" \
  --result_path "$RESULT_PATH" \
  --model_name "$SERVED_MODEL_NAME" \
  --use_vllm \
  --vllm_base_url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --temperature 0 \
  --max_actions 5 \
  --no_interactive_context \
  2>&1 | tee "$EVAL_LOG"

python - "$RESULT_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    rows = json.load(handle)["results"]

total = len(rows)
correct = sum(row.get("is_correct") is True for row in rows)
selected = sum(row.get("selected") is not None for row in rows)
errors = sum(row.get("status") == "error" for row in rows)

print(f"Examples: {total}")
print(f"Correct:  {correct}/{total} = {correct / total:.2%}")
print(f"Selected: {selected}/{total} = {selected / total:.2%}")
print(f"Errors:   {errors}")
PY
