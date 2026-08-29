#!/bin/bash
#SBATCH --job-name=citeguard-qwen3-base-test2025
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000M
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VERL_VENV="${VERL_VENV:-$PROJECT_ROOT/verl/.venv}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-base-tp4}"
TEST_DATA="${TEST_DATA:-$PROJECT_ROOT/training_data_collection/splits/test_2025_balanced.csv}"
RESULT_PATH="${RESULT_PATH:-$PROJECT_ROOT/results/qwen3_4b_base_tp4_test_2025.json}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_LOG="${VLLM_LOG:-$PROJECT_ROOT/logs/vllm-${SERVED_MODEL_NAME}-${SLURM_JOB_ID}.log}"
EVAL_LOG="${EVAL_LOG:-$PROJECT_ROOT/logs/eval-${SERVED_MODEL_NAME}-${SLURM_JOB_ID}.log}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"

source "$VERL_VENV/bin/activate"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Base model: $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Results: $RESULT_PATH"
nvidia-smi -L

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting the base Qwen3-4B vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
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

echo "vLLM is ready; starting the 2025 baseline evaluation."
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
