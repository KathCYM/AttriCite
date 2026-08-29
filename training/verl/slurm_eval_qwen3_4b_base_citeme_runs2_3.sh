#!/bin/bash
#SBATCH --job-name=citeguard-base-citeme-runs23
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000M
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VERL_VENV="${VERL_VENV:-$PROJECT_ROOT/verl/.venv}"
DATASET="${DATASET:-$PROJECT_ROOT/DATASET.csv}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/citeme_temperature_sampling}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-base-citeme-tp2}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_ACTIONS="${MAX_ACTIONS:-5}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
VLLM_LOG="${VLLM_LOG:-$PROJECT_ROOT/logs/vllm-${SERVED_MODEL_NAME}-${SLURM_JOB_ID}.log}"

cd "$PROJECT_ROOT"
mkdir -p "$RESULT_DIR" "$PROJECT_ROOT/logs"

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
echo "CiteMe dataset: $DATASET"
echo "Existing first run: $PROJECT_ROOT/results/qwen3.json"
echo "Running CiteMe repetitions 2 and 3 at temperature $TEMPERATURE"
nvidia-smi -L

if [[ ! -f "$DATASET" ]]; then
  echo "CiteMe dataset not found: $DATASET" >&2
  exit 1
fi

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting base Qwen3-4B vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
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

for run in 2 3; do
  RESULT_PATH="$RESULT_DIR/qwen3_4b_base_citeme_temp0p7_run${run}.json"
  EVAL_LOG="$PROJECT_ROOT/logs/eval-qwen3-base-citeme-temp0p7-run${run}-${SLURM_JOB_ID}.log"

  echo "Evaluating base Qwen3-4B on CiteMe: run $run/3"
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

echo "Completed base-model CiteMe runs 2 and 3."
