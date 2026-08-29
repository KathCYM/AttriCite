#!/bin/bash
#SBATCH --job-name=citeguard-temp-sweep
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
VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/training_data_collection/splits/validation_2024_rl.csv}"
SWEEP_DIR="${SWEEP_DIR:-$PROJECT_ROOT/results/temperature_sweep_step${STEP}_validation2024}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-citeguard-step${STEP}}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_LOG="${VLLM_LOG:-$PROJECT_ROOT/logs/vllm-temperature-sweep-step${STEP}-${SLURM_JOB_ID}.log}"
TEMPERATURES=(0.0 0.2 0.4 0.7)
NUM_RUNS="${NUM_RUNS:-5}"

cd "$PROJECT_ROOT"
mkdir -p "$MERGED_DIR" "$SWEEP_DIR" "$PROJECT_ROOT/logs"

source "$VERL_VENV/bin/activate"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Checkpoint: $ACTOR_DIR"
echo "Validation data: $VAL_DATA"
echo "Temperatures: ${TEMPERATURES[*]}"
echo "Runs per temperature: $NUM_RUNS"
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

for temperature in "${TEMPERATURES[@]}"; do
  temperature_label="${temperature/./p}"
  for run in $(seq 1 "$NUM_RUNS"); do
    result_path="$SWEEP_DIR/qwen_step${STEP}_temp${temperature_label}_run${run}.json"
    eval_log="$SWEEP_DIR/qwen_step${STEP}_temp${temperature_label}_run${run}.log"

    echo "Qwen sweep: temperature=$temperature run=$run/$NUM_RUNS"
    python -m src.main \
      --dataset "$VAL_DATA" \
      --result_path "$result_path" \
      --model_name "$SERVED_MODEL_NAME" \
      --use_vllm \
      --vllm_base_url "http://127.0.0.1:${VLLM_PORT}/v1" \
      --temperature "$temperature" \
      --max_actions 5 \
      --no_interactive_context \
      2>&1 | tee "$eval_log"
  done
done

# OpenAI sweep intentionally disabled for now. To enable it later, remove the
# leading "#" characters and set OPENAI_MODEL if needed.
# OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.4-mini-2026-03-17}"
# for temperature in "${TEMPERATURES[@]}"; do
#   temperature_label="${temperature/./p}"
#   for run in $(seq 1 "$NUM_RUNS"); do
#     result_path="$SWEEP_DIR/openai_temp${temperature_label}_run${run}.json"
#     eval_log="$SWEEP_DIR/openai_temp${temperature_label}_run${run}.log"
#     python -m src.main \
#       --dataset "$VAL_DATA" \
#       --result_path "$result_path" \
#       --model_name "$OPENAI_MODEL" \
#       --temperature "$temperature" \
#       --max_actions 5 \
#       --no_interactive_context \
#       2>&1 | tee "$eval_log"
#   done
# done

python - "$SWEEP_DIR" "$STEP" <<'PY'
import csv
import glob
import json
import os
import re
import statistics
import sys

sweep_dir, step = sys.argv[1:]
pattern = os.path.join(sweep_dir, f"qwen_step{step}_temp*_run*.json")
records = []

for path in sorted(glob.glob(pattern)):
    match = re.search(r"_temp(\d+p\d+)_run(\d+)\.json$", path)
    if not match:
        continue
    temperature = float(match.group(1).replace("p", "."))
    run = int(match.group(2))
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle).get("results", [])
    total = len(rows)
    correct = sum(row.get("is_correct") is True for row in rows)
    selected = sum(row.get("selected") is not None for row in rows)
    errors = sum(row.get("status") == "error" for row in rows)
    records.append(
        {
            "temperature": temperature,
            "run": run,
            "examples": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "selected": selected,
            "selection_rate": selected / total if total else 0.0,
            "errors": errors,
        }
    )

run_csv = os.path.join(sweep_dir, "runs.csv")
with open(run_csv, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

summary = []
for temperature in sorted({record["temperature"] for record in records}):
    group = [record for record in records if record["temperature"] == temperature]
    accuracies = [record["accuracy"] for record in group]
    selection_rates = [record["selection_rate"] for record in group]
    summary.append(
        {
            "temperature": temperature,
            "runs": len(group),
            "mean_accuracy": statistics.mean(accuracies),
            "std_accuracy": statistics.stdev(accuracies) if len(group) > 1 else 0.0,
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "mean_selection_rate": statistics.mean(selection_rates),
            "total_errors": sum(record["errors"] for record in group),
        }
    )

summary_csv = os.path.join(sweep_dir, "summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("\nTemperature sweep summary")
for row in summary:
    print(
        f"temp={row['temperature']:.1f} "
        f"accuracy={row['mean_accuracy']:.2%} +/- {row['std_accuracy']:.2%} "
        f"range=[{row['min_accuracy']:.2%}, {row['max_accuracy']:.2%}] "
        f"selected={row['mean_selection_rate']:.2%} "
        f"errors={row['total_errors']}"
    )
print(f"Per-run results: {run_csv}")
print(f"Summary: {summary_csv}")
PY
