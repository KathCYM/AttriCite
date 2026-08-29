# CiteGuard GRPO with veRL and vLLM

This scaffold targets veRL's current async agent-loop interface. It uses the 350
balanced 2024 examples for training and the disjoint 60-example 2024 split for
validation. The 300 balanced 2025 examples remain an untouched final test set.

## Prepare the server

Install veRL using its official vLLM installation instructions, then install the
small data dependency and the CiteGuard requirements:

```bash
pip install -r requirements.txt datasets pyarrow
```

Create the parquet files:

```bash
python -m training.verl.prepare_dataset \
  --train-csv training_data_collection/splits/train_2024_rl.csv \
  --validation-csv training_data_collection/splits/validation_2024_rl.csv \
  --test-csv training_data_collection/splits/test_2025_balanced.csv
```

The generated records use the custom `citeguard_agent` loop. It terminates the
trajectory immediately after a valid `select` action is executed and scored,
matching CiteGuard inference. Regenerate both parquet files whenever the agent
name or prompt construction in `prepare_dataset.py` changes.

RL prompts are derived from `src/retriever/prompt_templates/few_shot_tool.txt` and
the model emits the same `{reason, action}` JSON used by `src.main`. The custom
agent loop parses that JSON and dispatches it to veRL's native `citeguard` tool.
Because this prompt is substantially larger than the earlier short prompt, the
launcher reserves 12,288 prompt tokens and a 32,768-token total context.

## Required smoke test

The native function-calling trajectory differs from the existing evaluator's JSON
action syntax. Before a full run, launch a one-step pilot and inspect generations,
tool success/error rates, reward variance, and GPU memory:

```bash
NUM_GPUS=4 bash training/verl/run_grpo_qwen3_4b.sh \
  trainer.resume_mode=disable trainer.total_training_steps=1 \
  trainer.val_before_train=False trainer.save_freq=1 trainer.test_freq=-1
```

Do not proceed if most groups have identical rewards or Semantic Scholar errors.
Infrastructure errors deliberately receive zero reward rather than a policy penalty.

## Full run

```bash
sbatch training/verl/slurm_grpo_qwen3_4b.sh
```

The defaults target four L40S GPUs using full-parameter FSDP, no CPU offload,
eight rollouts per prompt, two prompts per update, and a 5-call cap. They retain
the same seven actions, terminal `select`, and paper-buffer rules as inference.
Search, snippet, and PDF observations use the same deterministic bounds in both
environments. Adjust batch sizes and `NUM_GPUS` to the server. Keep the 2025 test
set out of all tuning decisions; evaluate it once after choosing a checkpoint using
the 2024 validation set.

Reward is 1 only for a correct selected title and 0 otherwise. Infrastructure failures
are recorded separately and should be excluded from policy learning rather than treated
as negative task outcomes.
