#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/training/verl/data/train.parquet}"
VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/training/verl/data/validation.parquet}"
NUM_GPUS="${NUM_GPUS:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_full_grpo_json_fewshot_kl_n8_max5}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME}"
ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/training/verl/rollouts/$EXPERIMENT_NAME}"
VALIDATION_DIR="${VALIDATION_DIR:-$PROJECT_ROOT/training/verl/validation_rollouts/$EXPERIMENT_NAME}"

mkdir -p "$CHECKPOINT_DIR" "$ROLLOUT_DIR" "$VALIDATION_DIR"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size=2 \
  data.max_prompt_length=12288 \
  data.max_response_length=20480 \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.max_model_len=32768 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$NUM_GPUS" \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.load_format=dummy \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_ROOT/training/verl/citeguard_agent_loop.yaml" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=16064 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_ROOT/training/verl/citeguard_tool.yaml" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path="$PROJECT_ROOT/training/verl/reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=25 \
  trainer.test_freq=25 \
  trainer.max_actor_ckpt_to_keep=25 \
  trainer.total_epochs=3 \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  trainer.validation_data_dir="$VALIDATION_DIR" \
  trainer.log_val_generations=20 \
  trainer.logger='["console"]' \
  trainer.project_name=citeguard \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  "$@"
