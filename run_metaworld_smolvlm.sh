#!/usr/bin/env bash
# Run MetaWorld tasks with RL-VLM-F using local SmolVLM2 (vLLM OpenAI-compatible API).
# Requirements:
#   - vLLM server running, e.g.:
#       vllm serve HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
#         --dtype bfloat16 --trust-remote-code --max-model-len 4096 \
#         --port 8000 --api-key test-key
#   - Set SMOLVLM_VLLM_URL / SMOLVLM_VLLM_API_KEY to match your server.

set -e

PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTHONPATH
export GEMINI_API_KEY=${GEMINI_API_KEY:-dummy}
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SMOLVLM_VLLM_URL=${SMOLVLM_VLLM_URL:-http://localhost:8000/v1}
export SMOLVLM_VLLM_API_KEY=${SMOLVLM_VLLM_API_KEY:-test-key}

TASKS=(
  hammer-v3-goal-observable
  push-wall-v3-goal-observable
  faucet-close-v3-goal-observable
  push-back-v3-goal-observable
  stick-pull-v3-goal-observable
  handle-press-side-v3-goal-observable
  push-v3-goal-observable
  shelf-place-v3-goal-observable
  window-close-v3-goal-observable
  peg-unplug-side-v3-goal-observable
)

SEEDS=(0 1 2 3 4)

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ">>> Running ${task} seed ${seed}"
    python train_PEBBLE.py \
      env=metaworld_${task} \
      seed=${seed} \
      reward=learn_from_preference \
      vlm=smolvlm_vllm \
      vlm_label=1 \
      segment=1 \
      image_reward=1 \
      reward_batch=40 \
      reward_update=5 \
      num_interact=4000 \
      num_train_steps=1000000 \
      agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 \
      gradient_update=1 activation=tanh num_unsup_steps=9000 \
      feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 \
      num_eval_episodes=1 \
      cached_label_path=None \
      exp_name=smolvlm_vllm
  done
done

echo "All jobs dispatched."
