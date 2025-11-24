#!/usr/bin/env bash
# Quick launcher for MetaWorld tasks using cached VLM labels (no real API calls).

set -euo pipefail

# Go to repo root
cd "$(dirname "$0")"

# Dummy keys to satisfy import-time checks; no real VLM calls when cached labels are used.
export GEMINI_API_KEY="${GEMINI_API_KEY:-dummy}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# Headless rendering
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Make local packages (rlkit, softgym, etc.) importable
export PYTHONPATH="$PWD"

# Choose your MetaWorld task + cached labels here.
# Replace env/cached_label_path below to match other tasks from run.sh.

python train_PEBBLE.py \
  env=metaworld_soccer-v2 \
  seed=0 \
  reward=learn_from_preference \
  vlm=gemini_free_form \
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
  cached_label_path=data/cached_labels/Soccer/seed_1/

echo "Done."
