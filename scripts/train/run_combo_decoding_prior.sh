#!/bin/bash
# Train the effective vocabulary-logit prior and compare generation with the
# same checkpoint with the prior disabled. No result is reported unless the
# corresponding public benchmark manifest exists.
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${COGROUND_CUDA_DEVICES:-0,1}
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

CONFIG=configs/experiments/revision/gemma4_refcoco_decoding_prior.yaml
STEPS=3000
SEEDS=(17 18 19 20 21)
mkdir -p logs

for SEED in "${SEEDS[@]}"; do
  EXP="revision_gemma4_refcoco_decoding_prior_seed${SEED}"
  CKPT="results/runs/${EXP}/step_$(printf '%08d' "$STEPS")"
  if [ ! -f "$CKPT/trainer_state.pt" ]; then
    bash scripts/train/launch_2xa100.sh "$CONFIG" \
      experiment_id="$EXP" \
      seed="$SEED" \
      training.max_steps="$STEPS" \
      training.save_every="$STEPS" \
      > "logs/${EXP}_train.log" 2>&1
  fi

  if [ -f data/pope/manifests/pope_val.jsonl ]; then
    python scripts/eval/run_benchmark.py \
      --config configs/eval/gemma4_pope_revision.yaml \
      --checkpoint "$CKPT" \
      experiment_id="revision_gemma4_pope_prior_seed${SEED}" \
      seed="$SEED" \
      > "logs/${EXP}_pope_prior.log" 2>&1
    python scripts/eval/run_benchmark.py \
      --config configs/eval/gemma4_pope_revision.yaml \
      --checkpoint "$CKPT" \
      experiment_id="revision_gemma4_pope_no_prior_seed${SEED}" \
      variant="coground_no_decoding_prior" \
      seed="$SEED" \
      decoding.enabled=false \
      decoding.lambda_dec=0.0 \
      > "logs/${EXP}_pope_no_prior.log" 2>&1
  fi

  if [ -f data/coco/manifests/chair_val.jsonl ]; then
    python scripts/eval/run_benchmark.py \
      --config configs/eval/gemma4_coco_chair_revision.yaml \
      --checkpoint "$CKPT" \
      experiment_id="revision_gemma4_chair_prior_seed${SEED}" \
      seed="$SEED" \
      > "logs/${EXP}_chair_prior.log" 2>&1
    python scripts/eval/run_benchmark.py \
      --config configs/eval/gemma4_coco_chair_revision.yaml \
      --checkpoint "$CKPT" \
      experiment_id="revision_gemma4_chair_no_prior_seed${SEED}" \
      variant="coground_no_decoding_prior" \
      seed="$SEED" \
      decoding.enabled=false \
      decoding.lambda_dec=0.0 \
      > "logs/${EXP}_chair_no_prior.log" 2>&1
  fi
done
