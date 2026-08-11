#!/usr/bin/env bash
# Staged corrected training campaign. Every run occupies both A100s through DDP.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"

STAGE=${1:-core}
if [[ $# -gt 0 ]]; then
  shift
fi
GLOBAL_OVERRIDES=("$@")
STEPS=${COGROUND_STEPS:-3000}
RUN_ROOT=${COGROUND_OUTPUT_DIR:-results/runs}
LOG_DIR=${COGROUND_LOG_DIR:-logs/2xa100}
mkdir -p "$LOG_DIR"

read -r -a CORE_SEEDS <<< "${COGROUND_CORE_SEEDS:-17 18 19 20 21}"
read -r -a TRANSFER_SEEDS <<< "${COGROUND_TRANSFER_SEEDS:-17 18 19}"
read -r -a DECODING_SEEDS <<< "${COGROUND_DECODING_SEEDS:-17 18 19 20 21}"
read -r -a LAYERS <<< "${COGROUND_LAYERS:-0 2 4 6 8 12 16 20 24}"

run_one() {
  local config=$1
  local experiment=$2
  local seed=$3
  shift 3
  local checkpoint="${RUN_ROOT}/${experiment}/step_$(printf '%08d' "$STEPS")"
  if [[ -f "${checkpoint}/trainer_state.pt" ]]; then
    echo "SKIP complete: ${experiment}"
    return
  fi
  echo "TRAIN ${experiment} on GPUs ${COGROUND_CUDA_DEVICES:-0,1}"
  bash scripts/train/launch_2xa100.sh \
    "$config" \
    experiment_id="$experiment" \
    seed="$seed" \
    output_dir="$RUN_ROOT" \
    training.max_steps="$STEPS" \
    training.save_every="$STEPS" \
    "${GLOBAL_OVERRIDES[@]}" \
    "$@" \
    2>&1 | tee -a "${LOG_DIR}/${experiment}.log"
}

run_core() {
  local configs=(
    configs/experiments/revision/gemma4_refcoco_coground.yaml
    configs/experiments/revision/gemma4_refcoco_matched_mlp.yaml
    configs/experiments/revision/gemma4_refcoco_bilinear.yaml
  )
  local names=(coground matched_mlp bilinear)
  local index seed
  for index in "${!configs[@]}"; do
    for seed in "${CORE_SEEDS[@]}"; do
      run_one "${configs[$index]}" "revision_gemma4_refcoco_${names[$index]}_seed${seed}" "$seed"
    done
  done
}

run_ablations() {
  local configs=(
    configs/experiments/revision/ablations/frozen_cosine.yaml
    configs/experiments/revision/ablations/occupancy_solver.yaml
    configs/experiments/revision/ablations/identity_block.yaml
    configs/experiments/revision/ablations/no_simplex.yaml
    configs/experiments/revision/ablations/shuffled_grounding.yaml
    configs/experiments/revision/ablations/independent_decoding_state.yaml
  )
  local names=(frozen_cosine occupancy_solver identity_block no_simplex shuffled_grounding independent_decoding)
  local index
  for index in "${!configs[@]}"; do
    run_one "${configs[$index]}" "revision_gemma4_refcoco_${names[$index]}_seed17" 17
  done
}

run_layer_sweep() {
  local configs=(
    configs/experiments/revision/gemma4_refcoco_coground.yaml
    configs/experiments/revision/gemma4_refcoco_matched_mlp.yaml
    configs/experiments/revision/gemma4_refcoco_bilinear.yaml
  )
  local names=(coground matched_mlp bilinear)
  local index layer
  for index in "${!configs[@]}"; do
    for layer in "${LAYERS[@]}"; do
      run_one \
        "${configs[$index]}" \
        "revision_gemma4_refcoco_${names[$index]}_lt${layer}_seed17" \
        17 \
        method.layer_tap="$layer" \
        grounding_baseline.layer_tap="$layer"
    done
  done
}

run_transfer() {
  local configs=(
    configs/experiments/revision/gemma4_refcocog_coground.yaml
    configs/experiments/revision/gemma4_refcocog_matched_mlp.yaml
    configs/experiments/revision/gemma4_refcocog_bilinear.yaml
    configs/experiments/revision/paligemma2_refcoco_coground.yaml
    configs/experiments/revision/paligemma2_refcoco_matched_mlp.yaml
    configs/experiments/revision/paligemma2_refcoco_bilinear.yaml
  )
  local names=(
    gemma4_refcocog_coground
    gemma4_refcocog_matched_mlp
    gemma4_refcocog_bilinear
    paligemma2_refcoco_coground
    paligemma2_refcoco_matched_mlp
    paligemma2_refcoco_bilinear
  )
  local index seed
  for index in "${!configs[@]}"; do
    for seed in "${TRANSFER_SEEDS[@]}"; do
      run_one "${configs[$index]}" "revision_${names[$index]}_seed${seed}" "$seed"
    done
  done
}

run_decoding() {
  local seed
  for seed in "${DECODING_SEEDS[@]}"; do
    run_one \
      configs/experiments/revision/gemma4_refcoco_decoding_prior.yaml \
      "revision_gemma4_refcoco_decoding_prior_seed${seed}" \
      "$seed"
  done
}

case "$STAGE" in
  core) run_core ;;
  ablations) run_ablations ;;
  layer-sweep) run_layer_sweep ;;
  transfer) run_transfer ;;
  decoding) run_decoding ;;
  all)
    run_core
    run_ablations
    run_layer_sweep
    run_transfer
    run_decoding
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Expected one of: core, ablations, layer-sweep, transfer, decoding, all" >&2
    exit 2
    ;;
esac
