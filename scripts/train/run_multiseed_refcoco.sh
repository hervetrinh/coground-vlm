#!/usr/bin/env bash
# Five-seed CoGround-only convenience launcher on two A100s.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
STEPS=${COGROUND_STEPS:-3000}
RUN_ROOT=${COGROUND_OUTPUT_DIR:-results/runs}
read -r -a SEEDS <<< "${COGROUND_CORE_SEEDS:-17 18 19 20 21}"
mkdir -p logs/2xa100

for seed in "${SEEDS[@]}"; do
  experiment="revision_gemma4_refcoco_coground_seed${seed}"
  checkpoint="${RUN_ROOT}/${experiment}/step_$(printf '%08d' "$STEPS")/trainer_state.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "SKIP complete: ${experiment}"
    continue
  fi
  bash scripts/train/launch_2xa100.sh \
    configs/experiments/revision/gemma4_refcoco_coground.yaml \
    experiment_id="$experiment" \
    seed="$seed" \
    output_dir="$RUN_ROOT" \
    training.max_steps="$STEPS" \
    training.save_every="$STEPS" \
    2>&1 | tee -a "logs/2xa100/${experiment}.log"
done
