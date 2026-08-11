#!/usr/bin/env bash
# Launch one resumable CoGround training run with one process per A100.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train/launch_2xa100.sh CONFIG [OmegaConf overrides ...]" >&2
  exit 2
fi

CONFIG=$1
shift
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${COGROUND_CUDA_DEVICES:-0,1}
export PYTHONPATH=${PROJECT_ROOT}/src
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#VISIBLE_GPUS[@]} -ne 2 ]]; then
  echo "COGROUND_CUDA_DEVICES must expose exactly two GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

PREFLIGHT_ARGS=(--config "$CONFIG" --world-size 2)
if [[ ${COGROUND_ALLOW_NON_A100:-0} == 1 ]]; then
  PREFLIGHT_ARGS+=(--allow-non-a100)
fi
if [[ ${COGROUND_SKIP_PREFLIGHT:-0} != 1 ]]; then
  python scripts/train/preflight_2xa100.py "${PREFLIGHT_ARGS[@]}" "$@"
fi

RESUME_ARGS=()
case "${COGROUND_RESUME:-auto}" in
  none|false|0)
    ;;
  auto|true|1)
    RESUME_ARGS+=(--resume auto)
    ;;
  *)
    RESUME_ARGS+=(--resume "$COGROUND_RESUME")
    ;;
esac

accelerate launch \
  --config_file configs/accelerate/2xa100.yaml \
  --num_processes 2 \
  scripts/train/train.py \
  --config "$CONFIG" \
  "${RESUME_ARGS[@]}" \
  "$@"
