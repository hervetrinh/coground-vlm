#!/usr/bin/env bash
# Compatibility wrapper: the corrected main CoGround config already includes grounding BCE.
set -euo pipefail
exec bash "$(dirname "$0")/run_multiseed_refcoco.sh"
