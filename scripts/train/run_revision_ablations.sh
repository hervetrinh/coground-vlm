#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_revision_campaign_2xa100.sh" ablations "$@"
