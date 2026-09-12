#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
sha256sum -c SMALL_ROOM30_ANYWHERE_SHA256SUMS.txt > /dev/null
mode="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
case "$mode" in
  preflight)
    python -u prepare/train_small_room30_anywhere.py --preflight-only \
      --output-dir "${OUTPUT:-outputs/small_room30_student_anywhere_preflight01}" "$@" ;;
  train)
    python -u prepare/train_small_room30_anywhere.py \
      --output-dir "${OUTPUT:-outputs/small_room30_student_anywhere01}" "$@" ;;
  view)
    python -u prepare/view_small_room30_anywhere.py \
      --summary "${SUMMARY:-outputs/small_room30_student_anywhere01/summary.json}" \
      --port "${PORT:-8100}" "$@" ;;
  *) echo 'Usage: student_anywhere.sh preflight|train|view [additional Python arguments]'; exit 2 ;;
esac
