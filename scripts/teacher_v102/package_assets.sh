#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

SUMMARY="${1:-data/history_affordance_relational_teacher_v9_all_sittable_v1/experiments/teacher_lora_v101/fullfield_supervision_s20261030_v1/summary.json}"
OUTPUT="${2:-dist/teacher-v102-assets-v1.tar.gz}"

python prepare/package_teacher_v102_assets.py --repo-root "$ROOT" --v101-summary "$SUMMARY" --output "$OUTPUT"
