#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

TAG="teacher-v102-assets-v1"
ASSET="dist/teacher-v102-assets-v1.tar.gz"
CHECKSUM="$ASSET.sha256"

command -v gh >/dev/null || { echo "[STOP] GitHub CLI (gh) is not installed"; exit 1; }
gh auth status
test -f "$ASSET" || { echo "[STOP] missing $ASSET"; exit 1; }
test -f "$CHECKSUM" || { echo "[STOP] missing $CHECKSUM"; exit 1; }
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "[STOP] release $TAG already exists; refusing to replace published assets"
  exit 1
fi

gh release create "$TAG" "$ASSET" "$CHECKSUM" --target main --title "Teacher-v10.2 reproduction assets v1" --notes "Portable, SHA-256-bound prerequisites for the Teacher-v10.2 reproduction. No Teacher-v10.2 checkpoint is included."
