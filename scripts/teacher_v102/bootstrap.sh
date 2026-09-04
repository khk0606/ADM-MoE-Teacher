#!/usr/bin/env bash
set -euo pipefail
trap 'BOOTSTRAP_RC=$?; echo "[DONE] Teacher-v10.2 bootstrap exit=$BOOTSTRAP_RC"' EXIT

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

command -v conda >/dev/null || {
  echo "[STOP] Conda is required. Install Miniconda or Anaconda, then rerun this command."
  exit 1
}

if conda env list | awk '$1 == "afford" { found=1 } END { exit(found ? 0 : 1) }'; then
  echo "[ENV] updating existing Conda environment: afford"
  conda env update --name afford --file environment.teacher-v102.yml
else
  echo "[ENV] creating Conda environment: afford"
  conda env create --file environment.teacher-v102.yml
fi

conda run --name afford --no-capture-output \
  python -m pip install --upgrade pip setuptools wheel
conda run --name afford --no-capture-output \
  python -m pip install -r requirements.txt
conda run --name afford --no-capture-output \
  bash scripts/teacher_v102/reproduce.sh

echo "[NEXT] conda run --name afford --no-capture-output bash scripts/teacher_v102/view.sh"
