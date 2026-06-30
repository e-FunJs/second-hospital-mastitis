#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="nonpuerperal-mastitis-rag"
BASE_DIR="/home/amax/E-FUN/Secondo_Ospedale"
PROJECT_DIR="${BASE_DIR}/${PROJECT_NAME}"

mkdir -p "${BASE_DIR}"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llamafactory
else
  echo "conda not found; please activate llamafactory manually before running Python commands."
fi

python -m pip install -e .

echo "Project ready at: ${PROJECT_DIR}"
echo "Try:"
echo "bash scripts/run_pubmed_searches.sh 100"
