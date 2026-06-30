#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "llamafactory" ]]; then
    conda activate llamafactory
  fi
fi

python -m rag_medical.merge_registry \
  --source-dir data/registry/raw \
  --out data/registry/processed/literature_registry.csv \
  --summary data/registry/processed/literature_registry_summary.md

echo
echo "Generated processed registry files:"
ls -lh data/registry/processed/literature_registry.csv data/registry/processed/literature_registry_summary.md
