#!/usr/bin/env bash
set -euo pipefail

MAX_RESULTS="${1:-100}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "llamafactory" ]]; then
    conda activate llamafactory
  fi
fi

mkdir -p data/registry/raw

python -m rag_medical.search_pubmed \
  --query-key core_english \
  --max-results "${MAX_RESULTS}" \
  --out data/registry/raw/core.csv

python -m rag_medical.search_pubmed \
  --query-key treatment_outcome \
  --max-results "${MAX_RESULTS}" \
  --out data/registry/raw/outcome.csv

python -m rag_medical.search_pubmed \
  --query-key ultrasound \
  --max-results "${MAX_RESULTS}" \
  --out data/registry/raw/ultrasound.csv

python -m rag_medical.search_pubmed \
  --query-key therapies \
  --max-results "${MAX_RESULTS}" \
  --out data/registry/raw/therapy.csv

echo
echo "Generated registry files:"
ls -lh data/registry/raw/core.csv data/registry/raw/outcome.csv data/registry/raw/ultrasound.csv data/registry/raw/therapy.csv

echo
echo "Next step:"
echo "Use the generated CSV files to identify PMID/PMCID records, then fetch PMC Open Access full text."
