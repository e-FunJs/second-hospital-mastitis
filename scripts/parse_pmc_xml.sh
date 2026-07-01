#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

mkdir -p data/articles/processed

CMD=(
  python -m rag_medical.parse_pmc_xml
  --xml-dir data/articles/raw/pmc_xml
  --out data/articles/processed/article_sections.jsonl
  --manifest data/articles/processed/article_parse_manifest.csv
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}"

echo
echo "Parsed article section file:"
ls -lh data/articles/processed/article_sections.jsonl

echo
echo "Parse manifest:"
ls -lh data/articles/processed/article_parse_manifest.csv

