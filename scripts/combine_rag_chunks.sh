#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

python -m rag_medical.combine_chunks \
  --input data/articles/processed/article_chunks.jsonl \
  --input data/articles/processed/abstract_chunks.jsonl \
  --out data/articles/processed/rag_chunks.jsonl

echo
echo "Generated mixed RAG chunk file:"
ls -lh data/articles/processed/rag_chunks.jsonl
