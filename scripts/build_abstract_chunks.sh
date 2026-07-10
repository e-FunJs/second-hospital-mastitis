#!/usr/bin/env bash
# 用途：把没有 PMC 全文的 PubMed 摘要转成 RAG chunk。
# 输入：data/registry/processed/literature_registry.csv。
# 输出：data/articles/processed/abstract_chunks.jsonl。

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

python -m rag_medical.abstract_chunks \
  --registry data/registry/processed/literature_registry.csv \
  --out data/articles/processed/abstract_chunks.jsonl

echo
echo "Generated abstract chunk file:"
ls -lh data/articles/processed/abstract_chunks.jsonl
