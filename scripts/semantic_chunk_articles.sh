#!/usr/bin/env bash
# 用途：对解析后的文章段落做语义分块。
# 输入：data/articles/processed/article_sections.jsonl。
# 输出：article_chunks.jsonl 与 chunk_manifest.csv。

set -euo pipefail

LIMIT_GROUPS="${1:-}"

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
  python -m rag_medical.semantic_chunk
  --input data/articles/processed/article_sections.jsonl
  --out data/articles/processed/article_chunks.jsonl
  --manifest data/articles/processed/chunk_manifest.csv
  --model-path models/bge/bge-m3
)

if [[ -n "${LIMIT_GROUPS}" ]]; then
  CMD+=(--limit-groups "${LIMIT_GROUPS}")
fi

"${CMD[@]}"

echo
echo "Generated semantic chunk files:"
ls -lh data/articles/processed/article_chunks.jsonl data/articles/processed/chunk_manifest.csv
