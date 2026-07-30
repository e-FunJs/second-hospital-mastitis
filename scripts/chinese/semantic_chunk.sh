#!/usr/bin/env bash
# 用途：用本地 BGE-M3 对中文页级正文执行章节感知的语义分块。
# 输入：data/articles/processed/chinese/article_pages.jsonl。
# 输出：article_chunks.jsonl 与 chunk_manifest.csv。
# 说明：只生成 chunk，不生成最终知识库 embedding 或 FAISS 索引。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

python -m rag_medical.chinese.semantic_chunk "$@"

echo
echo "Generated Chinese semantic chunk files:"
ls -lh \
  data/articles/processed/chinese/article_chunks.jsonl \
  data/articles/processed/chinese/chunk_manifest.csv
