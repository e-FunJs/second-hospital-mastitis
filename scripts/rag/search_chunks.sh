#!/usr/bin/env bash
# 用途：调试向量检索，直接查看 query 命中的 chunk。
# 输入：查询文本、默认 data/index/${RAG_CORPUS}/${RAG_TIER}/ 下的索引。
# 输出：终端打印检索结果；不默认写文件。

set -euo pipefail

QUERY="${1:-}"
TOP_K="${2:-8}"
CORPUS="${RAG_CORPUS:-english}"
TIER="${RAG_TIER:-broad}"
INDEX_DIR="${RAG_INDEX_DIR:-data/index/${CORPUS}/${TIER}}"

if [[ -z "${QUERY}" ]]; then
  echo "Usage: bash scripts/rag/search_chunks.sh \"query text\" [top_k]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

python -m rag_medical.common.search_chunks "${QUERY}" \
  --top-k "${TOP_K}" \
  --index "${INDEX_DIR}/faiss.index" \
  --metadata "${INDEX_DIR}/chunk_metadata.jsonl"
