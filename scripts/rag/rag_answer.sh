#!/usr/bin/env bash
# 用途：执行第一层 RAG 检索并生成 evidence 与 prompt。
# 输入：用户问题、默认 data/index/${RAG_CORPUS}/${RAG_TIER}/ 下的索引。
# 输出：data/rag/answers/${RAG_CORPUS}/${RAG_TIER}/。

set -euo pipefail

QUESTION="${1:-}"
TOP_K="${2:-8}"
CORPUS="${RAG_CORPUS:-english}"
TIER="${RAG_TIER:-broad}"
INDEX_DIR="${RAG_INDEX_DIR:-data/index/${CORPUS}/${TIER}}"
OUTPUT_DIR="${RAG_OUTPUT_DIR:-data/rag/answers/${CORPUS}/${TIER}}"

if [[ -z "${QUESTION}" ]]; then
  echo "Usage: bash scripts/rag/rag_answer.sh \"question text\" [top_k]" >&2
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

python -m rag_medical.common.rag_answer "${QUESTION}" \
  --top-k "${TOP_K}" \
  --index "${INDEX_DIR}/faiss.index" \
  --metadata "${INDEX_DIR}/chunk_metadata.jsonl" \
  --output-dir "${OUTPUT_DIR}"
