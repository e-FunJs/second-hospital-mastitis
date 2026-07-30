#!/usr/bin/env bash
# 用途：为指定语料的 embedding 构建 FAISS 检索索引。
# 输入/输出：默认 data/index/${RAG_CORPUS}/${RAG_TIER}/。

set -euo pipefail

CORPUS="${RAG_CORPUS:-english}"
TIER="${RAG_TIER:-broad}"
INDEX_DIR="${RAG_INDEX_DIR:-data/index/${CORPUS}/${TIER}}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

mkdir -p "${INDEX_DIR}"

python -m rag_medical.common.build_faiss_index \
  --embeddings "${INDEX_DIR}/chunk_embeddings.npy" \
  --metadata "${INDEX_DIR}/chunk_metadata.jsonl" \
  --index-out "${INDEX_DIR}/faiss.index" \
  --manifest "${INDEX_DIR}/faiss_manifest.json"

echo
echo "Generated FAISS index files:"
ls -lh "${INDEX_DIR}/faiss.index" "${INDEX_DIR}/faiss_manifest.json"
