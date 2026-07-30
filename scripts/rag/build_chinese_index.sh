#!/usr/bin/env bash
# 用途：用通用 embedding 和 FAISS 模块构建中文 strict 检索索引。
# 输入：data/articles/processed/chinese/filtered/rag_chunks_strict.jsonl。
# 输出：完整运行写入 data/index/chinese/strict/；
#       传入 chunk 数量时作为冒烟测试写入 data/index/chinese/smoke/。
# 用法：bash scripts/rag/build_chinese_index.sh [可选的 chunk 数量]。

set -euo pipefail

LIMIT="${1:-}"
INPUT_PATH="data/articles/processed/chinese/filtered/rag_chunks_strict.jsonl"
if [[ -n "${LIMIT}" ]]; then
  INDEX_DIR="${CHINESE_INDEX_DIR:-data/index/chinese/smoke}"
else
  INDEX_DIR="${CHINESE_INDEX_DIR:-data/index/chinese/strict}"
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

python -m rag_medical.common.check_corpus \
  --input "${INPUT_PATH}" \
  --expected-language zh \
  --report data/articles/processed/chinese/common_readiness_report.json

RAG_CORPUS=chinese \
RAG_TIER=strict \
RAG_INPUT="${INPUT_PATH}" \
RAG_INDEX_DIR="${INDEX_DIR}" \
bash scripts/rag/build_embeddings.sh "${LIMIT}"

RAG_CORPUS=chinese \
RAG_TIER=strict \
RAG_INDEX_DIR="${INDEX_DIR}" \
bash scripts/rag/build_faiss_index.sh

python -m rag_medical.common.validate_index \
  --index-dir "${INDEX_DIR}" \
  --expected-language zh \
  --report "${INDEX_DIR}/index_validation_report.json"

echo
echo "Generated Chinese strict index:"
ls -lh \
  "${INDEX_DIR}/chunk_embeddings.npy" \
  "${INDEX_DIR}/chunk_metadata.jsonl" \
  "${INDEX_DIR}/faiss.index" \
  "${INDEX_DIR}/embedding_manifest.json" \
  "${INDEX_DIR}/faiss_manifest.json" \
  "${INDEX_DIR}/index_validation_report.json"
