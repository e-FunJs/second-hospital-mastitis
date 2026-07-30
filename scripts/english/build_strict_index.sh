#!/usr/bin/env bash
# 用途：基于“规则筛选 + BGE 动物语义复核”后的最终 strict chunk 构建 FAISS 索引。
# 前置步骤：
#   1. bash scripts/english/filter_corpus.sh
#   2. bash scripts/english/animal_filter.sh
# 输入：data/articles/processed/english/semantic/rag_chunks_strict.jsonl。
# 输出：data/index/english/strict/chunk_embeddings.npy、chunk_metadata.jsonl、faiss.index 与两个 manifest。

set -euo pipefail

LIMIT="${1:-}"
STRICT_CHUNKS="data/articles/processed/english/semantic/rag_chunks_strict.jsonl"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

if [[ ! -f "${STRICT_CHUNKS}" ]]; then
  echo "Final semantic-filtered chunks not found: ${STRICT_CHUNKS}" >&2
  echo "Run: bash scripts/english/animal_filter.sh" >&2
  exit 2
fi

mkdir -p data/index/english/strict

EMBED_CMD=(
  python -m rag_medical.common.build_embeddings
  --input "${STRICT_CHUNKS}"
  --embedding-out data/index/english/strict/chunk_embeddings.npy
  --metadata-out data/index/english/strict/chunk_metadata.jsonl
  --manifest data/index/english/strict/embedding_manifest.json
  --config configs/embedding.yaml
)

if [[ -n "${LIMIT}" ]]; then
  EMBED_CMD+=(--limit "${LIMIT}")
fi

"${EMBED_CMD[@]}"

python -m rag_medical.common.build_faiss_index \
  --embeddings data/index/english/strict/chunk_embeddings.npy \
  --metadata data/index/english/strict/chunk_metadata.jsonl \
  --index-out data/index/english/strict/faiss.index \
  --manifest data/index/english/strict/faiss_manifest.json

echo
echo "Generated final strict index files:"
ls -lh \
  data/index/english/strict/chunk_embeddings.npy \
  data/index/english/strict/chunk_metadata.jsonl \
  data/index/english/strict/embedding_manifest.json \
  data/index/english/strict/faiss.index \
  data/index/english/strict/faiss_manifest.json
