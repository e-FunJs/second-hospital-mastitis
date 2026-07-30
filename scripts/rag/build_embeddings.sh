#!/usr/bin/env bash
# 用途：把指定语料的 RAG chunk 编码成向量矩阵。
# 输入：默认 data/articles/processed/${RAG_CORPUS}/rag_chunks.jsonl。
# 输出：默认 data/index/${RAG_CORPUS}/${RAG_TIER}/ 下的 embedding 文件。
# 说明：RAG_CORPUS 默认 english，RAG_TIER 默认 broad；可用 RAG_INPUT 覆盖输入。

set -euo pipefail

LIMIT="${1:-}"
CORPUS="${RAG_CORPUS:-english}"
TIER="${RAG_TIER:-broad}"
INPUT_PATH="${RAG_INPUT:-data/articles/processed/${CORPUS}/rag_chunks.jsonl}"
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

CMD=(
  python -m rag_medical.common.build_embeddings
  --input "${INPUT_PATH}"
  --embedding-out "${INDEX_DIR}/chunk_embeddings.npy"
  --metadata-out "${INDEX_DIR}/chunk_metadata.jsonl"
  --manifest "${INDEX_DIR}/embedding_manifest.json"
  --config configs/embedding.yaml
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}"

echo
echo "Generated embedding files:"
ls -lh \
  "${INDEX_DIR}/chunk_embeddings.npy" \
  "${INDEX_DIR}/chunk_metadata.jsonl" \
  "${INDEX_DIR}/embedding_manifest.json"
