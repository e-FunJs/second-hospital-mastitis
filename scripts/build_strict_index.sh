#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

mkdir -p data/index_strict

EMBED_CMD=(
  python -m rag_medical.build_embeddings
  --input data/articles/processed/rag_chunks_strict.jsonl
  --embedding-out data/index_strict/chunk_embeddings.npy
  --metadata-out data/index_strict/chunk_metadata.jsonl
  --manifest data/index_strict/embedding_manifest.json
  --config configs/embedding.yaml
)

if [[ -n "${LIMIT}" ]]; then
  EMBED_CMD+=(--limit "${LIMIT}")
fi

"${EMBED_CMD[@]}"

python -m rag_medical.build_faiss_index \
  --embeddings data/index_strict/chunk_embeddings.npy \
  --metadata data/index_strict/chunk_metadata.jsonl \
  --index-out data/index_strict/faiss.index \
  --manifest data/index_strict/faiss_manifest.json

echo
echo "Generated strict index files:"
ls -lh \
  data/index_strict/chunk_embeddings.npy \
  data/index_strict/chunk_metadata.jsonl \
  data/index_strict/embedding_manifest.json \
  data/index_strict/faiss.index \
  data/index_strict/faiss_manifest.json
