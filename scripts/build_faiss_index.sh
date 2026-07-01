#!/usr/bin/env bash
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

mkdir -p data/index

python -m rag_medical.build_faiss_index \
  --embeddings data/index/chunk_embeddings.npy \
  --metadata data/index/chunk_metadata.jsonl \
  --index-out data/index/faiss.index \
  --manifest data/index/faiss_manifest.json

echo
echo "Generated FAISS index files:"
ls -lh data/index/faiss.index data/index/faiss_manifest.json
