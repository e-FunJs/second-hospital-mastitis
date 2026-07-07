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

mkdir -p data/index

CMD=(
  python -m rag_medical.build_embeddings
  --input data/articles/processed/rag_chunks.jsonl
  --embedding-out data/index/chunk_embeddings.npy
  --metadata-out data/index/chunk_metadata.jsonl
  --manifest data/index/embedding_manifest.json
  --config configs/embedding.yaml
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}"

echo
echo "Generated embedding files:"
ls -lh data/index/chunk_embeddings.npy data/index/chunk_metadata.jsonl data/index/embedding_manifest.json
