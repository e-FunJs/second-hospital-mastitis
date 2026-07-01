#!/usr/bin/env bash
set -euo pipefail

QUERY="${1:-}"
TOP_K="${2:-8}"

if [[ -z "${QUERY}" ]]; then
  echo "Usage: bash scripts/search_chunks.sh \"query text\" [top_k]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

python -m rag_medical.search_chunks "${QUERY}" --top-k "${TOP_K}"
