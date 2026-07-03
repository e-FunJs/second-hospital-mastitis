#!/usr/bin/env bash
set -euo pipefail

ANSWER_PATH="${1:-}"
MODE="${2:-rules}"

if [[ -z "${ANSWER_PATH}" ]]; then
  echo "Usage: bash scripts/evaluate_answer.sh path/to/answer.json [rules|judge|all]" >&2
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

python -m rag_medical.evaluate_answer \
  --answer "${ANSWER_PATH}" \
  --mode "${MODE}"
