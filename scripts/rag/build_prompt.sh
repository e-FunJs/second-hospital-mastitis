#!/usr/bin/env bash
# 用途：把 step09 清洗证据转换为可独立审计和优化的 Qwen 报告 prompt。
# 输入：一个 *_cleaned_evidence.json；输出：默认同目录同名前缀的 *_prompt.txt。

set -euo pipefail

EVIDENCE_PATH="${1:-}"
OUTPUT_PATH="${2:-}"

if [[ -z "${EVIDENCE_PATH}" ]]; then
  echo "Usage: bash scripts/rag/build_prompt.sh path/to/cleaned_evidence.json [output.txt]" >&2
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

CMD=(
  python -m rag_medical.common.step10_build_prompt
  --evidence "${EVIDENCE_PATH}"
)

if [[ -n "${OUTPUT_PATH}" ]]; then
  CMD+=(--output "${OUTPUT_PATH}")
fi

"${CMD[@]}"
