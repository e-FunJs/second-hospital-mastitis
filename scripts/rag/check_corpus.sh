#!/usr/bin/env bash
# 用途：检查中文 strict chunk 是否满足通用 embedding 输入契约。
# 输入：默认 data/articles/processed/chinese/filtered/rag_chunks_strict.jsonl。
# 输出：common_readiness_report.json；不加载 BGE，不生成向量或索引。

set -euo pipefail

INPUT_PATH="${1:-data/articles/processed/chinese/filtered/rag_chunks_strict.jsonl}"
REPORT_PATH="${2:-data/articles/processed/chinese/common_readiness_report.json}"

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
  --report "${REPORT_PATH}"
