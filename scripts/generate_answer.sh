#!/usr/bin/env bash
# 用途：调用本地 LLM，根据 RAG prompt 生成回答。
# 输入：data/rag/answers/*_prompt.txt。
# 输出：同名前缀的 *_answer.md 与 *_answer.json。

set -euo pipefail

PROMPT_PATH="${1:-}"
MAX_NEW_TOKENS="${2:-}"

if [[ -z "${PROMPT_PATH}" ]]; then
  echo "Usage: bash scripts/generate_answer.sh path/to/prompt.txt [max_new_tokens]" >&2
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

CMD=(
  python -m rag_medical.generate_answer
  --prompt "${PROMPT_PATH}"
  --config configs/llm.yaml
)

if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  CMD+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi

"${CMD[@]}"
