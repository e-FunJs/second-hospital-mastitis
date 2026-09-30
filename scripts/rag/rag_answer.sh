#!/usr/bin/env bash
# 用途：固定检索中文 strict 与英文 strict 文献，并生成双语 evidence。
# 输入：用户问题；提问语言仅决定哪边使用原问题、哪边使用 Qwen 翻译。
# 输出：默认 data/rag/answers/bilingual/。

set -euo pipefail

QUESTION="${1:-}"
TOP_K="${2:-8}"
OUTPUT_DIR="${RAG_OUTPUT_DIR:-data/rag/answers/bilingual}"

if [[ -z "${QUESTION}" ]]; then
  echo "Usage: bash scripts/rag/rag_answer.sh \"question text\" [top_k]" >&2
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

python -m rag_medical.common.step08_rag_answer "${QUESTION}" \
  --top-k "${TOP_K}" \
  --chinese-index "${RAG_CHINESE_INDEX:-data/index/chinese/strict/faiss.index}" \
  --chinese-metadata "${RAG_CHINESE_METADATA:-data/index/chinese/strict/chunk_metadata.jsonl}" \
  --english-index "${RAG_ENGLISH_INDEX:-data/index/english/strict/faiss.index}" \
  --english-metadata "${RAG_ENGLISH_METADATA:-data/index/english/strict/chunk_metadata.jsonl}" \
  --output-dir "${OUTPUT_DIR}"
