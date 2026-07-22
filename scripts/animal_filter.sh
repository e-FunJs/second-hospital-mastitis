#!/usr/bin/env bash
# 用途：在第一轮 strict/review/excluded 筛选之后，运行 BGE 动物乳腺炎语义复核。
# 前置步骤：bash scripts/filter_corpus.sh
# 输入：第一轮 registry/chunk 文件、broad rag_chunks.jsonl 与 models/bge/bge-m3。
# 输出：
#   data/registry/filtered/semantic/（最终 registry、锚点、审计表和报告）
#   data/articles/processed/semantic/（最终 strict/review/excluded chunk）

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

python -m rag_medical.animal_filter "$@"

echo
echo "Generated animal semantic filter files:"
ls -lh \
  data/registry/filtered/semantic/literature_registry_strict.csv \
  data/registry/filtered/semantic/literature_registry_review.csv \
  data/registry/filtered/semantic/literature_registry_excluded.csv \
  data/registry/filtered/semantic/animal_audit.csv \
  data/registry/filtered/semantic/anchors.jsonl \
  data/registry/filtered/semantic/filter_report.md \
  data/articles/processed/semantic/rag_chunks_strict.jsonl \
  data/articles/processed/semantic/rag_chunks_review.jsonl \
  data/articles/processed/semantic/rag_chunks_excluded.jsonl
