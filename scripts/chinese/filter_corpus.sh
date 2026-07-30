#!/usr/bin/env bash
# 用途：对中文文献和 chunk 执行 strict/review/excluded 医学规则筛选。
# 输入：中文 literature_registry.csv 与 article_chunks.jsonl。
# 输出：data/registry/chinese/filtered 与
#       data/articles/processed/chinese/filtered 下的分层语料和报告。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

python -m rag_medical.chinese.filter_corpus "$@"

echo
echo "Generated Chinese medical filter files:"
ls -lh \
  data/registry/chinese/filtered/literature_registry_strict.csv \
  data/registry/chinese/filtered/literature_registry_review.csv \
  data/registry/chinese/filtered/literature_registry_excluded.csv \
  data/registry/chinese/filtered/filter_report.md \
  data/articles/processed/chinese/filtered/rag_chunks_strict.jsonl \
  data/articles/processed/chinese/filtered/rag_chunks_review.jsonl \
  data/articles/processed/chinese/filtered/rag_chunks_excluded.jsonl
