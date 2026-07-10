#!/usr/bin/env bash
# 用途：执行严格医学筛选，生成 strict/review/excluded 三层语料。
# 输入：data/registry/processed/literature_registry.csv 与 data/articles/processed/rag_chunks.jsonl。
# 输出：data/registry/filtered/*、data/registry/filtered/view/*_view.csv/tsv 与 rag_chunks_{strict,review,excluded}.jsonl。

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

python -m rag_medical.filter_corpus "$@"

echo
echo "Generated strict corpus files:"
ls -lh \
  data/registry/filtered/literature_registry_strict.csv \
  data/registry/filtered/literature_registry_review.csv \
  data/registry/filtered/literature_registry_excluded.csv \
  data/registry/filtered/filter_report.md \
  data/registry/filtered/view/literature_registry_strict_view.csv \
  data/registry/filtered/view/literature_registry_review_view.csv \
  data/registry/filtered/view/literature_registry_excluded_view.csv \
  data/registry/filtered/view/literature_registry_strict_view.tsv \
  data/registry/filtered/view/literature_registry_review_view.tsv \
  data/registry/filtered/view/literature_registry_excluded_view.tsv \
  data/articles/processed/rag_chunks_strict.jsonl \
  data/articles/processed/rag_chunks_review.jsonl \
  data/articles/processed/rag_chunks_excluded.jsonl
