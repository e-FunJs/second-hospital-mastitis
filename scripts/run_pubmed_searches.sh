#!/usr/bin/env bash
# 用途：批量运行 configs/queries.yaml 中全部 PubMed 检索 query。
# 输入：configs/queries.yaml。
# 输出：data/registry/raw/pubmed_<query_key>.csv。

set -euo pipefail

MAX_RESULTS="${1:-100}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

mkdir -p data/registry/raw

python -m rag_medical.search_pubmed \
  --all \
  --max-results "${MAX_RESULTS}" \
  --out data/registry/raw/pubmed.csv

echo
echo "Generated registry files:"
find data/registry/raw -maxdepth 1 -type f -name 'pubmed_*.csv' -printf '%f\n' | sort

echo
echo "Next step:"
echo "Run scripts/merge_registry.sh, then fetch PMC Open Access full text."
