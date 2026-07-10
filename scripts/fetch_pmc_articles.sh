#!/usr/bin/env bash
# 用途：根据 registry 中的 PMCID 下载 PMC XML 全文。
# 输入：data/registry/processed/literature_registry.csv。
# 输出：data/articles/raw/pmc_xml/*.xml 与 pmc_download_manifest.csv。

set -euo pipefail

LIMIT="${1:-}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

mkdir -p data/articles/raw/pmc_xml data/articles/processed

CMD=(
  python -m rag_medical.fetch_pmc_articles
  --registry data/registry/processed/literature_registry.csv
  --out-dir data/articles/raw/pmc_xml
  --manifest data/articles/processed/pmc_download_manifest.csv
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}"

echo
echo "Downloaded PMC XML files:"
find data/articles/raw/pmc_xml -maxdepth 1 -type f -name 'PMC*.xml' | wc -l

echo
echo "Manifest:"
ls -lh data/articles/processed/pmc_download_manifest.csv

