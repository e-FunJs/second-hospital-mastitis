#!/usr/bin/env bash
# 用途：解析中文 CNKI PDF；文字层不足的页面自动执行 Tesseract 中文 OCR。
# 输入：data/articles/raw/chinese/cnki_pdf/*.pdf。
# 输出：data/articles/processed/chinese/ 下的页级 JSONL、缓存、清单和报告，
#       以及 data/registry/chinese/processed/literature_registry.csv。

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

python -m rag_medical.chinese.parse_pdf \
  --workers "${CNKI_PARSE_WORKERS:-8}" \
  "$@"

echo
echo "Generated Chinese PDF parse files:"
ls -lh \
  data/articles/processed/chinese/article_pages.jsonl \
  data/articles/processed/chinese/pdf_parse_manifest.csv \
  data/articles/processed/chinese/pdf_parse_report.md \
  data/registry/chinese/processed/literature_registry.csv
