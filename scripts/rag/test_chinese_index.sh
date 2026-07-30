#!/usr/bin/env bash
# 用途：验证中文 strict 索引结构，并执行配置中的真实医学问题检索。
# 输入：data/index/chinese/strict/ 与 configs/chinese_retrieval_tests.yaml。
# 输出：index_validation_report.json 与 retrieval_smoke_report.json。

set -euo pipefail

INDEX_DIR="${CHINESE_INDEX_DIR:-data/index/chinese/strict}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "hospital" ]]; then
    conda activate hospital
  fi
fi

python -m rag_medical.common.validate_index \
  --index-dir "${INDEX_DIR}" \
  --expected-language zh \
  --report "${INDEX_DIR}/index_validation_report.json"

python -m rag_medical.common.retrieval_smoke \
  --queries configs/chinese_retrieval_tests.yaml \
  --index-dir "${INDEX_DIR}" \
  --report "${INDEX_DIR}/retrieval_smoke_report.json"

echo
echo "Chinese index validation reports:"
ls -lh \
  "${INDEX_DIR}/index_validation_report.json" \
  "${INDEX_DIR}/retrieval_smoke_report.json"
