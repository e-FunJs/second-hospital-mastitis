#!/usr/bin/env bash
# 用途：初始化/进入项目目录，并激活 hospital 环境。
# 输入：无固定数据输入。
# 输出：通常不生成数据文件；主要用于服务器环境准备。

set -euo pipefail

PROJECT_NAME="second-hospital-mastitis"
BASE_DIR="/home/amax/E-FUN"
PROJECT_DIR="${BASE_DIR}/${PROJECT_NAME}"

mkdir -p "${BASE_DIR}"
cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hospital
else
  echo "conda not found; please activate hospital manually before running Python commands."
fi

python -m pip install -e .

echo "Project ready at: ${PROJECT_DIR}"
echo "Try:"
echo "bash scripts/run_pubmed_searches.sh 100"
