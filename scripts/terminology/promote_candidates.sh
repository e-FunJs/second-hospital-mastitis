#!/usr/bin/env bash
# 默认只生成晋升提案；显式传入 --apply 才会更新正式 aliases.jsonl。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/amax/anaconda3/envs/hospital/bin/python}"

cd "$PROJECT_ROOT"
PYTHONPATH=src "$PYTHON_BIN" -m rag_medical.terminology.promote_candidates \
  --config configs/terminology.yaml "$@"
