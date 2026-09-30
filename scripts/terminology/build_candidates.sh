#!/usr/bin/env bash
# 输入：data/terminology/observations.jsonl；输出：聚合后的 candidates.jsonl。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/amax/anaconda3/envs/hospital/bin/python}"

cd "$PROJECT_ROOT"
PYTHONPATH=src "$PYTHON_BIN" -m rag_medical.terminology.candidates \
  --observations data/terminology/observations.jsonl \
  --output data/terminology/candidates.jsonl
