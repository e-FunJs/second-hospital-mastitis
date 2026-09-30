#!/usr/bin/env bash
# 一键执行固定页面 OCR 对比：准备任务 -> Tesseract -> DeepSeek-OCR-2 -> 评估。
# 默认只处理人工标注涉及的页面，不会对全部中文 PDF 重新 OCR。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/amax/anaconda3/bin/conda}"

RUN_ID="$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="${PROJECT_DIR}/data/rag/answers/OCR"
GOLD_PAGES="${PROJECT_DIR}/data/rag/OCR_Eval_Metirc/Eval_Full_Page.jsonl"
GOLD_SNIPPETS="${PROJECT_DIR}/data/rag/OCR_Eval_Metirc/Eval_Text.jsonl"
MASTER_TASKS="${PROJECT_DIR}/cache_output_test/full_cn_rebuild_20260917_2011/ocr_tasks.jsonl"
GPU="0"
DRY_RUN=0
OVERWRITE=0

usage() {
    cat <<'EOF'
用法：
  bash src/rag_medical/OCR/run_ocr_eval.sh [选项]

选项：
  --run-id ID             本次实验标识，默认精确到分钟。
  --output-root PATH      实验输出根目录。
  --gold-pages PATH       整页人工标注 JSONL。
  --gold-snippets PATH    局部人工标注 JSONL。
  --master-tasks PATH     原始 OCR 总任务清单。
  --gpu ID                DeepSeek OCR 使用的 GPU，默认 0。
  --overwrite             允许覆盖同一 run-id 的已有结果。
  --dry-run               只显示将执行的命令，不运行 OCR。
  -h, --help              显示帮助。

结果目录：<output-root>/eval_<run-id>/
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --gold-pages) GOLD_PAGES="$2"; shift 2 ;;
        --gold-snippets) GOLD_SNIPPETS="$2"; shift 2 ;;
        --master-tasks) MASTER_TASKS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
    esac
done

EVALUATOR="${SCRIPT_DIR}/step03_evaluate_ocr.py"
TESSERACT_SCRIPT="${SCRIPT_DIR}/step01_tesseract_ocr.py"
DEEPSEEK_SCRIPT="${SCRIPT_DIR}/step02_deepseek_ocr.py"
RUN_DIR="${OUTPUT_ROOT}/eval_${RUN_ID}"
TASKS_FILE="${RUN_DIR}/eval_tasks.jsonl"
TESSERACT_RESULTS="${RUN_DIR}/tesseract_${RUN_ID}.jsonl"
DEEPSEEK_RESULTS="${RUN_DIR}/deepseek_ocr2_${RUN_ID}.jsonl"

for required in "$CONDA_EXE" "$EVALUATOR" "$TESSERACT_SCRIPT" \
                "$DEEPSEEK_SCRIPT" "$GOLD_PAGES" "$GOLD_SNIPPETS" "$MASTER_TASKS"; do
    if [[ ! -e "$required" ]]; then
        echo "缺少必要文件：$required" >&2
        exit 2
    fi
done

COMMON_ENV=(env "PYTHONPATH=${PROJECT_DIR}/src")
PREPARE_CMD=(
    "${COMMON_ENV[@]}" "$CONDA_EXE" run --no-capture-output -n hospital
    python "$EVALUATOR" prepare
    --gold-pages "$GOLD_PAGES"
    --gold-snippets "$GOLD_SNIPPETS"
    --master-tasks "$MASTER_TASKS"
    --output "$TASKS_FILE"
)
TESSERACT_CMD=(
    "${COMMON_ENV[@]}" "$CONDA_EXE" run --no-capture-output -n hospital
    python "$TESSERACT_SCRIPT"
    --tasks "$TASKS_FILE"
    --output-dir "$RUN_DIR"
    --run-id "$RUN_ID"
)
DEEPSEEK_CMD=(
    "${COMMON_ENV[@]}" "$CONDA_EXE" run --no-capture-output -n hospital-ocr
    python "$DEEPSEEK_SCRIPT"
    --tasks "$TASKS_FILE"
    --output-dir "$RUN_DIR"
    --run-id "$RUN_ID"
    --gpu "$GPU"
)
EVALUATE_CMD=(
    "${COMMON_ENV[@]}" "$CONDA_EXE" run --no-capture-output -n hospital
    python "$EVALUATOR" evaluate
    --gold-pages "$GOLD_PAGES"
    --gold-snippets "$GOLD_SNIPPETS"
    --tesseract-results "$TESSERACT_RESULTS"
    --deepseek-results "$DEEPSEEK_RESULTS"
    --output-dir "$RUN_DIR"
)

if [[ "$OVERWRITE" -eq 1 ]]; then
    PREPARE_CMD+=(--overwrite)
    TESSERACT_CMD+=(--overwrite)
    DEEPSEEK_CMD+=(--overwrite)
    EVALUATE_CMD+=(--overwrite)
fi

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

echo "OCR 对比实验：${RUN_ID}"
echo "结果目录：${RUN_DIR}"
echo "将依次执行："
print_command "${PREPARE_CMD[@]}"
print_command "${TESSERACT_CMD[@]}"
print_command "${DEEPSEEK_CMD[@]}"
print_command "${EVALUATE_CMD[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "dry-run 完成：未执行 OCR。"
    exit 0
fi

mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"
"${PREPARE_CMD[@]}"
"${TESSERACT_CMD[@]}"
"${DEEPSEEK_CMD[@]}"
"${EVALUATE_CMD[@]}"

echo "评估完成：${RUN_DIR}/report.md"
