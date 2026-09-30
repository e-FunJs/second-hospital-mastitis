"""对固定页面任务运行项目原有的 Tesseract OCR 后端。

输入：
    默认读取既有中文 OCR 页面任务 ``ocr_tasks.jsonl``；也可通过
    ``--tasks`` 指定其他 JSONL。每条任务至少需要 ``task_id``、
    ``document_id``、``page_number`` 和 ``pdf_path``。

输出：
    1. ``tesseract_<时间戳>.jsonl``：逐页 OCR 文本、状态和耗时。
    2. ``tesseract_manifest_<时间戳>.json``：本次任务数量、成功/失败
       数量及 OCR 参数。

说明：
    本文件直接复用 ``step01_parse_pdf.py`` 中保留的原 Tesseract 实现，
    包括原有图像预处理和 OCR 专用乱码过滤；不会执行正文清洗、语义分块、
    医学筛选、embedding 或检索。后续 DeepSeek OCR 脚本应使用相同任务清单，
    以 ``task_id`` 和 ``page_number`` 逐页配对。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from rag_medical.chinese.step01_parse_pdf import ParseConfig, PdfTools, ocr_page


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TASKS = Path(
    "cache_output_test/full_cn_rebuild_20260917_2011/ocr_tasks.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/rag/answers/OCR")


def project_path(path: Path) -> Path:
    """把命令行中的相对路径统一解释为项目根目录下的路径。"""

    return path if path.is_absolute() else PROJECT_DIR / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取并校验页面任务 JSONL。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"task at {path}:{line_number} must be an object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """先写临时文件再替换，避免中断时留下半份正式结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def write_json(path: Path, record: dict[str, Any]) -> None:
    """原子写入本次运行 manifest。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    """确保任务可以稳定地与未来 DeepSeek 结果逐页配对。"""

    required_fields = {"task_id", "document_id", "page_number", "pdf_path"}
    task_ids: list[str] = []
    for index, task in enumerate(tasks, start=1):
        missing = sorted(required_fields.difference(task))
        if missing:
            raise ValueError(f"task {index} missing fields: {', '.join(missing)}")
        task_ids.append(str(task["task_id"]))
        if int(task["page_number"]) < 1:
            raise ValueError(f"task {index} has invalid page_number")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values must be unique")


def process_task(
    task: dict[str, Any],
    tools: PdfTools,
    config: ParseConfig,
) -> dict[str, Any]:
    """处理单页；单页失败会记录原因，不中断其他页面。"""

    started = time.perf_counter()
    pdf_path = project_path(Path(str(task["pdf_path"]))).resolve()
    record: dict[str, Any] = {
        "task_id": str(task["task_id"]),
        "document_id": str(task["document_id"]),
        "page_number": int(task["page_number"]),
        "pdf_path": str(pdf_path),
        "extraction_method": "ocr",
        "ocr_backend": "tesseract",
        "status": "ok",
        "text": "",
        "removed_dense_bands": 0,
        "elapsed_seconds": 0.0,
        "error": "",
    }
    try:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        result = ocr_page(
            pdf_path=pdf_path,
            page_number=record["page_number"],
            tools=tools,
            config=config,
        )
        record["text"] = result.text
        record["removed_dense_bands"] = result.removed_dense_bands
    except Exception as exc:  # noqa: BLE001 - 单页错误需要落盘，便于批量任务继续。
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def run_tasks(
    tasks: list[dict[str, Any]],
    tools: PdfTools,
    config: ParseConfig,
    workers: int,
) -> list[dict[str, Any]]:
    """并发执行独立页面，同时保持最终 JSONL 与输入任务顺序一致。"""

    records: list[dict[str, Any] | None] = [None] * len(tasks)
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            records[index - 1] = process_task(task, tools, config)
            print(f"[{index}/{len(tasks)}] {task['task_id']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_indexes = {
                executor.submit(process_task, task, tools, config): index
                for index, task in enumerate(tasks)
            }
            for completed, future in enumerate(as_completed(future_indexes), start=1):
                index = future_indexes[future]
                records[index] = future.result()
                print(
                    f"[{completed}/{len(tasks)}] {tasks[index]['task_id']}",
                    flush=True,
                )
    return [record for record in records if record is not None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the retained Tesseract backend on fixed Chinese PDF pages."
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--limit", type=int, help="Only process the first N tasks.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ocr-language", default="chi_sim+eng")
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--ocr-psm", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks_path = project_path(args.tasks)
    output_dir = project_path(args.output_dir)
    if not tasks_path.is_file():
        print(f"task file not found: {tasks_path}", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2
    for binary in ("pdftoppm", "tesseract"):
        if shutil.which(binary) is None:
            print(f"required command not found: {binary}", file=sys.stderr)
            return 2

    output_path = output_dir / f"tesseract_{args.run_id}.jsonl"
    manifest_path = output_dir / f"tesseract_manifest_{args.run_id}.json"
    if not args.overwrite and (output_path.exists() or manifest_path.exists()):
        print(
            f"output already exists for run-id {args.run_id}; use --overwrite",
            file=sys.stderr,
        )
        return 2

    try:
        tasks = read_jsonl(tasks_path)
        validate_tasks(tasks)
    except (OSError, TypeError, ValueError) as exc:
        print(f"invalid task file: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no OCR tasks found", file=sys.stderr)
        return 2

    tools = PdfTools(timeout_seconds=args.timeout)
    config = ParseConfig(
        ocr_mode="always",
        ocr_backend="tesseract",
        ocr_language=args.ocr_language,
        ocr_dpi=args.ocr_dpi,
        ocr_psm=args.ocr_psm,
    )
    started = time.perf_counter()
    records = run_tasks(tasks, tools, config, workers=args.workers)
    elapsed_seconds = round(time.perf_counter() - started, 3)
    success_count = sum(record["status"] == "ok" for record in records)
    failed_count = len(records) - success_count

    write_jsonl(output_path, records)
    write_json(
        manifest_path,
        {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": args.run_id,
            "backend": "tesseract",
            "tasks_path": str(tasks_path.resolve()),
            "output_path": str(output_path.resolve()),
            "task_count": len(records),
            "success_count": success_count,
            "failed_count": failed_count,
            "elapsed_seconds": elapsed_seconds,
            "workers": args.workers,
            "ocr_language": args.ocr_language,
            "ocr_dpi": args.ocr_dpi,
            "ocr_psm": args.ocr_psm,
            "backend_stage": "existing_tesseract_backend_before_article_cleaning",
        },
    )

    print(f"tasks={len(records)}")
    print(f"success={success_count}")
    print(f"failed={failed_count}")
    print(f"output={output_path}")
    print(f"manifest={manifest_path}")
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
