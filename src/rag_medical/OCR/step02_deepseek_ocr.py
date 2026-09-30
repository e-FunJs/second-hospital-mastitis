"""对固定页面任务运行 DeepSeek-OCR-2 后端。

输入：
    默认读取与 Tesseract 对照实验相同的 ``ocr_tasks.jsonl``；可使用
    ``--tasks`` 指定其他页面任务。模型路径默认是项目中的
    ``models/ocr/deepseek-ocr-2``。

输出：
    1. ``deepseek_ocr2_<时间戳>.jsonl``：逐页 OCR 文本和运行状态。
    2. ``deepseek_ocr2_manifest_<时间戳>.json``：任务数量、成功/失败
       数量、总耗时和模型参数。

说明：
    本文件运行在 ``hospital-ocr`` 环境中，并复用已经通过全量验证的
    ``deepseek_ocr_worker.py``。页面图片、模型 result.mmd 和限量任务文件
    均使用临时目录，任务结束后自动删除；不会执行正文清洗或 RAG 后续流程。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TASKS = Path(
    "cache_output_test/full_cn_rebuild_20260917_2011/ocr_tasks.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/rag/answers/OCR")
DEFAULT_MODEL_PATH = Path("models/ocr/deepseek-ocr-2")
DEFAULT_WORKER_PATH = Path("src/rag_medical/chinese/deepseek_ocr_worker.py")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    """使用临时文件替换，避免正式结果只写入一部分。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    required = {"task_id", "document_id", "page_number", "pdf_path"}
    task_ids: list[str] = []
    for index, task in enumerate(tasks, start=1):
        missing = sorted(required.difference(task))
        if missing:
            raise ValueError(f"task {index} missing fields: {', '.join(missing)}")
        task_ids.append(str(task["task_id"]))
        if int(task["page_number"]) < 1:
            raise ValueError(f"task {index} has invalid page_number")
        pdf_path = project_path(Path(str(task["pdf_path"])))
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values must be unique")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR-2 on fixed Chinese PDF page tasks."
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER_PATH)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--limit", type=int, help="Only process the first N tasks.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks_path = project_path(args.tasks)
    output_dir = project_path(args.output_dir)
    model_path = project_path(args.model_path)
    worker_path = project_path(args.worker)

    for label, path in (
        ("task file", tasks_path),
        ("model directory", model_path),
        ("DeepSeek worker", worker_path),
    ):
        exists = path.is_dir() if label == "model directory" else path.is_file()
        if not exists:
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2
    if args.limit is not None and args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2

    output_path = output_dir / f"deepseek_ocr2_{args.run_id}.jsonl"
    manifest_path = output_dir / f"deepseek_ocr2_manifest_{args.run_id}.json"
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

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_wrapper_") as temp_dir:
        temp_root = Path(temp_dir)
        worker_tasks = temp_root / "tasks.jsonl"
        worker_output = temp_root / "results.jsonl"
        write_jsonl(worker_tasks, tasks)

        # 使用当前 hospital-ocr Python 启动已有 worker，模型依赖不会进入主环境。
        command = [
            sys.executable,
            str(worker_path),
            "--model-path",
            str(model_path),
            "--input-jsonl",
            str(worker_tasks),
            "--output-jsonl",
            str(worker_output),
            "--gpu",
            str(args.gpu),
            "--base-size",
            str(args.base_size),
            "--image-size",
            str(args.image_size),
            "--render-dpi",
            str(args.render_dpi),
        ]
        completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
        if not worker_output.is_file():
            print(
                f"DeepSeek worker failed without producing results (exit={completed.returncode})",
                file=sys.stderr,
            )
            return completed.returncode or 2
        records = read_jsonl(worker_output)

    expected_ids = [str(task["task_id"]) for task in tasks]
    actual_ids = [str(record.get("task_id", "")) for record in records]
    if actual_ids != expected_ids:
        print("DeepSeek result task order/count does not match input", file=sys.stderr)
        return 2

    elapsed_seconds = round(time.perf_counter() - started, 3)
    success_count = sum(record.get("status") == "ok" for record in records)
    failed_count = len(records) - success_count
    write_jsonl(output_path, records)
    write_json(
        manifest_path,
        {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": args.run_id,
            "backend": "deepseek_ocr2",
            "tasks_path": str(tasks_path.resolve()),
            "output_path": str(output_path.resolve()),
            "model_path": str(model_path.resolve()),
            "task_count": len(records),
            "success_count": success_count,
            "failed_count": failed_count,
            "elapsed_seconds": elapsed_seconds,
            "gpu": str(args.gpu),
            "base_size": args.base_size,
            "image_size": args.image_size,
            "render_dpi": args.render_dpi,
            "backend_stage": "deepseek_result_mmd_before_article_cleaning",
        },
    )

    print(f"tasks={len(records)}")
    print(f"success={success_count}")
    print(f"failed={failed_count}")
    print(f"output={output_path}")
    print(f"manifest={manifest_path}")
    return 0 if completed.returncode == 0 and failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
