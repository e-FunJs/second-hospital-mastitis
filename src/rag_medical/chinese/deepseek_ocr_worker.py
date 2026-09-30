"""
用途：在独立的 hospital-ocr 环境中加载一次 DeepSeek-OCR-2，批量识别页面图片。

输入：
    --input-jsonl 指向任务清单。每行必须包含 task_id、document_id、page_number，
    并提供 image_path，或提供 pdf_path 由 worker 临时渲染该页。

输出：
    --output-jsonl 指向识别结果。每行保留任务标识、OCR 正文、后端名称和错误信息，
    供中文 PDF 主解析器按 document_id + page_number 合并回原文页序。

说明：
    - 模型加载和 infer 参数遵循 DeepSeek-OCR-2 官方 Hugging Face 示例。
    - 本文件只负责 OCR，不做正文清洗、文章截断、分块或检索。
    - 单页失败会写入失败记录；全部任务结束后返回非零退出码，避免静默漏页。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取任务清单，并在真正加载大模型前完成格式校验。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
            missing = {
                key
                for key in ("task_id", "document_id", "page_number")
                if record.get(key) in (None, "")
            }
            if missing:
                raise ValueError(
                    f"missing fields in {path} line {line_number}: {sorted(missing)}"
                )
            if not record.get("image_path") and not record.get("pdf_path"):
                raise ValueError(
                    f"missing image_path or pdf_path in {path} line {line_number}"
                )
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """原子写出 OCR 结果，避免中途中断留下看似完整的文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def load_official_model(model_path: Path):
    """按官方示例加载本地 DeepSeek-OCR-2 模型和 tokenizer。"""

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR-2 requires an available NVIDIA CUDA device")

    # 以下加载方式与官方演示保持一致；local_files_only 防止服务器意外联网补文件。
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    return tokenizer, model


def infer_one_page(
    *,
    model: Any,
    tokenizer: Any,
    image_path: Path,
    output_dir: Path,
    base_size: int,
    image_size: int,
) -> str:
    """调用官方 infer 接口识别一页，并读取其 result.mmd 正文。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    # 官方 infer 会把整页识别正文同时打印到 stdout。批量任务若由主流程捕获，
    # 会重复占用大量内存；这里只抑制打印，result.mmd 和返回结果均不受影响。
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        result = model.infer(
            tokenizer,
            prompt=OCR_PROMPT,
            image_file=str(image_path),
            output_path=str(output_dir),
            base_size=base_size,
            image_size=image_size,
            crop_mode=True,
            save_results=True,
        )

    result_path = output_dir / "result.mmd"
    if result_path.exists():
        return result_path.read_text(encoding="utf-8").strip()
    if isinstance(result, str) and result.strip():
        return result.strip()
    raise RuntimeError(f"DeepSeek OCR produced no result.mmd for {image_path}")


def render_pdf_page(
    *,
    pdf_path: Path,
    page_number: int,
    work_dir: Path,
    pdftoppm: str,
    render_dpi: int,
) -> Path:
    """按页临时渲染 PDF；图片用完即随临时目录删除，不长期占用磁盘。"""

    work_dir.mkdir(parents=True, exist_ok=True)
    image_base = work_dir / "page"
    command = [
        pdftoppm,
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-r",
        str(render_dpi),
        "-cropbox",
        "-aa",
        "yes",
        "-aaVector",
        "yes",
        "-png",
        "-singlefile",
        str(pdf_path),
        str(image_base),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"pdftoppm failed ({result.returncode}): {result.stderr.strip()[:500]}"
        )
    image_path = image_base.with_suffix(".png")
    if not image_path.is_file():
        raise FileNotFoundError(f"pdftoppm did not produce page image: {image_path}")
    return image_path


def run_tasks(
    tasks: list[dict[str, Any]],
    *,
    model_path: Path,
    base_size: int,
    image_size: int,
    pdftoppm: str,
    render_dpi: int,
) -> tuple[list[dict[str, Any]], int]:
    """模型只加载一次，随后顺序处理全部页面，降低显存和启动开销。"""

    tokenizer, model = load_official_model(model_path)
    records: list[dict[str, Any]] = []
    failed_count = 0

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_results_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, task in enumerate(tasks, start=1):
            task_id = str(task["task_id"])
            source_name = Path(str(task.get("image_path") or task.get("pdf_path"))).name
            print(f"[{index}/{len(tasks)}] OCR {task_id}: {source_name}", flush=True)
            record = {
                "task_id": task_id,
                "document_id": str(task["document_id"]),
                "page_number": int(task["page_number"]),
                "image_path": str(task.get("image_path") or ""),
                "pdf_path": str(task.get("pdf_path") or ""),
                "extraction_method": "ocr",
                "ocr_backend": "deepseek_ocr2",
                "status": "ok",
                "text": "",
                "error": "",
            }
            try:
                task_dir = temp_root / task_id
                if task.get("image_path"):
                    image_path = Path(str(task["image_path"]))
                else:
                    pdf_path = Path(str(task["pdf_path"]))
                    if not pdf_path.is_file():
                        raise FileNotFoundError(f"PDF not found: {pdf_path}")
                    image_path = render_pdf_page(
                        pdf_path=pdf_path,
                        page_number=int(task["page_number"]),
                        work_dir=task_dir / "render",
                        pdftoppm=pdftoppm,
                        render_dpi=render_dpi,
                    )
                if not image_path.is_file():
                    raise FileNotFoundError(f"page image not found: {image_path}")
                record["text"] = infer_one_page(
                    model=model,
                    tokenizer=tokenizer,
                    image_path=image_path,
                    output_dir=task_dir / "result",
                    base_size=base_size,
                    image_size=image_size,
                )
            except Exception as exc:  # noqa: BLE001 - 记录单页失败后继续其余任务。
                failed_count += 1
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  failed: {record['error']}", file=sys.stderr, flush=True)
            records.append(record)

    # 显式释放模型引用，方便该 worker 被长期流程重复调用时及时归还显存。
    del model
    del tokenizer
    return records, failed_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch OCR page images with a local DeepSeek-OCR-2 model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--render-dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model_path.is_dir():
        print(f"model directory not found: {args.model_path}", file=sys.stderr)
        return 2
    if not args.input_jsonl.is_file():
        print(f"input JSONL not found: {args.input_jsonl}", file=sys.stderr)
        return 2
    if shutil.which(args.pdftoppm) is None:
        print(f"pdftoppm command not found: {args.pdftoppm}", file=sys.stderr)
        return 2

    tasks = read_jsonl(args.input_jsonl)
    if not tasks:
        write_jsonl(args.output_jsonl, [])
        print("no OCR tasks")
        return 0
    task_ids = [str(task["task_id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        print("task_id values must be unique", file=sys.stderr)
        return 2

    # 必须在导入 torch 前限定设备；worker 内部看到的目标卡始终编号为 cuda:0。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    records, failed_count = run_tasks(
        tasks,
        model_path=args.model_path.resolve(),
        base_size=args.base_size,
        image_size=args.image_size,
        pdftoppm=args.pdftoppm,
        render_dpi=args.render_dpi,
    )
    write_jsonl(args.output_jsonl, records)
    print(f"tasks={len(records)}")
    print(f"failed={failed_count}")
    print(f"output={args.output_jsonl}")
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
