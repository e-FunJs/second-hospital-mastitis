"""
用途：逐页解析中文 CNKI PDF；原生文字不足的页面自动使用 Tesseract 中文 OCR。

默认输入：
    data/articles/raw/chinese/cnki_pdf/*.pdf

默认输出：
    1. data/articles/processed/chinese/article_pages.jsonl
       每行保存一页正文、页码、提取方式和可追溯来源。
    2. data/articles/processed/chinese/pdf_parse_manifest.csv
       每篇 PDF 的页数、原生提取页数、OCR 页数、空页和错误。
    3. data/articles/processed/chinese/pdf_parse_report.md
       面向人工检查的解析汇总报告。
    4. data/registry/chinese/processed/literature_registry.csv
       供中文医学筛选使用的文献级 registry。
    5. data/articles/processed/chinese/page_cache/*.jsonl
       逐篇缓存；长时间 OCR 中断后可以继续，不必重做已完成文献。

说明：
    - 原始 PDF 始终只读，不删除、不移动、不覆盖。
    - 优先使用 pdftotext；只有页面文字量不足时才渲染并调用 Tesseract。
    - 本模块不做语义分块、embedding、FAISS 建库或 LLM 生成。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PAGE_OUTPUT_NAME = "article_pages.jsonl"
MANIFEST_OUTPUT_NAME = "pdf_parse_manifest.csv"
REPORT_OUTPUT_NAME = "pdf_parse_report.md"
REGISTRY_OUTPUT_NAME = "literature_registry.csv"


@dataclass(frozen=True)
class PdfTools:
    """外部 PDF/OCR 工具及单次命令超时配置。"""

    pdfinfo: str = "pdfinfo"
    pdftotext: str = "pdftotext"
    pdftoppm: str = "pdftoppm"
    tesseract: str = "tesseract"
    timeout_seconds: int = 180


@dataclass(frozen=True)
class ParseConfig:
    """控制原生文本质量判断与 OCR 图像清晰度。"""

    ocr_mode: str = "auto"
    ocr_language: str = "chi_sim+eng"
    ocr_dpi: int = 300
    ocr_psm: int = 3
    min_native_chars: int = 60
    min_native_cjk_chars: int = 20
    min_output_chars: int = 20
    max_private_use_chars: int = 24
    max_private_use_ratio: float = 0.015


@dataclass
class DocumentResult:
    """保存单篇文献的解析统计，最终写入 manifest 和 registry。"""

    document_id: str
    title: str
    source_path: str
    page_count: int = 0
    native_pages: int = 0
    ocr_pages: int = 0
    low_text_pages: int = 0
    empty_pages: int = 0
    text_chars: int = 0
    cjk_chars: int = 0
    status: str = "parsed"
    elapsed_seconds: float = 0.0
    error: str = ""


# -----------------------------------------------------------------------------
# 文本规范化与质量判断
# -----------------------------------------------------------------------------


def normalize_line(line: str) -> str:
    """保留行结构，只压平行内空白和常见不可见字符。"""

    cleaned = line.replace("\u00a0", " ").replace("\u3000", " ").replace("\x00", "")
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def normalize_page_text(text: str | None) -> str:
    """清理单页文字，但不把全文压成一行，便于后续识别中文章节标题。"""

    lines = [normalize_line(line) for line in (text or "").replace("\r", "\n").splitlines()]
    output: list[str] = []
    blank_pending = False
    for line in lines:
        if not line:
            blank_pending = bool(output)
            continue
        if blank_pending:
            output.append("")
            blank_pending = False
        output.append(line)
    return "\n".join(output).strip()


def content_char_count(text: str) -> int:
    """统计真正有信息的中英文及数字字符，排除版式空格和标点。"""

    return len(re.findall(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]", text))


def cjk_char_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def private_use_char_count(text: str) -> int:
    """统计 PDF 损坏字体映射常产生的 Unicode 私用区字符。"""

    return len(re.findall(r"[\ue000-\uf8ff]", text))


def native_text_is_usable(text: str, config: ParseConfig) -> bool:
    private_use_chars = private_use_char_count(text)
    visible_chars = len(re.sub(r"\s+", "", text))
    private_use_ratio = private_use_chars / max(visible_chars, 1)
    return (
        content_char_count(text) >= config.min_native_chars
        and cjk_char_count(text) >= config.min_native_cjk_chars
        and private_use_chars <= config.max_private_use_chars
        and private_use_ratio <= config.max_private_use_ratio
    )


def normalized_header_key(line: str) -> str:
    """生成页眉页脚比较键；去掉页码差异后再判断是否跨页重复。"""

    compact = re.sub(r"\s+", "", line).lower()
    compact = re.sub(r"(?:第)?\d+(?:页)?", "#", compact)
    return compact


def remove_repeated_marginal_lines(page_records: list[dict[str, Any]]) -> None:
    """删除跨页高频页眉、页脚和纯页码行，正文内容保持原样。

    只检查每页前两行和后两行，并要求至少在 3 页且 30% 页面中重复，
    避免误删正文中偶然重复的治疗术语。
    """

    if len(page_records) < 3:
        return

    occurrences: Counter[str] = Counter()
    for record in page_records:
        lines = [line for line in str(record.get("text") or "").splitlines() if line.strip()]
        keys = {
            normalized_header_key(line)
            for line in lines[:2] + lines[-2:]
            if (
                2 <= len(normalized_header_key(line)) <= 80
                and not re.search(r"[。！？!?；;]", line)
            )
        }
        occurrences.update(keys)

    threshold = max(3, int(len(page_records) * 0.30 + 0.999))
    repeated = {key for key, count in occurrences.items() if count >= threshold}

    for record in page_records:
        kept: list[str] = []
        for line in str(record.get("text") or "").splitlines():
            compact = re.sub(r"\s+", "", line)
            is_page_number = bool(re.fullmatch(r"(?:第)?[-—–]?\d+[-—–]?(?:页)?", compact))
            is_repeated_margin = (
                not re.search(r"[。！？!?；;]", line)
                and normalized_header_key(line) in repeated
            )
            if is_page_number or is_repeated_margin:
                continue
            kept.append(line)
        record["text"] = normalize_page_text("\n".join(kept))
        record["text_char_count"] = content_char_count(record["text"])
        record["cjk_char_count"] = cjk_char_count(record["text"])


# -----------------------------------------------------------------------------
# 外部命令与 PDF 页面提取
# -----------------------------------------------------------------------------


def require_binary(binary: str, purpose: str) -> None:
    if shutil.which(binary) is None:
        raise FileNotFoundError(f"{purpose} command not found: {binary}")


def run_command(
    command: list[str],
    timeout_seconds: int,
    *,
    text: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=text,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)} | {stderr.strip()[:500]}"
        )
    return result


def pdf_page_count(pdf_path: Path, tools: PdfTools) -> int:
    result = run_command([tools.pdfinfo, str(pdf_path)], tools.timeout_seconds)
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"cannot read page count from pdfinfo: {pdf_path}")
    return int(match.group(1))


def split_pdftotext_pages(raw_text: str, page_count: int) -> list[str]:
    """把 pdftotext 的换页符输出校准为固定 page_count 条记录。"""

    pages = raw_text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    pages = [normalize_page_text(page) for page in pages]
    if len(pages) < page_count:
        pages.extend([""] * (page_count - len(pages)))
    if len(pages) > page_count:
        pages = pages[: page_count - 1] + ["\n".join(pages[page_count - 1 :])]
    return pages


def extract_native_pages(pdf_path: Path, page_count: int, tools: PdfTools) -> list[str]:
    result = run_command(
        [tools.pdftotext, "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        max(tools.timeout_seconds, page_count * 3),
    )
    return split_pdftotext_pages(result.stdout, page_count)


def ocr_page(pdf_path: Path, page_number: int, tools: PdfTools, config: ParseConfig) -> str:
    """逐页渲染灰度 PGM 后调用 Tesseract，临时图像离开上下文即删除。"""

    with tempfile.TemporaryDirectory(prefix="cnki_ocr_") as temp_dir:
        image_base = Path(temp_dir) / "page"
        render_command = [
            tools.pdftoppm,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            str(config.ocr_dpi),
            "-gray",
            "-singlefile",
            str(pdf_path),
            str(image_base),
        ]
        run_command(render_command, max(tools.timeout_seconds, 300), text=False)
        image_path = image_base.with_suffix(".pgm")
        if not image_path.exists():
            raise FileNotFoundError(f"pdftoppm did not produce OCR image: {image_path}")

        ocr_command = [
            tools.tesseract,
            str(image_path),
            "stdout",
            "-l",
            config.ocr_language,
            "--psm",
            str(config.ocr_psm),
            "--dpi",
            str(config.ocr_dpi),
        ]
        result = run_command(ocr_command, max(tools.timeout_seconds, 300))
        return normalize_page_text(result.stdout)


# -----------------------------------------------------------------------------
# 文献标识、缓存与输出
# -----------------------------------------------------------------------------


def document_id_for_path(pdf_path: Path) -> str:
    digest = hashlib.sha1(pdf_path.stem.encode("utf-8")).hexdigest()[:12].upper()
    return f"CNKI-{digest}"


def relative_source_path(path: Path, project_dir: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
    return records


def cache_is_complete(cache_path: Path, document_id: str, page_count: int) -> bool:
    if not cache_path.exists():
        return False
    try:
        records = read_jsonl(cache_path)
    except (OSError, ValueError):
        return False
    return (
        len(records) == page_count
        and all(record.get("document_id") == document_id for record in records)
        and [record.get("page_number") for record in records] == list(range(1, page_count + 1))
    )


def result_from_cached_records(
    document_id: str,
    title: str,
    source_path: str,
    records: list[dict[str, Any]],
) -> DocumentResult:
    methods = Counter(str(record.get("extraction_method") or "") for record in records)
    return DocumentResult(
        document_id=document_id,
        title=title,
        source_path=source_path,
        page_count=len(records),
        native_pages=methods.get("native_text", 0),
        ocr_pages=methods.get("ocr", 0),
        low_text_pages=sum(bool(record.get("low_text")) for record in records),
        empty_pages=sum(not str(record.get("text") or "").strip() for record in records),
        text_chars=sum(int(record.get("text_char_count") or 0) for record in records),
        cjk_chars=sum(int(record.get("cjk_char_count") or 0) for record in records),
        status="cached",
    )


def parse_document(
    pdf_path: Path,
    project_dir: Path,
    cache_dir: Path,
    tools: PdfTools,
    config: ParseConfig,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], DocumentResult]:
    started = time.monotonic()
    document_id = document_id_for_path(pdf_path)
    title = pdf_path.stem
    source_path = relative_source_path(pdf_path, project_dir)
    cache_path = cache_dir / f"{document_id}.jsonl"
    page_count = pdf_page_count(pdf_path, tools)

    if not overwrite and cache_is_complete(cache_path, document_id, page_count):
        cached = read_jsonl(cache_path)
        return cached, result_from_cached_records(document_id, title, source_path, cached)

    native_pages = extract_native_pages(pdf_path, page_count, tools)
    page_records: list[dict[str, Any]] = []
    result = DocumentResult(
        document_id=document_id,
        title=title,
        source_path=source_path,
        page_count=page_count,
    )

    for page_number, native_text in enumerate(native_pages, start=1):
        use_native = native_text_is_usable(native_text, config)
        should_ocr = config.ocr_mode == "always" or (
            config.ocr_mode == "auto" and not use_native
        )
        text = native_text
        method = "native_text"

        if should_ocr:
            text = ocr_page(pdf_path, page_number, tools, config)
            method = "ocr"
        elif config.ocr_mode == "never" and not use_native:
            method = "native_low_text"

        text_chars = content_char_count(text)
        cjk_chars = cjk_char_count(text)
        low_text = text_chars < config.min_output_chars
        page_records.append(
            {
                "document_id": document_id,
                "title": title,
                "journal": "",
                "year": "",
                "doi": "",
                "language": "zh",
                "source_type": "cnki_pdf",
                "source_path": source_path,
                "page_number": page_number,
                "page_count": page_count,
                "extraction_method": method,
                "native_text_char_count": content_char_count(native_text),
                "text_char_count": text_chars,
                "cjk_char_count": cjk_chars,
                "low_text": low_text,
                "text": text,
            }
        )
        print(
            f"  [{document_id}] page {page_number:>3}/{page_count}: "
            f"{method} chars={text_chars} cjk={cjk_chars}"
        )

    remove_repeated_marginal_lines(page_records)
    result.native_pages = sum(
        record["extraction_method"] in {"native_text", "native_low_text"}
        for record in page_records
    )
    result.ocr_pages = sum(record["extraction_method"] == "ocr" for record in page_records)
    result.low_text_pages = sum(bool(record["low_text"]) for record in page_records)
    result.empty_pages = sum(not str(record["text"]).strip() for record in page_records)
    result.text_chars = sum(int(record["text_char_count"]) for record in page_records)
    result.cjk_chars = sum(int(record["cjk_char_count"]) for record in page_records)
    result.status = "parsed" if result.empty_pages == 0 else "parsed_with_empty_pages"
    result.elapsed_seconds = round(time.monotonic() - started, 3)
    write_jsonl(cache_path, page_records)
    return page_records, result


def parse_document_safely(
    pdf_path: Path,
    project_dir: Path,
    cache_dir: Path,
    tools: PdfTools,
    config: ParseConfig,
    overwrite: bool,
    index: int,
    total: int,
) -> DocumentResult:
    """解析单篇 PDF，并把异常转换为可写入清单的失败记录。"""

    print(f"[{index}/{total}] {pdf_path.name}")
    started = time.monotonic()
    try:
        _, result = parse_document(
            pdf_path=pdf_path,
            project_dir=project_dir,
            cache_dir=cache_dir,
            tools=tools,
            config=config,
            overwrite=overwrite,
        )
        return result
    except Exception as exc:  # noqa: BLE001 - 单篇失败不能中断其余文献。
        result = DocumentResult(
            document_id=document_id_for_path(pdf_path),
            title=pdf_path.stem,
            source_path=relative_source_path(pdf_path, project_dir),
            status="failed",
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"  [{result.document_id}] failed: {result.error}", file=sys.stderr)
        return result


def write_manifest(path: Path, results: list[DocumentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(DocumentResult("", "", "")).keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def registry_record(result: DocumentResult) -> dict[str, Any]:
    return {
        "document_id": result.document_id,
        "title": result.title,
        "language": "zh",
        "source_type": "cnki_pdf",
        "source_path": result.source_path,
        "page_count": result.page_count,
        "native_pages": result.native_pages,
        "ocr_pages": result.ocr_pages,
        "empty_pages": result.empty_pages,
        "parse_status": result.status,
        "journal": "",
        "year": "",
        "authors": "",
        "doi": "",
        "keywords": "",
        "abstract": "",
    }


def write_registry(path: Path, results: list[DocumentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [registry_record(result) for result in results if not result.status.startswith("failed")]
    fields = list(rows[0].keys()) if rows else list(registry_record(DocumentResult("", "", "")).keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, results: list[DocumentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = [result for result in results if result.status.startswith("failed")]
    total_pages = sum(result.page_count for result in results)
    lines = [
        "# 中文 CNKI PDF 解析报告",
        "",
        f"- 创建时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 文献数：{len(results)}",
        f"- 总页数：{total_pages}",
        f"- 原生文字页：{sum(result.native_pages for result in results)}",
        f"- OCR 页：{sum(result.ocr_pages for result in results)}",
        f"- 低文字量页：{sum(result.low_text_pages for result in results)}",
        f"- 空页：{sum(result.empty_pages for result in results)}",
        f"- 失败文献：{len(failed)}",
        "",
        "## 失败文献",
        "",
    ]
    if failed:
        lines.extend(f"- {result.title}: {result.error}" for result in failed)
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `native_text` 表示直接从 PDF 文字层提取。",
            "- `ocr` 表示该页文字层不足，使用 Tesseract `chi_sim+eng` 识别。",
            "- 低文字量页不等于文件损坏，可能是封面、图表页或空白页，需要抽查。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_cache(
    pdf_paths: list[Path],
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[DocumentResult]]:
    all_pages: list[dict[str, Any]] = []
    results: list[DocumentResult] = []
    for pdf_path in pdf_paths:
        document_id = document_id_for_path(pdf_path)
        cache_path = cache_dir / f"{document_id}.jsonl"
        if not cache_path.exists():
            continue
        records = read_jsonl(cache_path)
        all_pages.extend(records)
        results.append(
            result_from_cached_records(
                document_id,
                pdf_path.stem,
                str(records[0].get("source_path") or "") if records else "",
                records,
            )
        )
    return all_pages, results


# -----------------------------------------------------------------------------
# CLI 主流程
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Chinese CNKI PDFs with native text extraction and page-level OCR fallback."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/articles/raw/chinese/cnki_pdf"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/articles/processed/chinese"),
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("data/registry/chinese/processed"),
    )
    parser.add_argument("--ocr", choices=["auto", "never", "always"], default="auto")
    parser.add_argument("--ocr-language", default="chi_sim+eng")
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--ocr-psm", type=int, default=3)
    parser.add_argument("--min-native-chars", type=int, default=60)
    parser.add_argument("--min-native-cjk-chars", type=int, default=20)
    parser.add_argument("--min-output-chars", type=int, default=20)
    parser.add_argument("--max-private-use-chars", type=int, default=24)
    parser.add_argument("--max-private-use-ratio", type=float, default=0.015)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--select", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of PDFs parsed concurrently. Each PDF keeps an independent cache.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.pdf_dir.exists():
        print(f"PDF directory not found: {args.pdf_dir}", file=sys.stderr)
        return 2

    tools = PdfTools(timeout_seconds=args.timeout)
    require_binary(tools.pdfinfo, "PDF page count")
    require_binary(tools.pdftotext, "PDF native text extraction")
    if args.ocr != "never":
        require_binary(tools.pdftoppm, "PDF page rendering")
        require_binary(tools.tesseract, "Chinese OCR")

    config = ParseConfig(
        ocr_mode=args.ocr,
        ocr_language=args.ocr_language,
        ocr_dpi=args.ocr_dpi,
        ocr_psm=args.ocr_psm,
        min_native_chars=args.min_native_chars,
        min_native_cjk_chars=args.min_native_cjk_chars,
        min_output_chars=args.min_output_chars,
        max_private_use_chars=args.max_private_use_chars,
        max_private_use_ratio=args.max_private_use_ratio,
    )
    pdf_paths = sorted(args.pdf_dir.glob("*.pdf"), key=lambda path: path.name.casefold())
    if args.select:
        pdf_paths = [
            path for path in pdf_paths if any(selector in path.name for selector in args.select)
        ]
    if args.limit is not None:
        pdf_paths = pdf_paths[: args.limit]
    if not pdf_paths:
        print(f"no PDF files selected in: {args.pdf_dir}", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2

    cache_dir = args.processed_dir / "page_cache"
    project_dir = Path.cwd()
    tasks = [
        (
            pdf_path,
            project_dir,
            cache_dir,
            tools,
            config,
            args.overwrite,
            index,
            len(pdf_paths),
        )
        for index, pdf_path in enumerate(pdf_paths, start=1)
    ]

    # 每个任务只写自己的 document_id 缓存文件，不共享可变状态；executor.map
    # 返回顺序与输入顺序一致，因此并行不会改变最终 JSONL、CSV 的文献顺序。
    if args.workers == 1:
        results = [parse_document_safely(*task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(lambda task: parse_document_safely(*task), tasks))

    all_pages: list[dict[str, Any]] = []
    successful_results: list[DocumentResult] = []
    for pdf_path, result in zip(pdf_paths, results):
        if result.status.startswith("failed"):
            continue
        cache_path = cache_dir / f"{result.document_id}.jsonl"
        records = read_jsonl(cache_path)
        all_pages.extend(records)
        successful_results.append(result)

    write_jsonl(args.processed_dir / PAGE_OUTPUT_NAME, all_pages)
    write_manifest(args.processed_dir / MANIFEST_OUTPUT_NAME, results)
    write_report(args.processed_dir / REPORT_OUTPUT_NAME, results)
    write_registry(args.registry_dir / REGISTRY_OUTPUT_NAME, successful_results)

    failed_count = sum(result.status.startswith("failed") for result in results)
    print(f"documents={len(results)}")
    print(f"pages={len(all_pages)}")
    print(f"ocr_pages={sum(result.ocr_pages for result in results)}")
    print(f"failed_documents={failed_count}")
    print(f"pages_out={args.processed_dir / PAGE_OUTPUT_NAME}")
    print(f"manifest={args.processed_dir / MANIFEST_OUTPUT_NAME}")
    print(f"registry={args.registry_dir / REGISTRY_OUTPUT_NAME}")
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
