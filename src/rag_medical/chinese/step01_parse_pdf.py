"""
用途：逐页解析中文 CNKI PDF；原生文字不可用的页面自动使用 DeepSeek-OCR-2。

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
    - 优先使用 pdftotext；只有原生文字未通过质量门槛时才进入 OCR。
    - DeepSeek-OCR-2 是默认 OCR 后端，在独立 hospital-ocr 环境中批量运行；
      原有 Tesseract OCR 实现完整保留，可通过 --ocr-backend tesseract 使用。
    - 页面文字写出前会清理可明确识别的文章属性、页眉页脚和页码，
      不以关键词删除临床正文，避免丢失治疗结果或结论。
    - 本模块不做语义分块、embedding、FAISS 建库或 LLM 生成。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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

import numpy as np
from PIL import Image


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
    ocr_backend: str = "deepseek"
    ocr_language: str = "chi_sim+eng"
    ocr_dpi: int = 300
    ocr_psm: int = 3
    min_native_chars: int = 60
    min_native_cjk_chars: int = 20
    min_output_chars: int = 20
    max_private_use_chars: int = 24
    max_private_use_ratio: float = 0.015
    ocr_ink_threshold: int = 180
    ocr_dense_row_ratio: float = 0.40
    ocr_max_dense_band_height: int = 24


@dataclass(frozen=True)
class OcrPageResult:
    """保存 OCR 文本及图像预处理审计信息。"""

    text: str
    removed_dense_bands: int


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


def deepseek_markdown_to_text(text: str | None) -> str:
    """去掉 DeepSeek 版面识别产生的标题标记，保留标题文字和正文内容。

    现有正文清洗依赖“参考文献”等纯文本标题识别，因此这里只移除行首 Markdown
    ``#`` 标记，不改写表格、数字、剂量或医学符号，后续逻辑仍与原流程一致。
    """

    lines = [re.sub(r"^\s{0,3}#{1,6}\s*", "", line) for line in (text or "").splitlines()]
    return normalize_page_text("\n".join(lines))


# 这些规则只针对 PDF 排版产生的“文章属性行”，不根据医学关键词删正文。
ARTICLE_METADATA_PATTERN = re.compile(
    r"(?:文章编号|中图分类号|文献标识码|基金项目|收稿日期|修回日期|"
    r"作者简介|通讯作者|通信作者|作者单位|DOI\s*[:：]|ISSN\s*[:：]|CN\s*[:：])",
    flags=re.IGNORECASE,
)
JOURNAL_HEADER_PATTERN = re.compile(
    r"(?:\d{4}\s*年.*?(?:卷|期)|第?\s*\d+\s*卷.*?第?\s*\d+\s*期|"
    r"(?:医学|卫生|护理|药学|临床|外科|内科|乳腺).{0,24}(?:杂志|学报).{0,24}(?:\d{4}|卷|期))"
)
INSTITUTION_PATTERN = re.compile(
    r"(?:医院|保健院|大学|学院|研究所|研究院|疾控中心|卫生院)"
)
PAGE_NUMBER_PATTERN = re.compile(r"(?:第)?[-—–]?\d+[-—–]?(?:页)?")
REFERENCE_HEADING_PATTERN = re.compile(r"^(?:参考文献|参考资料|references)\s*[:：]?$", re.IGNORECASE)
# OCR 常把“【摘要】目的”识别为“【摘要目的”，因此只匹配“摘要”二字；
# 该规则仅在参考文献后的候选新题名附近使用，不会独立触发截断。
ABSTRACT_MARKER_PATTERN = re.compile(r"摘要")


def text_comparison_key(text: str) -> str:
    """生成文章标题比对键，仅保留中英文和数字，忽略 PDF 排版差异。"""

    return "".join(re.findall(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]", text)).lower()


def is_institution_line(line: str) -> bool:
    """判断括号中的医院/学校署名；限制为整行，避免误删正文中的机构名称。"""

    compact = re.sub(r"\s+", "", line)
    return bool(
        re.fullmatch(r"[（(].{2,120}[）)]", compact)
        and INSTITUTION_PATTERN.search(compact)
        and not re.search(r"[。！？!?；;]", compact)
    )


def is_garbled_byline(line: str) -> bool:
    """识别首页题名后的乱码作者行，如 OCR 产生的 ``a 44, x Sm``。"""

    compact = re.sub(r"\s+", "", line)
    return bool(
        re.fullmatch(r"[A-Za-z]\d{1,3}(?:[,，][A-Za-z])+(?:[A-Za-z]{1,3})?", compact)
    )


def non_content_line_category(line: str) -> str | None:
    """返回可安全删除的版式行类别；None 代表保留为正文。"""

    compact = re.sub(r"\s+", "", line)
    if not compact:
        return None
    if PAGE_NUMBER_PATTERN.fullmatch(compact):
        return "page_number"
    if ARTICLE_METADATA_PATTERN.search(compact):
        return "article_metadata"
    if is_institution_line(line):
        return "author_affiliation"
    if (
        JOURNAL_HEADER_PATTERN.search(compact)
        and not re.search(r"[。！？!?；;]", compact)
        and len(compact) <= 100
    ):
        return "journal_header"
    return None


def add_cleaning_audit(record: dict[str, Any], category: str, line_count: int) -> None:
    """追加清洗审计计数，避免页面级与文章级清洗互相覆盖结果。"""

    if line_count <= 0:
        return
    categories = Counter(record.get("cleaning_removed_categories") or {})
    categories[category] += line_count
    record["cleaning_removed_categories"] = dict(sorted(categories.items()))
    record["cleaning_removed_line_count"] = int(
        record.get("cleaning_removed_line_count") or 0
    ) + line_count


def is_probable_article_title(line: str) -> bool:
    """保守识别新文章题名：必须是无句末标点的、足够长的独立文本行。"""

    compact = re.sub(r"\s+", "", line)
    cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", compact))
    return bool(
        8 <= len(compact) <= 80
        and cjk_count >= 8
        and not re.search(r"[。！？!?；;]", compact)
        and not compact.startswith(("[", "［", "【", "(", "（"))
        and not re.match(r"^\d+(?:\.\d+)?", compact)
    )


def has_nearby_abstract(lines: list[str], start_index: int) -> bool:
    """题名后短距离内出现摘要，才确认该位置属于下一篇文章的开始。"""

    return any(
        ABSTRACT_MARKER_PATTERN.search(line)
        for line in lines[start_index + 1 : start_index + 8]
    )


def clean_article_page_text(record: dict[str, Any], *, is_first_page: bool) -> None:
    """清理单页文章属性并记录审计信息，便于人工确认未误删正文。

    对首页，若能用 PDF 文件名精确定位本篇标题，则丢弃标题前的残留内容。
    CNKI 部分 PDF 会把上一文章的末页与本篇首页放在同一 PDF 页中；该规则
    仅在标题完全匹配时触发，因此不会依据模糊关键词截断正文。
    """

    lines = [line for line in str(record.get("text") or "").splitlines() if line.strip()]
    removed_categories: Counter[str] = Counter()
    title_key = text_comparison_key(str(record.get("title") or ""))

    if is_first_page and title_key:
        for index, line in enumerate(lines):
            if text_comparison_key(line) == title_key:
                if index:
                    removed_categories["preceding_article"] += index
                lines = lines[index + 1 :]
                removed_categories["document_title"] += 1
                break

    # 乱码作者行只可能位于首页标题与摘要之间；不在表格、结果段落中应用此规则。
    if is_first_page:
        abstract_index = next(
            (index for index, line in enumerate(lines) if "摘要" in line), len(lines)
        )
        retained_prefix: list[str] = []
        for line in lines[:abstract_index]:
            if is_garbled_byline(line):
                removed_categories["garbled_byline"] += 1
                continue
            retained_prefix.append(line)
        lines = retained_prefix + lines[abstract_index:]

    kept: list[str] = []
    for line in lines:
        category = non_content_line_category(line)
        if category:
            removed_categories[category] += 1
            continue
        kept.append(line)

    record["text"] = normalize_page_text("\n".join(kept))
    record["text_char_count"] = content_char_count(record["text"])
    record["cjk_char_count"] = cjk_char_count(record["text"])
    record["cleaning_removed_line_count"] = sum(removed_categories.values())
    record["cleaning_removed_categories"] = dict(sorted(removed_categories.items()))


def clean_article_pages(page_records: list[dict[str, Any]]) -> None:
    """在页眉页脚去重前清理每页中可确定为非正文的排版信息。"""

    for index, record in enumerate(page_records):
        clean_article_page_text(record, is_first_page=index == 0)


def truncate_following_article_pages(page_records: list[dict[str, Any]]) -> None:
    """根据参考文献后的“新题名 + 摘要”组合截断同页或后续文章。

    CNKI 下载的部分 PDF 是期刊页而非单文 PDF。本函数只在本篇已出现
    “参考文献”后，并且候选题名后的 7 行内出现“摘要”时才截断，避免
    把参考文献中的普通书目条目当成新文章。
    """

    article_finished = False
    for record in page_records:
        lines = [line for line in str(record.get("text") or "").splitlines() if line.strip()]
        if article_finished:
            add_cleaning_audit(record, "following_article_page", len(lines))
            record["text"] = ""
            record["text_char_count"] = 0
            record["cjk_char_count"] = 0
            record["layout_article_boundary_detected"] = True
            continue

        reference_seen = False
        for index, line in enumerate(lines):
            if REFERENCE_HEADING_PATTERN.fullmatch(re.sub(r"\s+", "", line)):
                reference_seen = True
                continue
            if (
                reference_seen
                and is_probable_article_title(line)
                and has_nearby_abstract(lines, index)
            ):
                add_cleaning_audit(record, "following_article", len(lines) - index)
                record["text"] = normalize_page_text("\n".join(lines[:index]))
                record["text_char_count"] = content_char_count(record["text"])
                record["cjk_char_count"] = cjk_char_count(record["text"])
                record["layout_article_boundary_detected"] = True
                article_finished = True
                break


def content_char_count(text: str) -> int:
    """统计真正有信息的中英文及数字字符，排除版式空格和标点。"""

    return len(re.findall(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]", text))


def cjk_char_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def private_use_char_count(text: str) -> int:
    """统计 PDF 损坏字体映射常产生的 Unicode 私用区字符。"""

    return len(re.findall(r"[\ue000-\uf8ff]", text))


def native_text_is_usable(text: str, config: ParseConfig) -> bool:
    """判断本页能否直接采用原生文字；任一硬性条件失败即转 OCR。

    这里保持保守和简单：正文量、中文量、私用区乱码、连续重复乱码都属于
    独立否决条件，不做七项打分或多数投票，便于解释每页为何进入 OCR。
    """

    private_use_chars = private_use_char_count(text)
    visible_chars = len(re.sub(r"\s+", "", text))
    private_use_ratio = private_use_chars / max(visible_chars, 1)
    return (
        content_char_count(text) >= config.min_native_chars
        and cjk_char_count(text) >= config.min_native_cjk_chars
        and private_use_chars <= config.max_private_use_chars
        and private_use_ratio <= config.max_private_use_ratio
        and repeated_fragment_coverage(text) < 0.35
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


def dense_horizontal_bands(pixels: np.ndarray, config: ParseConfig) -> list[tuple[int, int]]:
    """定位极细且横向墨点覆盖率很高的装饰线带，不匹配普通正文行。"""

    dark_ratio_by_row = (pixels < config.ocr_ink_threshold).mean(axis=1)
    candidate_rows = np.flatnonzero(dark_ratio_by_row >= config.ocr_dense_row_ratio)
    if not len(candidate_rows):
        return []

    bands: list[tuple[int, int]] = []
    start = previous = int(candidate_rows[0])
    for row in candidate_rows[1:]:
        row = int(row)
        if row > previous + 1:
            if 2 <= previous - start + 1 <= config.ocr_max_dense_band_height:
                bands.append((start, previous))
            start = row
        previous = row
    if 2 <= previous - start + 1 <= config.ocr_max_dense_band_height:
        bands.append((start, previous))
    return bands


def preprocess_ocr_image(image_path: Path, config: ParseConfig) -> tuple[Path, int]:
    """移除非正文装饰横带后生成临时 PGM，保留图片中的文字与普通表格内容。

    规则只清理横向覆盖率至少 40%、高度不超过 24 像素的线带。该条件针对
    CNKI 期刊页的箭头分隔线，普通中文正文行无法达到此横向覆盖率。
    """

    pixels = np.array(Image.open(image_path).convert("L"))
    bands = dense_horizontal_bands(pixels, config)
    if not bands:
        return image_path, 0

    processed = pixels.copy()
    for start, end in bands:
        # 额外留出 1 像素边缘，避免抗锯齿残留再次被 OCR 误识别为字符。
        processed[max(0, start - 1) : min(processed.shape[0], end + 2), :] = 255

    output_path = image_path.with_name(f"{image_path.stem}_preprocessed.pgm")
    Image.fromarray(processed).save(output_path)
    return output_path, len(bands)


# 正常英文摘要会含有大量连接词；医学缩写作为中性 token，不计入乱码。
ENGLISH_CONTEXT_WORDS = {
    "a", "after", "and", "as", "at", "by", "clinical", "conclusion",
    "disease", "for", "from", "in", "is", "mastitis", "method", "of",
    "on", "or", "patients", "results", "study", "the", "to", "treatment",
    "was", "were", "with",
}


def repeated_fragment_coverage(text: str) -> float:
    """计算长度 1~6 的连续重复片段覆盖率，用于识别“邓马邓马……”类乱码。"""

    compact = "".join(re.findall(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]", text))
    if len(compact) < 80:
        return 0.0
    covered = [False] * len(compact)
    for unit_length in range(1, 7):
        pattern = re.compile(rf"(.{{{unit_length}}})\1{{3,}}")
        for match in pattern.finditer(compact):
            covered[match.start() : match.end()] = [True] * (match.end() - match.start())
    return sum(covered) / len(covered)


def is_pseudo_english_candidate(text: str) -> bool:
    """仅初筛以拉丁字符为主、但缺少正常英文上下文的 OCR 段落。"""

    visible = re.findall(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]", text)
    if len(visible) < 80:
        return False
    cjk_ratio = cjk_char_count(text) / len(visible)
    tokens = re.findall(r"[A-Za-z]+(?:[-_/][A-Za-z0-9]+)*", text)
    if cjk_ratio >= 0.15 or len(tokens) < 12:
        return False

    normal = 0
    suspicious = 0
    for token in tokens:
        lower = token.lower()
        # 大写缩写、字母数字混合词和希腊字母附近的医学名称均不计为无效词。
        if re.fullmatch(r"[A-Z]{2,10}", token) or re.search(r"\d|[-_/]", token):
            continue
        if lower in ENGLISH_CONTEXT_WORDS:
            normal += 1
        elif len(token) == 1 and lower not in {"a", "i"}:
            suspicious += 1
        elif not re.search(r"[aeiouy]", lower) or (
            not token.islower() and not token.istitle()
        ):
            suspicious += 1

    token_count = max(len(tokens), 1)
    symbol_ratio = len(re.findall(r"[^\w\s.,;:!?()\[\]{}%+\-/]", text)) / max(len(text), 1)
    return normal / token_count < 0.12 and (
        suspicious / token_count >= 0.35 or symbol_ratio >= 0.12
    )


def low_ocr_confidence_ratio(tsv_text: str) -> float:
    """读取 Tesseract TSV 的词级置信度，忽略空白和版面节点。"""

    confidences: list[float] = []
    for row in csv.DictReader(io.StringIO(tsv_text), delimiter="\t"):
        if not str(row.get("text") or "").strip():
            continue
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            continue
        if confidence >= 0:
            confidences.append(confidence)
    if not confidences:
        return 1.0
    return sum(value < 35 for value in confidences) / len(confidences)


def remove_severe_ocr_garbage(text: str, low_confidence_ratio: float) -> str:
    """直接删除高置信乱码段；轻微异常保留，避免误删医学正文。"""

    paragraphs = re.split(r"\n\s*\n", text)
    kept: list[str] = []
    for paragraph in paragraphs:
        repeated_garbage = repeated_fragment_coverage(paragraph) >= 0.35
        pseudo_english = (
            is_pseudo_english_candidate(paragraph) and low_confidence_ratio >= 0.60
        )
        if not repeated_garbage and not pseudo_english:
            kept.append(paragraph)
    return normalize_page_text("\n\n".join(kept))


def ocr_page(
    pdf_path: Path,
    page_number: int,
    tools: PdfTools,
    config: ParseConfig,
) -> OcrPageResult:
    """原有 Tesseract 后端：逐页渲染、清理装饰横带并识别正文。

    DeepSeek 接入后仍完整保留此实现，便于显卡不可用时显式选择
    ``--ocr-backend tesseract``，也便于后续做两种 OCR 的效果对照。
    """

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

        ocr_image_path, removed_dense_bands = preprocess_ocr_image(image_path, config)

        ocr_command = [
            tools.tesseract,
            str(ocr_image_path),
            "stdout",
            "-l",
            config.ocr_language,
            "--psm",
            str(config.ocr_psm),
            "--dpi",
            str(config.ocr_dpi),
        ]
        result = run_command(ocr_command, max(tools.timeout_seconds, 300))
        text = normalize_page_text(result.stdout)
        low_confidence_ratio = 0.0
        if any(is_pseudo_english_candidate(part) for part in re.split(r"\n\s*\n", text)):
            # 只有疑似伪英文时才追加一次 TSV 识别，避免所有 OCR 页面耗时翻倍。
            confidence_result = run_command(
                ocr_command + ["tsv"], max(tools.timeout_seconds, 300)
            )
            low_confidence_ratio = low_ocr_confidence_ratio(confidence_result.stdout)
        return OcrPageResult(
            text=remove_severe_ocr_garbage(text, low_confidence_ratio),
            removed_dense_bands=removed_dense_bands,
        )


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


def load_deepseek_ocr_results(path: Path) -> dict[tuple[str, int], str]:
    """加载已完成的 DeepSeek 页级结果，用于长批次中断后的可靠续跑。

    ``status=ok`` 且正文为空是合法结果：论文中的空白页、分隔页可能没有任何
    可检索文字。真正失败的记录仍会阻止解析，避免静默丢失有正文的页面。
    """

    results: dict[tuple[str, int], str] = {}
    failures: list[str] = []
    for record in read_jsonl(path):
        key = (str(record.get("document_id") or ""), int(record.get("page_number") or 0))
        if record.get("status") != "ok":
            failures.append(
                f"{record.get('task_id')}: {record.get('error') or 'OCR failed'}"
            )
            continue
        if not key[0] or key[1] < 1:
            failures.append(f"invalid OCR result key: {key}")
            continue
        if key in results:
            failures.append(f"duplicate OCR result: {key}")
            continue
        results[key] = deepseek_markdown_to_text(str(record.get("text") or ""))
    if failures:
        raise ValueError(f"invalid DeepSeek OCR results: {'; '.join(failures[:20])}")
    print(f"loaded_deepseek_ocr_results={len(results)} from {path}")
    return results


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


def run_deepseek_ocr_batch(
    pdf_paths: list[Path],
    *,
    cache_dir: Path,
    tools: PdfTools,
    config: ParseConfig,
    overwrite: bool,
    python_path: Path,
    worker_path: Path,
    model_path: Path,
    gpu: str,
) -> dict[tuple[str, int], str]:
    """先筛出必须 OCR 的页面，再由独立环境一次加载模型并批量识别。

    主流程只传 PDF 路径和页码。worker 在处理单页时临时渲染图片并立即释放，
    避免把数百页 PNG 同时落盘；返回值按 ``(document_id, page_number)`` 索引，
    后续仍由原解析流程按页序合并、清洗并写入缓存。
    """

    tasks: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        document_id = document_id_for_path(pdf_path)
        try:
            page_count = pdf_page_count(pdf_path, tools)
            cache_path = cache_dir / f"{document_id}.jsonl"
            if not overwrite and cache_is_complete(cache_path, document_id, page_count):
                continue
            native_pages = extract_native_pages(pdf_path, page_count, tools)
        except Exception as exc:  # noqa: BLE001 - 正式解析阶段仍会按文献记录该错误。
            print(
                f"DeepSeek OCR planning skipped {pdf_path.name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        for page_number, native_text in enumerate(native_pages, start=1):
            should_ocr = config.ocr_mode == "always" or (
                config.ocr_mode == "auto" and not native_text_is_usable(native_text, config)
            )
            if not should_ocr:
                continue
            tasks.append(
                {
                    "task_id": f"{document_id}-p{page_number:04d}",
                    "document_id": document_id,
                    "page_number": page_number,
                    "pdf_path": str(pdf_path.resolve()),
                }
            )

    if not tasks:
        print("deepseek_ocr_tasks=0 (all selected pages use native text or cache)")
        return {}

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_batch_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "tasks.jsonl"
        output_path = temp_root / "results.jsonl"
        write_jsonl(input_path, tasks)
        command = [
            str(python_path),
            str(worker_path),
            "--model-path",
            str(model_path),
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--gpu",
            gpu,
            "--pdftoppm",
            tools.pdftoppm,
            "--render-dpi",
            str(config.ocr_dpi),
        ]
        # 单页生成可能持续数分钟；总超时随任务数增长，避免沿用普通命令的短超时。
        completed = run_command(
            command,
            max(tools.timeout_seconds, 600 + len(tasks) * 600),
        )
        if completed.stdout.strip():
            print(completed.stdout.strip())
        records = read_jsonl(output_path)

    expected = {
        (str(task["document_id"]), int(task["page_number"])) for task in tasks
    }
    results: dict[tuple[str, int], str] = {}
    failures: list[str] = []
    for record in records:
        key = (str(record.get("document_id") or ""), int(record.get("page_number") or 0))
        text = deepseek_markdown_to_text(str(record.get("text") or ""))
        if record.get("status") != "ok":
            failures.append(
                f"{record.get('task_id')}: {record.get('error') or 'OCR failed'}"
            )
            continue
        if key in results:
            failures.append(f"duplicate OCR result: {key}")
            continue
        results[key] = text

    missing = sorted(expected - set(results))
    if failures or missing:
        details = "; ".join(failures + [f"missing result: {key}" for key in missing])
        raise RuntimeError(f"DeepSeek OCR batch incomplete: {details[:1500]}")
    print(f"deepseek_ocr_tasks={len(tasks)}")
    return results


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
    deepseek_ocr_results: dict[tuple[str, int], str] | None = None,
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
        ocr_backend = ""
        removed_dense_bands = 0

        if should_ocr:
            if config.ocr_backend == "deepseek":
                # DeepSeek 已由批处理 worker 一次加载模型并完成；此处只按页取回结果。
                key = (document_id, page_number)
                if deepseek_ocr_results is None or key not in deepseek_ocr_results:
                    raise RuntimeError(f"missing DeepSeek OCR result: {key}")
                text = deepseek_ocr_results[key]
                ocr_backend = "deepseek_ocr2"
            else:
                # 原有 Tesseract 路径保留，不改变其图像预处理和乱码过滤行为。
                ocr_result = ocr_page(pdf_path, page_number, tools, config)
                text = ocr_result.text
                removed_dense_bands = ocr_result.removed_dense_bands
                ocr_backend = "tesseract"
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
                "ocr_backend": ocr_backend,
                "native_text_char_count": content_char_count(native_text),
                "ocr_preprocess_removed_dense_bands": removed_dense_bands,
                "text_char_count": text_chars,
                "cjk_char_count": cjk_chars,
                "low_text": low_text,
                "text": text,
            }
        )
        print(
            f"  [{document_id}] page {page_number:>3}/{page_count}: "
            f"{method} chars={text_chars} cjk={cjk_chars} "
            f"dense_bands_removed={removed_dense_bands}"
        )

    # 先去除每页可明确识别的文章属性，再处理跨页重复的页眉页脚。
    clean_article_pages(page_records)
    truncate_following_article_pages(page_records)
    remove_repeated_marginal_lines(page_records)
    for record in page_records:
        record["low_text"] = int(record["text_char_count"]) < config.min_output_chars
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
    deepseek_ocr_results: dict[tuple[str, int], str] | None,
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
            deepseek_ocr_results=deepseek_ocr_results,
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
    parser.add_argument(
        "--ocr-backend",
        choices=["deepseek", "tesseract"],
        default="deepseek",
        help="OCR backend. The original Tesseract implementation remains available.",
    )
    parser.add_argument("--ocr-language", default="chi_sim+eng")
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--ocr-psm", type=int, default=3)
    parser.add_argument(
        "--deepseek-python",
        type=Path,
        default=Path("/home/amax/anaconda3/envs/hospital-ocr/bin/python"),
    )
    parser.add_argument(
        "--deepseek-model-path",
        type=Path,
        default=Path("models/ocr/deepseek-ocr-2"),
    )
    parser.add_argument(
        "--deepseek-worker",
        type=Path,
        default=Path(__file__).with_name("deepseek_ocr_worker.py"),
    )
    parser.add_argument(
        "--deepseek-results-in",
        type=Path,
        help="Reuse a complete DeepSeek page-result JSONL instead of rerunning OCR.",
    )
    parser.add_argument("--deepseek-gpu", default="0")
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
        if args.ocr_backend == "tesseract":
            require_binary(tools.tesseract, "Chinese OCR")
        else:
            if args.deepseek_results_in is not None and not args.deepseek_results_in.is_file():
                print(
                    f"DeepSeek OCR result file not found: {args.deepseek_results_in}",
                    file=sys.stderr,
                )
                return 2
            if args.deepseek_results_in is None and not args.deepseek_python.is_file():
                print(
                    f"hospital-ocr Python not found: {args.deepseek_python}",
                    file=sys.stderr,
                )
                return 2
            if args.deepseek_results_in is None and not args.deepseek_worker.is_file():
                print(f"DeepSeek OCR worker not found: {args.deepseek_worker}", file=sys.stderr)
                return 2
            if args.deepseek_results_in is None and not args.deepseek_model_path.is_dir():
                print(
                    f"DeepSeek OCR model not found: {args.deepseek_model_path}",
                    file=sys.stderr,
                )
                return 2

    config = ParseConfig(
        ocr_mode=args.ocr,
        ocr_backend=args.ocr_backend,
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
    deepseek_ocr_results: dict[tuple[str, int], str] | None = None
    if args.ocr != "never" and args.ocr_backend == "deepseek":
        if args.deepseek_results_in is not None:
            deepseek_ocr_results = load_deepseek_ocr_results(args.deepseek_results_in)
        else:
            deepseek_ocr_results = run_deepseek_ocr_batch(
                pdf_paths,
                cache_dir=cache_dir,
                tools=tools,
                config=config,
                overwrite=args.overwrite,
                python_path=args.deepseek_python.resolve(),
                worker_path=args.deepseek_worker.resolve(),
                model_path=args.deepseek_model_path.resolve(),
                gpu=str(args.deepseek_gpu),
            )
    tasks = [
        (
            pdf_path,
            project_dir,
            cache_dir,
            tools,
            config,
            args.overwrite,
            deepseek_ocr_results,
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
