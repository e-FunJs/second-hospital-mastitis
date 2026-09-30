"""
用途：把中文 PDF 页级文本按章节和语义边界切分成可检索的 RAG chunk。

默认输入：
    data/articles/processed/chinese/article_pages.jsonl

默认输出：
    1. data/articles/processed/chinese/article_chunks.jsonl
       中文语义检索子块；同时保存完整父块和同父块相邻子块关系。
    2. data/articles/processed/chinese/chunk_manifest.csv
       每篇文献、每个章节的句数、语义边界数、chunk 数和错误。

说明：
    - 中文章节标题是硬边界，不把摘要、方法、结果、讨论强行拼在一起。
    - BGE-M3 只用于比较相邻句群的语义相似度，不生成或改写文章内容。
    - 章节内先以平滑后的语义相似度低谷形成父块，再仅在完整句子边界拆成检索子块。
    - 本模块不生成最终知识库 embedding，也不创建 FAISS 索引。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ChunkConfig:
    """控制章节内语义父块与其检索子块的长度边界。"""

    window_size: int = 2
    smoothing_window: int = 3
    similarity_percentile: float = 20.0
    valley_margin: float = 0.02
    min_sentences: int = 2
    max_sentences: int = 12
    min_chars: int = 180
    max_chars: int = 900
    max_raw_chars: int = 4000
    include_references: bool = False


SECTION_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:中文)?摘要$"), "摘要"),
    (re.compile(r"^(?:关键词|关键字)$"), "关键词"),
    (re.compile(r"^(?:引言|前言|绪论)$"), "引言"),
    (
        re.compile(
            r"^(?:资料[与和]方法|材料[与和]方法|对象[与和]方法|病例[与和]方法|"
            r"临床资料[与和]方法|仪器[与和]方法|研究方法|实验方法)$"
        ),
        "资料与方法",
    ),
    (re.compile(r"^(?:一般资料|临床资料|研究对象)$"), "临床资料"),
    (re.compile(r"^(?:治疗方法|治疗方案|方法)$"), "治疗方法"),
    (re.compile(r"^(?:诊断标准|诊断)$"), "诊断"),
    (re.compile(r"^(?:纳入标准|排除标准|纳入与排除标准)$"), "纳入与排除标准"),
    (re.compile(r"^(?:观察指标|评价指标|疗效评价)$"), "评价指标"),
    (re.compile(r"^(?:疗效标准|疗效判定标准)$"), "疗效标准"),
    (re.compile(r"^(?:统计学方法|统计学处理)$"), "统计学方法"),
    (re.compile(r"^(?:结果|研究结果)$"), "结果"),
    (re.compile(r"^(?:不良反应|并发症)$"), "不良反应"),
    (re.compile(r"^(?:随访|预后)$"), "随访与预后"),
    (re.compile(r"^(?:讨论|分析与讨论)$"), "讨论"),
    (re.compile(r"^(?:结论|小结)$"), "结论"),
    (re.compile(r"^(?:参考文献|主要参考文献)$"), "参考文献"),
)

HEADING_START_WORD = re.compile(
    r"^(?:摘要|关键词|引言|前言|绪论|资料|材料|对象|病例|临床|一般|研究|"
    r"仪器|方法|治疗|方案|纳入|排除|观察|评价|指标|统计|结果|讨论|结论|小结|"
    r"病因|诊断|随访|不良反应|疗效|机制|分析)"
)

# Conservative layout cleanup rules for CNKI/PDF text before chunking.
# Chinese terms are written as Unicode escapes to keep this source stable across shells.
SECTION_ALIASES = (
    (re.compile(r"^(?:\u6458\u8981)$"), "\u6458\u8981"),
    (re.compile(r"^(?:\u5173\u952e\u8bcd|\u5173\u952e\u5b57)$"), "\u5173\u952e\u8bcd"),
    (re.compile(r"^(?:\u5f15\u8a00|\u524d\u8a00|\u7eea\u8bba)$"), "\u5f15\u8a00"),
    (re.compile(r"^(?:\u8d44\u6599[\u4e0e\u548c]\u65b9\u6cd5|\u6750\u6599[\u4e0e\u548c]\u65b9\u6cd5|\u5bf9\u8c61[\u4e0e\u548c]\u65b9\u6cd5|\u65b9\u6cd5|\u4e34\u5e8a\u8d44\u6599[\u4e0e\u548c]\u65b9\u6cd5|\u7814\u7a76\u65b9\u6cd5|\u5b9e\u9a8c\u65b9\u6cd5)$"), "\u8d44\u6599\u4e0e\u65b9\u6cd5"),
    (re.compile(r"^(?:\u4e00\u822c\u8d44\u6599|\u4e34\u5e8a\u8d44\u6599|\u7814\u7a76\u5bf9\u8c61)$"), "\u4e34\u5e8a\u8d44\u6599"),
    (re.compile(r"^(?:\u6cbb\u7597\u65b9\u6cd5|\u6cbb\u7597\u65b9\u6848|\u65b9\u6cd5)$"), "\u6cbb\u7597\u65b9\u6cd5"),
    (re.compile(r"^(?:\u8bca\u65ad\u6807\u51c6|\u8bca\u65ad)$"), "\u8bca\u65ad\u6807\u51c6"),
    (re.compile(r"^(?:\u7eb3\u5165\u6807\u51c6|\u6392\u9664\u6807\u51c6|\u7eb3\u5165\u53ca\u6392\u9664\u6807\u51c6)$"), "\u7eb3\u5165\u53ca\u6392\u9664\u6807\u51c6"),
    (re.compile(r"^(?:\u89c2\u5bdf\u6307\u6807|\u8bc4\u4ef7\u6307\u6807|\u7597\u6548\u8bc4\u4ef7)$"), "\u89c2\u5bdf\u6307\u6807"),
    (re.compile(r"^(?:\u7597\u6548\u6807\u51c6|\u7597\u6548\u5224\u5b9a\u6807\u51c6)$"), "\u7597\u6548\u6807\u51c6"),
    (re.compile(r"^(?:\u7edf\u8ba1\u5b66\u65b9\u6cd5|\u7edf\u8ba1\u5b66\u5904\u7406)$"), "\u7edf\u8ba1\u5b66\u65b9\u6cd5"),
    (re.compile(r"^(?:\u7ed3\u679c|\u7814\u7a76\u7ed3\u679c)$"), "\u7ed3\u679c"),
    (re.compile(r"^(?:\u4e0d\u826f\u53cd\u5e94|\u5e76\u53d1\u75c7)$"), "\u4e0d\u826f\u53cd\u5e94"),
    (re.compile(r"^(?:\u968f\u8bbf|\u9884\u540e)$"), "\u968f\u8bbf\u4e0e\u9884\u540e"),
    (re.compile(r"^(?:\u8ba8\u8bba|\u5206\u6790\u4e0e\u8ba8\u8bba)$"), "\u8ba8\u8bba"),
    (re.compile(r"^(?:\u7ed3\u8bba|\u5c0f\u7ed3)$"), "\u7ed3\u8bba"),
    (re.compile(r"^(?:\u53c2\u8003\u6587\u732e|\u4e3b\u8981\u53c2\u8003\u6587\u732e)$"), "\u53c2\u8003\u6587\u732e"),
)

HEADING_START_WORD = re.compile(
    r"^(?:\u6458\u8981|\u5173\u952e\u8bcd|\u5173\u952e\u5b57|\u5f15\u8a00|\u524d\u8a00|\u7eea\u8bba|\u6750\u6599|\u8d44\u6599|\u65b9\u6cd5|\u5bf9\u8c61|\u4e34\u5e8a|\u4e00\u822c|\u7814\u7a76|"
    r"\u6cbb\u7597|\u8bca\u65ad|\u7eb3\u5165|\u6392\u9664|\u89c2\u5bdf|\u8bc4\u4ef7|\u6307\u6807|\u7edf\u8ba1|\u7ed3\u679c|\u8ba8\u8bba|\u5c0f\u7ed3|"
    r"\u7ed3\u8bba|\u53c2\u8003|\u81f4\u8c22|\u4e0d\u826f\u53cd\u5e94|\u7597\u6548|\u968f\u8bbf|\u9884\u540e)"
)

LAYOUT_NOISE_LINE_PATTERNS = (
    re.compile(r"^(?:\u6587\u7ae0\u7f16\u53f7|\u4e2d\u56fe\u5206\u7c7b\u53f7|\u6587\u732e\u6807\u8bc6\u7801|\u6536\u7a3f\u65e5\u671f|\u57fa\u91d1\u9879\u76ee|\u4f5c\u8005\u7b80\u4ecb|\u901a\u4fe1\u4f5c\u8005|\u4f5c\u8005\u5355\u4f4d|DOI|doi)[:\uff1a]"),
    re.compile(r"^(?:\u56fe|\u8868)\s*\d+[\s:\uff1a.\uff0e\u3001]"),
    re.compile(r"^\u7b2c\s*\d+\s*\u5377\s*\u7b2c\s*\d+\s*\u671f"),
    re.compile(r"^[\u4e00-\u9fffA-Za-z\u00b7\u300a\u300b()\uff08\uff09\s]{2,40}\s*\d{4}\s*\u5e74\s*\u7b2c\s*\d+\s*\u5377\s*\u7b2c\s*\d+\s*\u671f"),
)

INLINE_SECTION_MARKER = re.compile(
    r"(?<!\d)(?:^|\s)(?P<number>\d{1,2})\s*"
    r"(?P<title>\u6458\u8981|\u5173\u952e\u8bcd|\u5f15\u8a00|\u524d\u8a00|\u8d44\u6599\u4e0e\u65b9\u6cd5|\u6750\u6599\u4e0e\u65b9\u6cd5|\u4e34\u5e8a\u8d44\u6599|\u6cbb\u7597\u65b9\u6cd5|"
    r"\u89c2\u5bdf\u6307\u6807|\u7597\u6548\u6807\u51c6|\u7edf\u8ba1\u5b66\u65b9\u6cd5|\u7ed3\u679c|\u8ba8\u8bba|\u7ed3\u8bba|\u53c2\u8003\u6587\u732e)"
    r"(?=\s|[:\uff1a]|[\u3400-\u4dbf\u4e00-\u9fff])"
)



# -----------------------------------------------------------------------------
# 中文文本与章节识别
# -----------------------------------------------------------------------------


def normalize_space(text: str | None) -> str:
    value = (text or "").replace("\u00a0", " ").replace("\u3000", " ")
    value = re.sub(r"[ \t\r\n]+", " ", value).strip()
    # OCR 和 PDF 文字层常在每个汉字之间插空格，这些空格没有语言意义。
    value = re.sub(
        r"(?<=[\u3400-\u4dbf\u4e00-\u9fff])\s+(?=[\u3400-\u4dbf\u4e00-\u9fff])",
        "",
        value,
    )
    return value


def join_text_lines(lines: list[str]) -> str:
    """合并 PDF 视觉行；中文接中文不加空格，中英文边界保留一个空格。"""

    output = ""
    for raw_line in lines:
        line = normalize_space(raw_line)
        if not line:
            continue
        if not output:
            output = line
            continue
        left = output[-1]
        right = line[0]
        if re.match(r"[\u3400-\u4dbf\u4e00-\u9fff]", left) and re.match(
            r"[\u3400-\u4dbf\u4e00-\u9fff]",
            right,
        ):
            output += line
        elif left == "-":
            output = output[:-1] + line
        else:
            output += " " + line
    return normalize_space(output)


def clean_body_line(line: str) -> str:
    """Remove high-confidence CNKI/PDF layout noise before semantic chunking."""

    cleaned = normalize_space(line)
    if not cleaned:
        return ""
    compact = re.sub(r"\s+", "", cleaned)
    for pattern in LAYOUT_NOISE_LINE_PATTERNS:
        if pattern.search(cleaned) or pattern.search(compact):
            return ""

    # Drop reference markers such as [13-14] / Chinese full-width variants.
    cleaned = re.sub(r"[\[\uff3b\u3010]\s*\d+(?:\s*[-,\uff0c\uff0d\u2014]\s*\d+)*\s*[\]\uff3d\u3011]", "", cleaned)
    cleaned = re.sub(r"\s+([\uff0c\u3002\uff1b;\uff1a:\u3001])", r"\1", cleaned)
    cleaned = re.sub(r"([\uff08(])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([\uff09)])", r"\1", cleaned)
    return normalize_space(cleaned)


def split_embedded_heading_line(line: str) -> list[tuple[str | None, str]]:
    """Split inline section markers caused by CNKI two-column extraction noise."""

    cleaned = clean_body_line(line)
    if not cleaned:
        return []

    heading = canonical_section_heading(cleaned)
    if heading:
        return [(heading, "")]

    match = INLINE_SECTION_MARKER.search(cleaned)
    if not match or match.start() == 0:
        return [(None, cleaned)]

    before = clean_body_line(cleaned[: match.start()])
    heading = canonical_section_heading(match.group("title"))
    after = clean_body_line(cleaned[match.end() :])

    parts: list[tuple[str | None, str]] = []
    if before:
        parts.append((None, before))
    if heading:
        parts.append((heading, ""))
    if after:
        after = re.sub(r"^\u6837(?=\u4e00\u4e2a\u95ee\u9898|\u95ee\u9898|\uff0c|,)", "", after)
        if after:
            parts.append((None, after))
    return parts


def split_complete_sentences(text: str) -> tuple[list[str], str]:
    """返回已闭合句子和待续尾句，绝不把半句当作检索单元。

    PDF 页尾常会在“方”这类字符处截断；尾句先由调用方带到同章节下一页，
    而不是立刻写入 chunk。若后续仍找不到句末标点，宁可不进入 RAG 证据，
    也不制造看似相关但无法解释的断条。
    """

    normalized = normalize_space(text)
    if not normalized:
        return [], ""

    pieces = re.findall(r".+?(?:[。！？!?；;]+[”’\"）)\]]*|$)", normalized)
    tail = ""
    if pieces and not re.search(r"[。！？!?；;]+[”’\"）)\]]*$", pieces[-1]):
        tail = normalize_space(pieces.pop())

    sentences: list[str] = []
    for piece in pieces:
        sentence = normalize_space(piece)
        if not sentence:
            continue
        sentences.append(sentence)

    output: list[str] = []
    for sentence in sentences:
        output.extend(split_long_sentence(sentence))
    return output, tail


def split_sentences(text: str) -> list[str]:
    """兼容旧调用：只返回完整句子；跨页续接由 section_sentence_items 负责。"""

    sentences, _ = split_complete_sentences(text)
    return sentences


def split_long_sentence(
    sentence: str,
    max_chars: int = 1800,
    max_units: int = 850,
) -> list[str]:
    """保留完整句子；异常超长句由后续子块层单独容纳，不按字符硬切。

    这里的输入已经是句末标点切出的句子。若在此再次按逗号或任意长度截断，
    会再次出现“根据治疗”这类断条；因此宁可形成一个超长单句子块，也不破坏
    原始句意。保留参数是为了兼容旧调用和后续审计。
    """

    _ = max_chars, max_units
    normalized = normalize_space(sentence)
    return [normalized] if normalized else []


def text_unit_count(text: str) -> int:
    """中文按单字、英文和数字按连续 token 计数，用于长度保护。"""

    return len(
        re.findall(
            r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?",
            text,
        )
    )


def title_after_numbered_prefix(compact: str) -> str | None:
    """返回明确章节编号后的标题；普通病例数字和年份不视为编号。"""

    chapter_match = re.match(r"^第[一二三四五六七八九十]+[章节]\s*", compact)
    if chapter_match:
        return compact[chapter_match.end() :].strip()

    chinese_match = re.match(r"^[一二三四五六七八九十]+、\s*", compact)
    if chinese_match:
        return compact[chinese_match.end() :].strip()

    # 小数点形式必须优先整体匹配；否则“1.3 纳入标准”会被错误拆成
    # 编号“1.”和标题“3 纳入标准”。
    arabic_match = re.match(
        r"^(?P<number>\d{1,2}(?:\.\d+){1,3})(?P<separator>[、．]|\s+)",
        compact,
    ) or re.match(
        r"^(?P<number>\d{1,2})(?P<separator>[、．.]|\s+)",
        compact,
    )
    if arabic_match:
        title = compact[arabic_match.end() :].strip()
        separator = arabic_match.group("separator")
        # “30 例患者”是正文，不是第 30 节；仅用空格分隔时，标题必须以
        # 常见章节词开头。带“.”或“、”的编号可保留自定义小节名称。
        if separator.isspace() and not HEADING_START_WORD.match(title):
            return None
        return title

    direct_match = re.match(r"^\d{1,2}(?:\.\d+){0,3}", compact)
    if direct_match:
        title = compact[direct_match.end() :].strip()
        if HEADING_START_WORD.match(title):
            return title
    return None


def canonical_section_heading(line: str) -> str | None:
    compact = normalize_space(line)
    compact = re.sub(r"^[\s【\[]+|[\s】\]]+$", "", compact)
    numbered_title = title_after_numbered_prefix(compact)
    alias_candidate = numbered_title if numbered_title is not None else compact
    for pattern, canonical in SECTION_ALIASES:
        if pattern.fullmatch(alias_candidate):
            return canonical

    # 自定义编号条目、目录页和表格行极易伪装成章节标题。它们不丢弃，
    # 而是作为正文交给 BGE 语义边界及最大长度规则继续处理。
    return None


def section_sentence_items(
    page_records: list[dict[str, Any]],
    include_references: bool = False,
) -> OrderedDict[str, list[dict[str, Any]]]:
    """把页级文本转换为章节内句子，并保留每句来自哪一页。"""

    sections: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    current_section = "正文"
    skip_references = False
    carried_tail = ""
    carried_section = ""

    for page_record in sorted(page_records, key=lambda row: int(row.get("page_number") or 0)):
        page_number = int(page_record.get("page_number") or 0)
        extraction_method = str(page_record.get("extraction_method") or "")
        document_title = re.sub(r"\s+", "", normalize_space(str(page_record.get("title") or "")))
        buffer: list[str] = []

        def flush_buffer() -> None:
            nonlocal carried_tail, carried_section
            if not buffer or skip_references:
                buffer.clear()
                return
            text = join_text_lines(buffer)
            # 只有同一章节的尾句才允许与本页首句相连，章节边界永远不跨越。
            if carried_tail and carried_section == current_section:
                text = join_text_lines([carried_tail, text])
                carried_tail = ""
                carried_section = ""
            sentences, carried_tail = split_complete_sentences(text)
            carried_section = current_section if carried_tail else ""
            for sentence in sentences:
                sections.setdefault(current_section, []).append(
                    {
                        "text": sentence,
                        "page_number": page_number,
                        "extraction_method": extraction_method,
                    }
                )
            buffer.clear()

        for raw_line in str(page_record.get("text") or "").splitlines():
            raw_cleaned = clean_body_line(raw_line)
            if not raw_cleaned:
                continue
            is_document_title = (
                bool(document_title)
                and re.sub(r"\s+", "", raw_cleaned).strip("\u300a\u300b[]") == document_title
            )
            line_parts = [(None, raw_cleaned)] if is_document_title else split_embedded_heading_line(raw_cleaned)
            for heading, line in line_parts:
                if heading:
                    flush_buffer()
                    # 一个未闭合尾句若恰好撞上新章节，不能跨章节强行拼接。
                    carried_tail = ""
                    carried_section = ""
                    current_section = heading
                    skip_references = heading == "\u53c2\u8003\u6587\u732e" and not include_references
                    continue
                if line and not skip_references:
                    buffer.append(line)
        flush_buffer()

    return OrderedDict((section, items) for section, items in sections.items() if items)


# -----------------------------------------------------------------------------
# 语义边界
# -----------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    if q >= 100:
        return ordered[-1]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def is_local_valley(scores: list[float], index: int, margin: float) -> bool:
    score = scores[index]
    left = scores[index - 1] if index > 0 else float("inf")
    right = scores[index + 1] if index < len(scores) - 1 else float("inf")
    return score <= left - margin and score <= right - margin


def smooth_similarities(scores: list[float], window_size: int) -> list[float]:
    """用中心加权的小窗口平滑相似度，削弱噪声但保留真实的单点话题低谷。

    普通移动平均或中位数可能把“前后相似、中间突降”的真实章节内转折抹平。
    这里让当前 gap 权重最高、两侧只作轻度校正；默认窗口为 3 时权重为 1:2:1。
    """

    if window_size <= 1 or len(scores) <= 2:
        return list(scores)
    radius = window_size // 2
    smoothed: list[float] = []
    for index in range(len(scores)):
        left = max(0, index - radius)
        right = min(len(scores), index + radius + 1)
        numerator = 0.0
        denominator = 0.0
        for neighbor in range(left, right):
            weight = float(radius + 1 - abs(neighbor - index))
            numerator += scores[neighbor] * weight
            denominator += weight
        smoothed.append(numerator / denominator)
    return smoothed


def choose_semantic_boundaries(
    sentences: list[str],
    similarities: list[float],
    config: ChunkConfig,
) -> list[int]:
    if len(sentences) <= 1:
        return []
    if len(similarities) != len(sentences) - 1:
        raise ValueError("similarities length must equal len(sentences) - 1")

    # 章节已经在 section_sentence_items 中作为硬边界；BGE 只判断章节内部。
    # 先平滑再取低分位数，避免单个噪声句把父块错误切碎。
    smoothed = smooth_similarities(similarities, config.smoothing_window)
    threshold = percentile(smoothed, config.similarity_percentile)
    boundaries: list[int] = []
    chunk_start = 0
    for score_index, score in enumerate(smoothed):
        boundary_after = score_index + 1
        sentence_count = boundary_after - chunk_start
        remaining = len(sentences) - boundary_after

        # 长度上限也只能在完整句子边界切；若切完会使尾部不足最小句数，
        # 留给后面的“短父块”处理，不制造一个必然过短的尾块。
        if sentence_count >= config.max_sentences and remaining >= config.min_sentences:
            boundaries.append(boundary_after)
            chunk_start = boundary_after
            continue
        if sentence_count < config.min_sentences or remaining < config.min_sentences:
            continue
        if score <= threshold and is_local_valley(smoothed, score_index, config.valley_margin):
            boundaries.append(boundary_after)
            chunk_start = boundary_after
    return boundaries


def cosine_similarity(left: Any, right: Any) -> float:
    numerator = float(sum(float(a) * float(b) for a, b in zip(left, right)))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def window_texts_for_gaps(sentences: list[str], window_size: int) -> list[str]:
    windows: list[str] = []
    for gap_index in range(len(sentences) - 1):
        left_start = max(0, gap_index - window_size + 1)
        right_end = min(len(sentences), gap_index + 1 + window_size)
        windows.append("".join(sentences[left_start : gap_index + 1]))
        windows.append("".join(sentences[gap_index + 1 : right_end]))
    return windows


def compute_gap_similarities(
    sentences: list[str],
    encode: Callable[[list[str]], Any],
    window_size: int,
) -> list[float]:
    if len(sentences) <= 1:
        return []
    embeddings = encode(window_texts_for_gaps(sentences, window_size))
    return [
        cosine_similarity(embeddings[index], embeddings[index + 1])
        for index in range(0, len(embeddings), 2)
    ]


# -----------------------------------------------------------------------------
# chunk 构造
# -----------------------------------------------------------------------------


def ranges_from_boundaries(
    sentence_count: int,
    boundaries: Iterable[int],
) -> list[tuple[int, int]]:
    cleaned = sorted({boundary for boundary in boundaries if 0 < boundary < sentence_count})
    ranges: list[tuple[int, int]] = []
    start = 0
    for boundary in cleaned:
        ranges.append((start, boundary))
        start = boundary
    ranges.append((start, sentence_count))
    return ranges


def range_text(start: int, end: int, items: list[dict[str, Any]]) -> str:
    return join_text_lines([str(item["text"]) for item in items[start:end]])


def range_is_short(start: int, end: int, items: list[dict[str, Any]], config: ChunkConfig) -> bool:
    """判断父块是否信息量过低；单句但内容充分时允许其独立存在。"""

    return (end - start) <= config.min_sentences and text_unit_count(range_text(start, end, items)) < config.min_chars


def merge_short_parent_ranges(
    ranges: list[tuple[int, int]],
    items: list[dict[str, Any]],
    smoothed_similarities: list[float],
    config: ChunkConfig,
) -> list[tuple[int, int]]:
    """保守处理过短父块，只允许与同章节、语义连续的邻块合并。

    不能简单把短结论塞给前一段：若边界本身是低谷，两个块通常是不同主题。
    因此只选择边界相似度不低于章节中位数的一侧；没有合适邻居时保留短块，
    并由 parent_is_short 标记，供后续人工审计。
    """

    if len(ranges) <= 1:
        return ranges

    merged = list(ranges)
    semantic_floor = percentile(smoothed_similarities, 50.0) if smoothed_similarities else 0.0
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, (start, end) in enumerate(merged):
            if not range_is_short(start, end, items, config):
                continue

            candidates: list[tuple[float, int, tuple[int, int]]] = []
            if index > 0:
                left_start, left_end = merged[index - 1]
                # 调用方可能只提供边界而不提供相似度；此时按语义门槛回退，
                # 仍由下方的句数和长度上限决定是否允许相邻短块合并。
                score = (
                    smoothed_similarities[start - 1]
                    if 0 <= start - 1 < len(smoothed_similarities)
                    else semantic_floor
                )
                candidates.append((score, index - 1, (left_start, end)))
            if index + 1 < len(merged):
                right_start, right_end = merged[index + 1]
                score = (
                    smoothed_similarities[end - 1]
                    if 0 <= end - 1 < len(smoothed_similarities)
                    else semantic_floor
                )
                candidates.append((score, index, (start, right_end)))

            # 只跨越非低谷边界，并要求合并后仍受父块硬上限保护。
            for score, replace_index, merged_range in sorted(candidates, reverse=True):
                merged_text = range_text(*merged_range, items)
                if (
                    score >= semantic_floor
                    and merged_range[1] - merged_range[0] <= config.max_sentences
                    and len(merged_text) <= config.max_raw_chars
                ):
                    merged[replace_index : replace_index + 2] = [merged_range]
                    changed = True
                    break
            if changed:
                break
    return merged


def split_oversized_range(
    start: int,
    end: int,
    items: list[dict[str, Any]],
    config: ChunkConfig,
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    current_start = start
    current_units = 0
    current_raw_chars = 0
    for index in range(start, end):
        sentence = normalize_space(str(items[index]["text"]))
        sentence_units = text_unit_count(sentence)
        separator_chars = 1 if current_raw_chars and sentence else 0
        if (
            index > current_start
            and (
                current_units + sentence_units > config.max_chars
                or current_raw_chars + separator_chars + len(sentence) > config.max_raw_chars
            )
        ):
            output.append((current_start, index))
            current_start = index
            current_units = sentence_units
            current_raw_chars = len(sentence)
        else:
            current_units += sentence_units
            current_raw_chars += separator_chars + len(sentence)
    if current_start < end:
        output.append((current_start, end))
    return output


def section_id(section: str) -> str:
    digest = hashlib.sha1(section.encode("utf-8")).hexdigest()[:8].upper()
    return f"SEC-{digest}"


def build_chunk_records(
    document: dict[str, Any],
    section: str,
    items: list[dict[str, Any]],
    boundaries: Iterable[int],
    config: ChunkConfig,
    similarities: list[float] | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []

    raw_similarities = similarities or []
    smoothed_similarities = smooth_similarities(raw_similarities, config.smoothing_window)
    parent_ranges = ranges_from_boundaries(len(items), boundaries)
    parent_ranges = merge_short_parent_ranges(parent_ranges, items, smoothed_similarities, config)

    chunks: list[dict[str, Any]] = []
    document_id = str(document.get("document_id") or "CNKI-UNKNOWN")
    chunk_index = 0
    for parent_index, (parent_start, parent_end) in enumerate(parent_ranges, start=1):
        parent_text = range_text(parent_start, parent_end, items)
        parent_id = f"{document_id}::{section_id(section)}::P{parent_index:03d}"
        # 子块仅为检索粒度。即使单句超长也不再按字符拆开，避免半句话证据。
        child_ranges = split_oversized_range(parent_start, parent_end, items, config)
        child_ids = [
            f"{document_id}::{section_id(section)}::{chunk_index + offset:03d}"
            for offset in range(1, len(child_ranges) + 1)
        ]
        for child_offset, (start, end) in enumerate(child_ranges):
            chunk_index += 1
            selected = items[start:end]
            text = range_text(start, end, items)
            pages = sorted({int(item["page_number"]) for item in selected})
            methods = sorted({str(item["extraction_method"]) for item in selected if item.get("extraction_method")})
            score_slice = raw_similarities[start : max(start, end - 1)]
            chunks.append(
                {
                "chunk_id": child_ids[child_offset],
                "document_id": document_id,
                "title": document.get("title", ""),
                "journal": document.get("journal", ""),
                "year": document.get("year", ""),
                "doi": document.get("doi", ""),
                "language": "zh",
                "source_type": "cnki_pdf",
                "source_path": document.get("source_path", ""),
                "section": section,
                "chunk_index": chunk_index,
                "parent_id": parent_id,
                "parent_index": parent_index,
                "parent_sentence_start": parent_start + 1,
                "parent_sentence_end": parent_end,
                "parent_sentence_count": parent_end - parent_start,
                "parent_text_unit_count": text_unit_count(parent_text),
                "parent_is_short": range_is_short(parent_start, parent_end, items, config),
                "parent_text": parent_text,
                "previous_chunk_id": child_ids[child_offset - 1] if child_offset > 0 else "",
                "next_chunk_id": child_ids[child_offset + 1] if child_offset + 1 < len(child_ids) else "",
                "source_pages": pages,
                "page_start": pages[0] if pages else "",
                "page_end": pages[-1] if pages else "",
                "extraction_methods": methods,
                "sentence_start": start + 1,
                "sentence_end": end,
                "sentence_count": end - start,
                "text_unit_count": text_unit_count(text),
                "char_count": len(text),
                "semantic_boundary_scores": [round(float(score), 6) for score in score_slice],
                "text": text,
                }
            )
    return chunks


# -----------------------------------------------------------------------------
# 文件与主流程
# -----------------------------------------------------------------------------


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
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_pages_by_document(
    records: Iterable[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        document_id = str(record.get("document_id") or "CNKI-UNKNOWN")
        groups.setdefault(document_id, []).append(record)
    return groups


def load_sentence_transformer(model_path: Path, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), device=device)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - 模型加载阶段会给出更明确的依赖错误。
        return "cpu"


def chunk_page_records(
    page_records: list[dict[str, Any]],
    model: Any,
    config: ChunkConfig,
    batch_size: int,
    limit_groups: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = group_pages_by_document(page_records)
    work_items: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
    for pages in documents.values():
        first = pages[0]
        sections = section_sentence_items(pages, include_references=config.include_references)
        for section, items in sections.items():
            work_items.append((first, section, items))
    if limit_groups is not None:
        work_items = work_items[:limit_groups]

    def encode(texts: list[str]):
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    chunks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, (document, section, items) in enumerate(work_items, start=1):
        sentences = [str(item["text"]) for item in items]
        row = {
            "document_id": document.get("document_id", ""),
            "title": document.get("title", ""),
            "section": section,
            "sentences": len(sentences),
            "similarity_gaps": 0,
            "semantic_boundaries": 0,
            "parent_chunks": 0,
            "short_parents": 0,
            "chunks": 0,
            "status": "parsed",
            "error": "",
        }
        try:
            similarities = compute_gap_similarities(sentences, encode, config.window_size)
            boundaries = choose_semantic_boundaries(sentences, similarities, config)
            group_chunks = build_chunk_records(
                document,
                section,
                items,
                boundaries,
                config,
                similarities,
            )
            chunks.extend(group_chunks)
            row["similarity_gaps"] = len(similarities)
            row["semantic_boundaries"] = len(boundaries)
            row["parent_chunks"] = len({str(chunk.get("parent_id") or "") for chunk in group_chunks})
            row["short_parents"] = len(
                {
                    str(chunk.get("parent_id") or "")
                    for chunk in group_chunks
                    if chunk.get("parent_is_short")
                }
            )
            row["chunks"] = len(group_chunks)
            print(
                f"[{index}/{len(work_items)}] {document.get('title')} | {section}: "
                f"sentences={len(sentences)} boundaries={len(boundaries)} "
                f"chunks={len(group_chunks)}"
            )
        except Exception as exc:  # noqa: BLE001 - 失败章节写入 manifest，其他文献继续。
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(work_items)}] failed: {row['error']}", file=sys.stderr)
        manifest_rows.append(row)
    return chunks, manifest_rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "document_id",
        "title",
        "section",
        "sentences",
        "similarity_gaps",
        "semantic_boundaries",
        "parent_chunks",
        "short_parents",
        "chunks",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantically chunk parsed Chinese CNKI PDF pages with BGE-M3."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/articles/processed/chinese/article_pages.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/articles/processed/chinese/article_chunks.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/articles/processed/chinese/chunk_manifest.csv"),
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/bge/bge-m3"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--similarity-percentile", type=float, default=20.0)
    parser.add_argument("--valley-margin", type=float, default=0.02)
    parser.add_argument("--min-sentences", type=int, default=2)
    parser.add_argument("--max-sentences", type=int, default=12)
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--max-raw-chars", type=int, default=4000)
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--limit-groups", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.model_path.exists():
        print(f"model path not found: {args.model_path}", file=sys.stderr)
        return 2

    config = ChunkConfig(
        window_size=args.window_size,
        smoothing_window=args.smoothing_window,
        similarity_percentile=args.similarity_percentile,
        valley_margin=args.valley_margin,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_raw_chars=args.max_raw_chars,
        include_references=args.include_references,
    )
    pages = read_jsonl(args.input)
    device = resolve_device(args.device)
    model = load_sentence_transformer(args.model_path, device)
    chunks, manifest_rows = chunk_page_records(
        pages,
        model,
        config,
        args.batch_size,
        args.limit_groups,
    )
    write_jsonl(args.out, chunks)
    write_manifest(args.manifest, manifest_rows)

    failed = sum(row["status"] != "parsed" for row in manifest_rows)
    print(f"page_records={len(pages)}")
    print(f"chunks={len(chunks)}")
    print(f"failed_groups={failed}")
    print(f"out={args.out}")
    print(f"manifest={args.manifest}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
