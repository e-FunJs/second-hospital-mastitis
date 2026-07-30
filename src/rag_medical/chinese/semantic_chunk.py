"""
用途：把中文 PDF 页级文本按章节和语义边界切分成可检索的 RAG chunk。

默认输入：
    data/articles/processed/chinese/article_pages.jsonl

默认输出：
    1. data/articles/processed/chinese/article_chunks.jsonl
       中文语义 chunk；保留文献、章节、页码和提取方式。
    2. data/articles/processed/chinese/chunk_manifest.csv
       每篇文献、每个章节的句数、语义边界数、chunk 数和错误。

说明：
    - 中文章节标题是硬边界，不把摘要、方法、结果、讨论强行拼在一起。
    - BGE-M3 只用于比较相邻句群的语义相似度，不生成或改写文章内容。
    - 长度阈值只防止 chunk 过短或过长，主要切分依据仍是语义相似度局部低谷。
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
    window_size: int = 2
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


def split_sentences(text: str) -> list[str]:
    """按中文结束标点切句，并把 OCR 产生的极短碎片并回相邻句。"""

    normalized = normalize_space(text)
    if not normalized:
        return []

    pieces = re.findall(r".+?(?:[。！？!?；;]+[”’\"）)\]]*|$)", normalized)
    sentences: list[str] = []
    pending = ""
    for piece in pieces:
        sentence = normalize_space(piece)
        if not sentence:
            continue
        if len(re.sub(r"\W", "", sentence)) < 6 and not re.search(r"[。！？!?；;]$", sentence):
            pending += sentence
            continue
        if pending:
            sentence = pending + sentence
            pending = ""
        sentences.append(sentence)
    if pending:
        if sentences:
            sentences[-1] += pending
        else:
            sentences.append(pending)

    output: list[str] = []
    for sentence in sentences:
        output.extend(split_long_sentence(sentence))
    return output


def split_long_sentence(
    sentence: str,
    max_chars: int = 1800,
    max_units: int = 850,
) -> list[str]:
    """为无句号的目录、英文摘要和 OCR 长行提供保底切分，不改写原文。"""

    remaining = normalize_space(sentence)
    output: list[str] = []
    while len(remaining) > max_chars or text_unit_count(remaining) > max_units:
        remaining_units = text_unit_count(remaining)
        target = min(
            max_chars,
            max(1, int(len(remaining) * max_units / max(remaining_units, 1))),
        )
        while target > 1 and text_unit_count(remaining[:target]) > max_units:
            target = max(1, int(target * 0.90))
        lower_bound = max(1, int(target * 0.60))
        boundary = max(
            remaining.rfind(separator, lower_bound, target)
            for separator in ("，", ",", "；", ";", "：", ":", "、", " ")
        )
        cut = boundary + 1 if boundary >= lower_bound else target
        if cut >= len(remaining):
            cut = max(1, len(remaining) // 2)
        output.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        output.append(remaining)
    return output


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

    for page_record in sorted(page_records, key=lambda row: int(row.get("page_number") or 0)):
        page_number = int(page_record.get("page_number") or 0)
        extraction_method = str(page_record.get("extraction_method") or "")
        document_title = re.sub(r"\s+", "", normalize_space(str(page_record.get("title") or "")))
        buffer: list[str] = []

        def flush_buffer() -> None:
            if not buffer or skip_references:
                buffer.clear()
                return
            text = join_text_lines(buffer)
            for sentence in split_sentences(text):
                sections.setdefault(current_section, []).append(
                    {
                        "text": sentence,
                        "page_number": page_number,
                        "extraction_method": extraction_method,
                    }
                )
            buffer.clear()

        for raw_line in str(page_record.get("text") or "").splitlines():
            line = normalize_space(raw_line)
            if not line:
                continue
            # 论文标题经常形如“100 例非哺乳期乳腺炎的治疗”，既包含数字也包含
            # “治疗”等章节词。先与文件名标题比较，避免把论文标题误判成章节。
            is_document_title = (
                bool(document_title)
                and re.sub(r"\s+", "", line).strip("【】[]") == document_title
            )
            heading = None if is_document_title else canonical_section_heading(line)
            if heading:
                flush_buffer()
                current_section = heading
                skip_references = heading == "参考文献" and not include_references
                continue
            if not skip_references:
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


def choose_semantic_boundaries(
    sentences: list[str],
    similarities: list[float],
    config: ChunkConfig,
) -> list[int]:
    if len(sentences) <= 1:
        return []
    if len(similarities) != len(sentences) - 1:
        raise ValueError("similarities length must equal len(sentences) - 1")

    threshold = percentile(similarities, config.similarity_percentile)
    boundaries: list[int] = []
    chunk_start = 0
    for score_index, score in enumerate(similarities):
        boundary_after = score_index + 1
        sentence_count = boundary_after - chunk_start
        remaining = len(sentences) - boundary_after

        if sentence_count >= config.max_sentences:
            boundaries.append(boundary_after)
            chunk_start = boundary_after
            continue
        if sentence_count < config.min_sentences or remaining < config.min_sentences:
            continue
        if score <= threshold and is_local_valley(similarities, score_index, config.valley_margin):
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


def merge_short_ranges(
    ranges: list[tuple[int, int]],
    items: list[dict[str, Any]],
    config: ChunkConfig,
) -> list[tuple[int, int]]:
    if len(ranges) <= 1:
        return ranges
    merged: list[tuple[int, int]] = []
    pending_start, pending_end = ranges[0]
    for start, end in ranges[1:]:
        pending_text = join_text_lines(
            [str(item["text"]) for item in items[pending_start:pending_end]]
        )
        next_text = join_text_lines([str(item["text"]) for item in items[start:end]])
        combined_text = join_text_lines([pending_text, next_text])
        if (
            text_unit_count(pending_text) < config.min_chars
            and text_unit_count(combined_text) <= config.max_chars
            and len(combined_text) <= config.max_raw_chars
        ):
            pending_end = end
            continue
        merged.append((pending_start, pending_end))
        pending_start, pending_end = start, end
    merged.append((pending_start, pending_end))

    # 最后一段没有“下一段”可供合并，旧逻辑会把短结论或短页脚单独留下。
    # 若与前一段合并后仍满足两个长度上限，则向前合并。
    if len(merged) >= 2:
        previous_start, previous_end = merged[-2]
        last_start, last_end = merged[-1]
        last_text = join_text_lines([str(item["text"]) for item in items[last_start:last_end]])
        combined_text = join_text_lines(
            [str(item["text"]) for item in items[previous_start:last_end]]
        )
        if (
            text_unit_count(last_text) < config.min_chars
            and text_unit_count(combined_text) <= config.max_chars
            and len(combined_text) <= config.max_raw_chars
        ):
            merged[-2:] = [(previous_start, last_end)]
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

    ranges = ranges_from_boundaries(len(items), boundaries)
    ranges = merge_short_ranges(ranges, items, config)
    final_ranges: list[tuple[int, int]] = []
    for start, end in ranges:
        text = join_text_lines([str(item["text"]) for item in items[start:end]])
        if text_unit_count(text) > config.max_chars or len(text) > config.max_raw_chars:
            final_ranges.extend(split_oversized_range(start, end, items, config))
        else:
            final_ranges.append((start, end))

    chunks: list[dict[str, Any]] = []
    document_id = str(document.get("document_id") or "CNKI-UNKNOWN")
    for chunk_index, (start, end) in enumerate(final_ranges, start=1):
        selected = items[start:end]
        text = join_text_lines([str(item["text"]) for item in selected])
        pages = sorted({int(item["page_number"]) for item in selected})
        methods = sorted({str(item["extraction_method"]) for item in selected if item.get("extraction_method")})
        score_slice = similarities[start : max(start, end - 1)] if similarities else []
        chunks.append(
            {
                "chunk_id": f"{document_id}::{section_id(section)}::{chunk_index:03d}",
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
