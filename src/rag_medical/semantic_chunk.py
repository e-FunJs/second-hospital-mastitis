from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


# -----------------------------------------------------------------------------
# 配置对象
# -----------------------------------------------------------------------------
# 这里把 chunk 参数集中放在一个 dataclass 
# 注意: 真正的切分依据仍然是 section 内的语义相似度低谷


@dataclass(frozen=True)
class ChunkConfig:
    window_size: int = 2
    similarity_percentile: float = 20.0
    valley_margin: float = 0.02
    min_sentences: int = 2
    max_sentences: int = 12
    min_words: int = 80
    max_words: int = 360


COMMON_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "Fig.",
    "fig.",
    "Dr.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "Prof.",
    "vs.",
    "al.",
    "No.",
}


# -----------------------------------------------------------------------------
# 文本基础处理
# -----------------------------------------------------------------------------
# 这部分只负责把段落变成相对稳定的句子序列。医学英文里有 e.g.、Fig.、et al. 等缩写
# 如果直接按句号切,会把一句话错误拆开,所以先保护常见缩写


def normalize_space(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_space(text)
    if not normalized:
        return []

    protected = normalized
    placeholders: dict[str, str] = {}
    for index, abbreviation in enumerate(sorted(COMMON_ABBREVIATIONS, key=len, reverse=True)):
        placeholder = f"<ABBR{index}>"
        if abbreviation in protected:
            protected = protected.replace(abbreviation, placeholder)
            placeholders[placeholder] = abbreviation

    pieces = re.split(r"(?<=[.!?。！？])\s+", protected)
    sentences: list[str] = []
    for piece in pieces:
        sentence = piece.strip()
        if not sentence:
            continue
        for placeholder, abbreviation in placeholders.items():
            sentence = sentence.replace(placeholder, abbreviation)
        sentences.append(sentence)
    return sentences


def word_count(text: str) -> int:
    # 英文文献以空格分词为主; 如果后续混入中文,此正则也可把连续中文字符计入长度
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]", text))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


# -----------------------------------------------------------------------------
# 语义边界选择   ※※※※※※※※
# -----------------------------------------------------------------------------
# similarities[i] 表示 sentence[i] 与 sentence[i + 1] 附近窗口的语义相似度。
# 当某个位置相似度明显低于本 section 的局部水平,并且不是太靠近 chunk 开头/结尾,
# 就认为这里可能发生了话题转换,可以作为 chunk 边界    


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
        raise ValueError("similarities length must be exactly len(sentences) - 1")

    threshold = percentile(similarities, config.similarity_percentile)
    boundaries: list[int] = []
    chunk_start = 0

    for similarity_index, score in enumerate(similarities):
        boundary_after = similarity_index + 1
        current_sentence_count = boundary_after - chunk_start
        remaining_sentence_count = len(sentences) - boundary_after

        if current_sentence_count >= config.max_sentences:
            boundaries.append(boundary_after)
            chunk_start = boundary_after
            continue

        if current_sentence_count < config.min_sentences:
            continue
        if remaining_sentence_count < config.min_sentences:
            continue

        # 关键点：低于分位数阈值还不够,还要是局部低谷。
        # 这样可以避免 section 整体相似度偏低时被切得过碎。
        if score <= threshold and is_local_valley(similarities, similarity_index, config.valley_margin):
            boundaries.append(boundary_after)
            chunk_start = boundary_after

    return boundaries


# -----------------------------------------------------------------------------
# 段落记录展开与 chunk 记录构造
# -----------------------------------------------------------------------------
# 输入 JSONL 是段落级记录；chunk 时需要先展开到句子级,但输出必须保留原段落索引,
# 这样将来 RAG 命中 chunk 后仍然能追溯回原始 PMC XML 的 section 和 paragraph。


def safe_chunk_id_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned[:80] or "section"


def sentence_items_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: int(row.get("paragraph_index") or 0)):
        paragraph_index = int(record.get("paragraph_index") or 0)
        for sentence in split_sentences(str(record.get("text") or "")):
            items.append({"text": sentence, "paragraph_index": paragraph_index})
    return items


def ranges_from_boundaries(sentence_count: int, boundaries: Iterable[int]) -> list[tuple[int, int]]:
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
    sentence_items: list[dict[str, Any]],
    config: ChunkConfig,
) -> list[tuple[int, int]]:
    if len(ranges) <= 1 or config.min_words <= 0:
        return ranges

    merged: list[tuple[int, int]] = []
    pending_start, pending_end = ranges[0]

    for start, end in ranges[1:]:
        pending_text = " ".join(item["text"] for item in sentence_items[pending_start:pending_end])
        next_text = " ".join(item["text"] for item in sentence_items[start:end])
        combined_words = word_count(pending_text + " " + next_text)

        # 关键点：语义边界优先,但太短的 chunk 检索价值低,因此只在合并后不过长时合并。
        if word_count(pending_text) < config.min_words and combined_words <= config.max_words:
            pending_end = end
            continue

        merged.append((pending_start, pending_end))
        pending_start, pending_end = start, end

    merged.append((pending_start, pending_end))
    return merged


def split_oversized_range(
    start: int,
    end: int,
    sentence_items: list[dict[str, Any]],
    config: ChunkConfig,
) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    current_start = start
    current_words = 0

    for index in range(start, end):
        sentence_words = word_count(sentence_items[index]["text"])
        sentence_count = index - current_start + 1
        should_cut = (
            sentence_count > config.min_sentences
            and current_words + sentence_words > config.max_words
        )
        if should_cut:
            chunks.append((current_start, index))
            current_start = index
            current_words = sentence_words
        else:
            current_words += sentence_words

    if current_start < end:
        chunks.append((current_start, end))
    return chunks


def build_chunk_records(
    records: list[dict[str, Any]],
    boundary_after_sentence_indices: Iterable[int],
    config: ChunkConfig | None = None,
    semantic_scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        return []

    sentence_items = sentence_items_from_records(records)
    if not sentence_items:
        return []

    effective_config = config or ChunkConfig(min_words=0, max_words=10_000)
    ranges = ranges_from_boundaries(len(sentence_items), boundary_after_sentence_indices)
    ranges = merge_short_ranges(ranges, sentence_items, effective_config)

    final_ranges: list[tuple[int, int]] = []
    for start, end in ranges:
        text = " ".join(item["text"] for item in sentence_items[start:end])
        if word_count(text) > effective_config.max_words:
            final_ranges.extend(split_oversized_range(start, end, sentence_items, effective_config))
        else:
            final_ranges.append((start, end))

    first = records[0]
    pmcid = str(first.get("pmcid") or "UNKNOWN")
    section = str(first.get("section") or "Body")
    section_id = safe_chunk_id_part(section)
    chunks: list[dict[str, Any]] = []

    for chunk_index, (start, end) in enumerate(final_ranges, start=1):
        selected = sentence_items[start:end]
        text = normalize_space(" ".join(item["text"] for item in selected))
        paragraph_indices = sorted({int(item["paragraph_index"]) for item in selected})
        score_slice = semantic_scores[start : max(start, end - 1)] if semantic_scores else []

        chunks.append(
            {
                "chunk_id": f"{pmcid}::{section_id}::{chunk_index:03d}",
                "pmcid": pmcid,
                "pmid": first.get("pmid", ""),
                "doi": first.get("doi", ""),
                "title": first.get("title", ""),
                "journal": first.get("journal", ""),
                "year": first.get("year", ""),
                "source_path": first.get("source_path", ""),
                "section": section,
                "chunk_index": chunk_index,
                "source_paragraph_indices": paragraph_indices,
                "sentence_start": start + 1,
                "sentence_end": end,
                "sentence_count": end - start,
                "word_count": word_count(text),
                "char_count": len(text),
                "semantic_boundary_scores": [round(score, 6) for score in score_slice],
                "text": text,
            }
        )

    return chunks


# -----------------------------------------------------------------------------
# BGE embedding 与相似度计算
# -----------------------------------------------------------------------------
# BGE-M3 只在这里用于计算相邻句子窗口的 embedding 相似度。它不生成新文本,
# 也不改写文献内容；chunk 输出的 text 仍然全部来自原始解析文本。


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def window_texts_for_gaps(sentences: list[str], window_size: int) -> list[str]:
    windows: list[str] = []
    for gap_index in range(len(sentences) - 1):
        left_start = max(0, gap_index - window_size + 1)
        left = " ".join(sentences[left_start : gap_index + 1])
        right_end = min(len(sentences), gap_index + 1 + window_size)
        right = " ".join(sentences[gap_index + 1 : right_end])
        windows.extend([left, right])
    return windows


def compute_gap_similarities(
    sentences: list[str],
    encode: Callable[[list[str]], Any],
    window_size: int,
) -> list[float]:
    if len(sentences) <= 1:
        return []

    window_texts = window_texts_for_gaps(sentences, window_size)
    embeddings = encode(window_texts)
    similarities: list[float] = []
    for index in range(0, len(embeddings), 2):
        similarities.append(float(cosine_similarity(embeddings[index], embeddings[index + 1])))
    return similarities


# -----------------------------------------------------------------------------
# 文件读写与分组
# -----------------------------------------------------------------------------
# 分组粒度是 pmcid + section。section 是硬边界；不同 section 不互相合并,
# 因为 Methods、Results、Discussion 的信息角色不同,强行合并会降低医学证据可解释性。


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_by_article_section(records: Iterable[dict[str, Any]]) -> OrderedDict[tuple[str, str], list[dict[str, Any]]]:
    groups: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for record in records:
        key = (str(record.get("pmcid") or "UNKNOWN"), str(record.get("section") or "Body"))
        groups.setdefault(key, []).append(record)
    return groups


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pmcid",
        "section",
        "paragraph_records",
        "sentences",
        "similarity_gaps",
        "semantic_boundaries",
        "chunks",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
# CLI 流程：读取 paragraph JSONL -> 分组 -> BGE 计算相似度 -> 选择语义边界 -> 写 chunk JSONL。
# 这里故意不把向量库写入本脚本,chunk 与 embedding index 是两个阶段,便于调参和复查。


def load_sentence_transformer(model_path: Path, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), device=device)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch 不可用时退回 CPU,并让模型加载阶段给出明确错误。
        return "cpu"


def chunk_records(
    records: list[dict[str, Any]],
    model: Any,
    config: ChunkConfig,
    batch_size: int,
    limit_groups: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = group_by_article_section(records)
    all_chunks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    def encode(texts: list[str]):
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

    selected_groups = list(groups.items())
    if limit_groups is not None:
        selected_groups = selected_groups[:limit_groups]

    for group_index, ((pmcid, section), group_records) in enumerate(selected_groups, start=1):
        sentence_items = sentence_items_from_records(group_records)
        sentences = [item["text"] for item in sentence_items]
        row = {
            "pmcid": pmcid,
            "section": section,
            "paragraph_records": len(group_records),
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
            chunks = build_chunk_records(group_records, boundaries, config, similarities)
            all_chunks.extend(chunks)
            row["similarity_gaps"] = len(similarities)
            row["semantic_boundaries"] = len(boundaries)
            row["chunks"] = len(chunks)
            print(
                f"[{group_index}/{len(selected_groups)}] {pmcid} | {section}: "
                f"sentences={len(sentences)} boundaries={len(boundaries)} chunks={len(chunks)}"
            )
        except Exception as exc:  # noqa: BLE001 - manifest 需要记录失败分组,方便后续定位。
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{group_index}/{len(selected_groups)}] {pmcid} | {section}: failed: {row['error']}")

        manifest_rows.append(row)

    return all_chunks, manifest_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic chunk PMC article sections with BGE embeddings.")
    parser.add_argument("--input", type=Path, default=Path("data/articles/processed/article_sections.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/articles/processed/article_chunks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/articles/processed/chunk_manifest.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/bge/bge-m3"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--similarity-percentile", type=float, default=20.0)
    parser.add_argument("--valley-margin", type=float, default=0.02)
    parser.add_argument("--min-sentences", type=int, default=2)
    parser.add_argument("--max-sentences", type=int, default=12)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-words", type=int, default=360)
    parser.add_argument("--limit-groups", type=int, help="Only process the first N article-section groups for smoke tests.")
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
        min_words=args.min_words,
        max_words=args.max_words,
    )
    device = resolve_device(args.device)
    records = read_jsonl(args.input)
    model = load_sentence_transformer(args.model_path, device)
    chunks, manifest_rows = chunk_records(records, model, config, args.batch_size, args.limit_groups)

    write_jsonl(args.out, chunks)
    write_manifest(args.manifest, manifest_rows)

    failed = sum(1 for row in manifest_rows if row.get("status") != "parsed")
    print(f"records={len(records)}")
    print(f"chunks={len(chunks)}")
    print(f"failed_groups={failed}")
    print(f"out={args.out}")
    print(f"manifest={args.manifest}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
