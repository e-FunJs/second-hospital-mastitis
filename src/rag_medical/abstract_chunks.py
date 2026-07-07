from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from rag_medical.semantic_chunk import normalize_space, split_sentences, word_count


# -----------------------------------------------------------------------------
# 基础清洗与 CSV 读取
# -----------------------------------------------------------------------------
# 本模块把 PubMed registry 中“没有 PMC 全文但有摘要”的记录转成 RAG chunk。
# 它不改写、不总结摘要内容，只做清洗、切句和 metadata 标准化。


def has_text(value: str | None) -> bool:
    return bool(normalize_space(value))


def normalize_pmcid(value: str | None) -> str:
    pmcid = normalize_space(value).upper()
    if not pmcid:
        return ""
    if pmcid.startswith("PMC"):
        return pmcid
    if pmcid.isdigit():
        return f"PMC{pmcid}"
    return pmcid


def safe_id_part(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned[:80] or fallback


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: normalize_space(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# 摘要切分
# -----------------------------------------------------------------------------
# PubMed 摘要通常已经比较短，因此这里不再调用 BGE 做语义边界计算。
# 做法是以句子为最小单位，把较长摘要按 max_words 轻量拆开；短摘要保持完整。
# 这样能扩大知识库，同时避免把一句医学结论截断。


def sentence_ranges_by_words(sentences: list[str], max_words: int) -> list[tuple[int, int]]:
    if not sentences:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    current_words = 0
    for index, sentence in enumerate(sentences):
        sentence_words = word_count(sentence)
        should_cut = index > start and current_words + sentence_words > max_words
        if should_cut:
            ranges.append((start, index))
            start = index
            current_words = sentence_words
            continue
        current_words += sentence_words

    ranges.append((start, len(sentences)))
    return ranges


def abstract_piece_text(sentences: list[str], start: int, end: int) -> str:
    return normalize_space(" ".join(sentences[start:end]))


# -----------------------------------------------------------------------------
# Chunk 记录构造
# -----------------------------------------------------------------------------
# 输出字段尽量对齐 PMC XML chunk，方便后面的 combine、embedding、FAISS 直接复用。
# source_type 是关键字段：后续回答和评估可以据此区分“全文证据”和“摘要证据”。


def row_identifier(row: dict[str, str]) -> tuple[str, str]:
    pmid = normalize_space(row.get("pmid"))
    doi = normalize_space(row.get("doi"))
    title = normalize_space(row.get("title"))
    if pmid:
        return "PMID", pmid
    if doi:
        return "DOI", safe_id_part(doi, "doi")
    return "TITLE", safe_id_part(title, "untitled")


def row_to_chunks(row: dict[str, str], max_words: int) -> list[dict[str, Any]]:
    abstract = normalize_space(row.get("abstract"))
    if not abstract:
        return []

    id_prefix, id_value = row_identifier(row)
    sentences = split_sentences(abstract) or [abstract]
    ranges = sentence_ranges_by_words(sentences, max_words=max_words)
    source_url = normalize_space(row.get("source_url"))
    if not source_url and id_prefix == "PMID":
        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{id_value}/"

    chunks: list[dict[str, Any]] = []
    for chunk_index, (start, end) in enumerate(ranges, start=1):
        text = abstract_piece_text(sentences, start, end)
        if not text:
            continue

        chunks.append(
            {
                "chunk_id": f"{id_prefix}{id_value}::Abstract::{chunk_index:03d}",
                "source_type": "pubmed_abstract",
                "pmcid": "",
                "pmid": normalize_space(row.get("pmid")),
                "doi": normalize_space(row.get("doi")),
                "title": normalize_space(row.get("title")),
                "journal": normalize_space(row.get("journal")),
                "year": normalize_space(row.get("year")),
                "source_path": source_url,
                "source_url": source_url,
                "section": "Abstract",
                "chunk_index": chunk_index,
                "source_paragraph_indices": [1],
                "sentence_start": start + 1,
                "sentence_end": end,
                "sentence_count": end - start,
                "word_count": word_count(text),
                "char_count": len(text),
                "semantic_boundary_scores": [],
                "text": text,
            }
        )
    return chunks


def registry_rows_to_chunks(
    rows: Iterable[dict[str, str]],
    max_words: int = 260,
    include_pmcid_records: bool = False,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for row in rows:
        if normalize_pmcid(row.get("pmcid")) and not include_pmcid_records:
            continue
        if not has_text(row.get("abstract")):
            continue
        chunks.extend(row_to_chunks(row, max_words=max_words))
    return chunks


def build_abstract_chunks(
    registry_path: Path,
    out_path: Path,
    max_words: int = 260,
    include_pmcid_records: bool = False,
) -> int:
    rows = read_registry(registry_path)
    chunks = registry_rows_to_chunks(rows, max_words=max_words, include_pmcid_records=include_pmcid_records)
    write_jsonl(out_path, chunks)
    return len(chunks)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG chunks from PubMed abstracts without PMC full text.")
    parser.add_argument("--registry", type=Path, default=Path("data/registry/processed/literature_registry.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/articles/processed/abstract_chunks.jsonl"))
    parser.add_argument("--max-words", type=int, default=260)
    parser.add_argument("--include-pmcid-records", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.registry.exists():
        print(f"registry not found: {args.registry}", file=sys.stderr)
        return 2

    count = build_abstract_chunks(
        registry_path=args.registry,
        out_path=args.out,
        max_words=args.max_words,
        include_pmcid_records=args.include_pmcid_records,
    )
    print(f"abstract_chunks={count}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
