"""
用途：在生成 embedding 前检查 chunk JSONL 是否满足通用 RAG 输入契约。

输入：
    任意语言的 rag_chunks*.jsonl。

输出：
    默认只在终端打印；提供 --report 时写入 JSON 检查报告。

检查内容：
    - JSONL 能否读取、chunk_id 是否唯一；
    - title、text、source_path、source_type、language 等追溯字段是否完整；
    - 文本是否为空、过短或异常超长；
    - 是否能被 common.build_embeddings 构造成 embedding 文本和 metadata；
    - 中文 chunk 是否保留 document_id、页码和 OCR/原生提取来源。

说明：
    本模块不会加载 BGE、不会计算向量、不会创建 FAISS 索引。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_medical.common.build_embeddings import (
    build_embedding_text,
    build_metadata_record,
    read_jsonl,
)


BASE_REQUIRED_FIELDS = (
    "chunk_id",
    "title",
    "text",
    "source_path",
    "source_type",
    "language",
)

CHINESE_REQUIRED_FIELDS = (
    "document_id",
    "source_pages",
    "page_start",
    "page_end",
    "extraction_methods",
)


def validate_corpus(
    chunks: list[dict[str, Any]],
    expected_language: str | None = None,
    min_text_chars: int = 40,
    max_text_chars: int = 6000,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    ids: set[str] = set()
    language_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()

    for row_index, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        language = str(chunk.get("language") or "").strip()
        text = str(chunk.get("text") or "").strip()
        language_counts[language or "<missing>"] += 1
        source_type_counts[str(chunk.get("source_type") or "<missing>")] += 1

        missing = [field for field in BASE_REQUIRED_FIELDS if not chunk.get(field)]
        if language == "zh":
            missing.extend(field for field in CHINESE_REQUIRED_FIELDS if not chunk.get(field))
        if missing:
            errors.append(
                {
                    "row_index": row_index,
                    "chunk_id": chunk_id,
                    "issue": "missing_fields",
                    "detail": sorted(set(missing)),
                }
            )

        if not chunk_id:
            continue
        if chunk_id in ids:
            errors.append(
                {
                    "row_index": row_index,
                    "chunk_id": chunk_id,
                    "issue": "duplicate_chunk_id",
                    "detail": chunk_id,
                }
            )
        ids.add(chunk_id)

        if expected_language and language != expected_language:
            errors.append(
                {
                    "row_index": row_index,
                    "chunk_id": chunk_id,
                    "issue": "unexpected_language",
                    "detail": language,
                }
            )
        if len(text) < min_text_chars:
            warnings.append(
                {
                    "row_index": row_index,
                    "chunk_id": chunk_id,
                    "issue": "short_text",
                    "detail": len(text),
                }
            )
        if len(text) > max_text_chars:
            warnings.append(
                {
                    "row_index": row_index,
                    "chunk_id": chunk_id,
                    "issue": "long_text",
                    "detail": len(text),
                }
            )

        # 这里只调用纯 Python 的文本/metadata 构造函数，不加载模型、不编码向量。
        embedding_text = build_embedding_text(chunk)
        metadata = build_metadata_record(chunk, row_index=row_index)
        if not embedding_text.strip() or "Text:" not in embedding_text:
            errors.append(
                {
                    "row_index": row_index,
                    "chunk_id": chunk_id,
                    "issue": "embedding_text_failed",
                    "detail": "",
                }
            )
        if metadata.get("row_index") != row_index or metadata.get("chunk_id") != chunk_id:
            errors.append(
                {
                    "row_index": row_index,
                    "chunk_id": chunk_id,
                    "issue": "metadata_alignment_failed",
                    "detail": metadata.get("chunk_id"),
                }
            )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "unique_chunk_ids": len(ids),
        "expected_language": expected_language or "",
        "language_counts": dict(language_counts),
        "source_type_counts": dict(source_type_counts),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "ready_for_common_embedding": len(chunks) > 0 and not errors,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a RAG chunk corpus without generating embeddings or an index."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-language", choices=["zh", "en"])
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--max-text-chars", type=int, default=6000)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    chunks = read_jsonl(args.input)
    report = validate_corpus(
        chunks,
        expected_language=args.expected_language,
        min_text_chars=args.min_text_chars,
        max_text_chars=args.max_text_chars,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"chunks={report['chunk_count']}")
    print(f"unique_chunk_ids={report['unique_chunk_ids']}")
    print(f"errors={report['error_count']}")
    print(f"warnings={report['warning_count']}")
    print(f"ready_for_common_embedding={report['ready_for_common_embedding']}")
    if args.report:
        print(f"report={args.report}")
    return 0 if report["ready_for_common_embedding"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
