from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


# -----------------------------------------------------------------------------
# JSONL 读写
# -----------------------------------------------------------------------------
# 本模块只做一件事：把不同来源的 chunk JSONL 合成一个 RAG 输入文件。
# 它不重新切分、不生成 embedding，避免多个阶段职责混在一起。


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# 合并规则
# -----------------------------------------------------------------------------
# 默认把没有 source_type 的旧全文 chunk 标为 pmc_full_text。
# 通过 chunk_id 去重，避免重复运行或重复来源造成 FAISS index 中证据膨胀。


def combine_chunk_files(input_paths: list[Path], out_path: Path) -> int:
    combined: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for path in input_paths:
        for record in read_jsonl(path):
            chunk_id = str(record.get("chunk_id") or "")
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            if chunk_id:
                seen_chunk_ids.add(chunk_id)

            normalized = dict(record)
            if not normalized.get("source_type"):
                normalized["source_type"] = "pmc_full_text"
            combined.append(normalized)

    write_jsonl(out_path, combined)
    return len(combined)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine full-text and abstract chunks for one RAG index.")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Input chunk JSONL path. Pass multiple --input values in priority order.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/articles/processed/rag_chunks.jsonl"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    count = combine_chunk_files(args.input, args.out)
    print(f"combined_chunks={count}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
