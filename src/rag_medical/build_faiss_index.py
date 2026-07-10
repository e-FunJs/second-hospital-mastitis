"""
用途：把已生成的 chunk embedding 矩阵构建成 FAISS 向量索引。
输入：data/index/chunk_embeddings.npy 与 data/index/chunk_metadata.jsonl。
输出：data/index/faiss.index 与 data/index/faiss_manifest.json。
说明：本文件只建索引，不负责生成 embedding，也不负责回答问题。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np


# -----------------------------------------------------------------------------
# 输入读取与一致性检查
# -----------------------------------------------------------------------------
# FAISS 索引只保存向量，不保存文本。因此必须严格保证 embeddings 第 N 行与
# metadata 第 N 行对应同一个 chunk；否则后续检索会命中错文献。


def read_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_index = record.get("row_index")
            if row_index != len(records):
                raise ValueError(
                    f"metadata row_index mismatch on line {line_number}: "
                    f"expected {len(records)}, got {row_index}"
                )
            records.append(record)
    return records


def load_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be a 2D matrix, got shape={embeddings.shape}")
    return np.asarray(embeddings, dtype=np.float32)


def validate_embeddings_and_metadata(
    embeddings: np.ndarray,
    metadata_records: list[dict[str, Any]],
) -> None:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    if embeddings.shape[0] != len(metadata_records):
        raise ValueError(
            f"metadata rows must match embedding rows: "
            f"embeddings={embeddings.shape[0]}, metadata rows={len(metadata_records)}"
        )
    if embeddings.shape[0] == 0:
        raise ValueError("cannot build FAISS index from zero embeddings")


# -----------------------------------------------------------------------------
# FAISS 索引构建
# -----------------------------------------------------------------------------
# BGE embedding 在上一阶段已经 normalize_embeddings=True；这里再次 normalize 是防御性处理。
# 对归一化向量而言，Inner Product 与 Cosine Similarity 等价。


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    normalized = np.asarray(embeddings, dtype=np.float32).copy()
    faiss.normalize_L2(normalized)
    return normalized


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    normalized = normalize_rows(embeddings)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index


def write_faiss_manifest(
    manifest_path: Path,
    embedding_path: str,
    metadata_path: str,
    index_path: str,
    chunk_count: int,
    embedding_dim: int,
    metric: str,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_path": embedding_path,
        "metadata_path": metadata_path,
        "index_path": index_path,
        "chunk_count": chunk_count,
        "embedding_dim": embedding_dim,
        "metric": metric,
        "faiss_index_type": "IndexFlatIP",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
# 本脚本只负责把 embedding 矩阵变成 FAISS index；query 检索放在 search_chunks.py。


def build_and_save_index(
    embedding_path: Path,
    metadata_path: Path,
    index_path: Path,
    manifest_path: Path,
) -> tuple[int, int]:
    embeddings = load_embeddings(embedding_path)
    metadata_records = read_metadata_jsonl(metadata_path)
    validate_embeddings_and_metadata(embeddings, metadata_records)

    index = build_faiss_index(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    write_faiss_manifest(
        manifest_path=manifest_path,
        embedding_path=str(embedding_path),
        metadata_path=str(metadata_path),
        index_path=str(index_path),
        chunk_count=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        metric="inner_product_cosine",
    )
    return int(embeddings.shape[0]), int(embeddings.shape[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from chunk embeddings.")
    parser.add_argument("--embeddings", type=Path, default=Path("data/index/chunk_embeddings.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("data/index/chunk_metadata.jsonl"))
    parser.add_argument("--index-out", type=Path, default=Path("data/index/faiss.index"))
    parser.add_argument("--manifest", type=Path, default=Path("data/index/faiss_manifest.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in [args.embeddings, args.metadata]:
        if not path.exists():
            print(f"input not found: {path}", file=sys.stderr)
            return 2

    count, dim = build_and_save_index(
        embedding_path=args.embeddings,
        metadata_path=args.metadata,
        index_path=args.index_out,
        manifest_path=args.manifest,
    )
    print(f"chunks={count}")
    print(f"embedding_dim={dim}")
    print(f"index_out={args.index_out}")
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
