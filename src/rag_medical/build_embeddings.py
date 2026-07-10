from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


# -----------------------------------------------------------------------------
# JSONL 读写与文本规范化
# -----------------------------------------------------------------------------
# 这个脚本的输入是 chunk 阶段产出的 rag_chunks.jsonl。
# 每一行是一个 chunk；本阶段只负责把原文片段编码成向量，不改写、不总结原文。


def normalize_space(text: str | None) -> str:
    return " ".join((text or "").split())


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


# -----------------------------------------------------------------------------
# embedding 输入文本构造
# -----------------------------------------------------------------------------
# BGE 看到的文本不是只有 chunk.text，而是 Title + Section + Text。
# 原因：有些 chunk 本身很短，加入题名和章节名能帮模型理解它属于治疗、影像、诊断
# 还是预后语境。PMCID/DOI 这类追溯字段不放入 embedding 文本，因为它们不是语义内容。


def build_embedding_text(chunk: dict[str, Any]) -> str:
    title = normalize_space(str(chunk.get("title") or ""))
    section = normalize_space(str(chunk.get("section") or ""))
    text = normalize_space(str(chunk.get("text") or ""))

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if section:
        parts.append(f"Section: {section}")
    parts.append(f"Text: {text}")
    return "\n".join(parts)


def build_metadata_record(chunk: dict[str, Any], row_index: int) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "chunk_id": chunk.get("chunk_id", ""),
        "source_type": chunk.get("source_type", "pmc_full_text"),
        "pmcid": chunk.get("pmcid", ""),
        "pmid": chunk.get("pmid", ""),
        "doi": chunk.get("doi", ""),
        "title": chunk.get("title", ""),
        "journal": chunk.get("journal", ""),
        "year": chunk.get("year", ""),
        "section": chunk.get("section", ""),
        "chunk_index": chunk.get("chunk_index", ""),
        "source_url": chunk.get("source_url", ""),
        "source_path": chunk.get("source_path", ""),
        "source_paragraph_indices": chunk.get("source_paragraph_indices", []),
        "sentence_start": chunk.get("sentence_start", ""),
        "sentence_end": chunk.get("sentence_end", ""),
        "word_count": chunk.get("word_count", ""),
        "char_count": chunk.get("char_count", ""),
        "text": chunk.get("text", ""),
    }


# -----------------------------------------------------------------------------
# 模型加载与批量编码
# -----------------------------------------------------------------------------
# 和你之前训练代码相似的地方：读取数据、加载模型、构造 batch、送进模型。
# 不同点：这里关闭训练语义，不需要 labels/loss/optimizer/backward；SentenceTransformer.encode
# 内部会做 tokenizer 和 pooling，我们只拿最终 embedding。


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - 没装 torch 时让后续模型加载给出明确错误。
        return "cpu"


def load_sentence_transformer(model_path: Path, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), device=device)


def encode_texts(
    model: Any,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


# -----------------------------------------------------------------------------
# 输出文件
# -----------------------------------------------------------------------------
# embeddings.npy 存矩阵，metadata.jsonl 存可读追溯信息。两者用 row_index 对齐：
# embeddings[row_index] 就对应 metadata 里同一个 row_index 的 chunk。


def write_embedding_outputs(
    embeddings: np.ndarray,
    metadata_records: list[dict[str, Any]],
    embedding_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    model_path: str,
    input_path: str,
    batch_size: int,
    device: str,
) -> None:
    if len(metadata_records) != int(embeddings.shape[0]):
        raise ValueError("metadata row count must match embedding row count")

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embeddings.astype(np.float32, copy=False))
    write_jsonl(metadata_path, metadata_records)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": input_path,
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "model_path": model_path,
        "batch_size": batch_size,
        "device": device,
        "normalize_embeddings": True,
        "dtype": str(embeddings.dtype),
        "chunk_count": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# 配置读取与主流程
# -----------------------------------------------------------------------------
# 默认模型路径来自 configs/embedding.yaml；命令行 --model-path 可以覆盖它。
# 这样后续换 bge-base、bge-m3 或其它 embedding 模型时，不需要改代码。


def model_path_from_config(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    value = config.get("embedding", {}).get("local_model_path")
    return Path(value) if value else None


def build_embeddings(
    input_path: Path,
    model_path: Path,
    embedding_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    batch_size: int,
    device: str,
    limit: int | None = None,
) -> tuple[int, int]:
    chunks = read_jsonl(input_path)
    if limit is not None:
        chunks = chunks[:limit]
    if not chunks:
        raise ValueError(f"no chunks found in {input_path}")

    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]
    metadata_records = [build_metadata_record(chunk, row_index=index) for index, chunk in enumerate(chunks)]

    model = load_sentence_transformer(model_path, device)
    embeddings = encode_texts(model, embedding_texts, batch_size=batch_size, normalize_embeddings=True)

    # 关键检查：如果这里不一致，后续 FAISS 检索会出现“向量命中 A，metadata 指向 B”的严重错位。
    if embeddings.shape[0] != len(metadata_records):
        raise RuntimeError("embedding count and metadata count are not aligned")

    write_embedding_outputs(
        embeddings=embeddings,
        metadata_records=metadata_records,
        embedding_path=embedding_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        model_path=str(model_path),
        input_path=str(input_path),
        batch_size=batch_size,
        device=device,
    )
    return int(embeddings.shape[0]), int(embeddings.shape[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BGE embeddings for semantic article chunks.")
    parser.add_argument("--input", type=Path, default=Path("data/articles/processed/rag_chunks.jsonl"))
    parser.add_argument("--embedding-out", type=Path, default=Path("data/index/chunk_embeddings.npy"))
    parser.add_argument("--metadata-out", type=Path, default=Path("data/index/chunk_metadata.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/index/embedding_manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--model-path", type=Path, help="Override embedding.local_model_path in config.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Only embed the first N chunks for smoke tests.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model_path or model_path_from_config(args.config)
    if model_path is None:
        print("model path not provided and not found in config", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    if not model_path.exists():
        print(f"model path not found: {model_path}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    count, dim = build_embeddings(
        input_path=args.input,
        model_path=model_path,
        embedding_path=args.embedding_out,
        metadata_path=args.metadata_out,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        device=device,
        limit=args.limit,
    )
    print(f"chunks={count}")
    print(f"embedding_dim={dim}")
    print(f"embedding_out={args.embedding_out}")
    print(f"metadata_out={args.metadata_out}")
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
