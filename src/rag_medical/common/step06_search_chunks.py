"""
用途：直接查询 FAISS 索引，查看与问题最相近的 RAG chunk。
输入：查询文本、默认 data/index/combined/faiss.index 与 chunk_metadata.jsonl。
输出：终端打印检索结果；可选 --json-out 保存 JSON。
说明：这是检索调试工具，不生成 LLM prompt，也不生成最终回答。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag_medical.common.step03_build_embeddings import (
    build_embedding_text,
    model_path_from_config,
    resolve_device,
)
from rag_medical.common.step04_build_faiss_index import read_metadata_jsonl


# -----------------------------------------------------------------------------
# 查询向量化
# -----------------------------------------------------------------------------
# 检索时也使用 BGE-M3。query 会被构造成和 chunk embedding 类似的文本格式，
# 但没有 title/section，只保留 Text。这样 query 和 corpus 使用同一个 encoder 空间。


def load_sentence_transformer(model_path: Path, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), device=device)


def encode_query(model: Any, query: str) -> np.ndarray:
    query_text = build_embedding_text({"text": query})
    embedding = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
    vector = np.asarray(embedding, dtype=np.float32)
    faiss.normalize_L2(vector)
    return vector[0]


# -----------------------------------------------------------------------------
# FAISS 检索与结果组装
# -----------------------------------------------------------------------------
# FAISS 返回的是 row index；这里再回到 metadata 中取 title、section、text 等证据字段。


def search_index(index: faiss.Index, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1).copy()
    faiss.normalize_L2(vector)
    scores, indices = index.search(vector, top_k)
    hits: list[dict[str, Any]] = []
    for score, row_index in zip(scores[0], indices[0]):
        if row_index < 0:
            continue
        hits.append({"row_index": int(row_index), "score": float(score)})
    return hits


def text_unit_count(text: str) -> int:
    """用与中文分块一致的近似单位计数控制上下文，不在句子中间截断。"""

    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def build_context(
    metadata: dict[str, Any],
    by_chunk_id: dict[str, dict[str, Any]],
    max_units: int,
) -> tuple[str, list[str], str]:
    """返回给 LLM 的上下文，严格限制在同一语义父块内。

    BGE 检索仍以子块为粒度；命中后优先返还完整父块。父块太长时，只按
    previous/next 补充紧邻子块，且每次加入都保持完整子块，从而避免 A+B 跨主题
    拼接和“根据治疗”式的半句话截断。
    """

    hit_id = str(metadata.get("chunk_id") or "")
    hit_text = str(metadata.get("text") or "")
    parent_text = str(metadata.get("parent_text") or "")
    parent_id = str(metadata.get("parent_id") or "")
    if parent_text and text_unit_count(parent_text) <= max_units:
        child_ids = [
            str(record.get("chunk_id") or "")
            for record in by_chunk_id.values()
            if str(record.get("parent_id") or "") == parent_id
        ]
        return parent_text, child_ids, "full_parent"

    # 老索引没有父子字段时退化为原始命中，不改变既有英文语料的行为。
    if not parent_id or hit_id not in by_chunk_id:
        return hit_text, [hit_id] if hit_id else [], "hit_only"

    selected = [metadata]
    selected_ids = {hit_id}
    # 先前后各扩一次，必要时再继续向外；加入失败后停止该方向，避免跳过中间语义。
    directions = ["previous_chunk_id", "next_chunk_id"]
    cursors = {key: str(metadata.get(key) or "") for key in directions}
    active = True
    while active:
        active = False
        for direction in directions:
            candidate_id = cursors[direction]
            if not candidate_id or candidate_id in selected_ids:
                continue
            candidate = by_chunk_id.get(candidate_id)
            if not candidate or str(candidate.get("parent_id") or "") != parent_id:
                cursors[direction] = ""
                continue
            candidate_text = str(candidate.get("text") or "")
            proposed_text = "\n".join(str(item.get("text") or "") for item in selected + [candidate])
            if text_unit_count(proposed_text) > max_units:
                cursors[direction] = ""
                continue
            selected.append(candidate)
            selected_ids.add(candidate_id)
            cursors[direction] = str(candidate.get(direction) or "")
            active = True

    # 依文章内句子位置排序，而非按加入顺序，保证交给 LLM 的文本可自然阅读。
    selected.sort(key=lambda item: int(item.get("sentence_start") or 0))
    return "\n".join(str(item.get("text") or "") for item in selected), [
        str(item.get("chunk_id") or "") for item in selected
    ], "neighbor_children"


def build_search_results(
    hits: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    context_max_units: int = 512,
) -> list[dict[str, Any]]:
    """把 FAISS 命中组装为证据，并在不跨父块的前提下补足上下文。"""

    by_chunk_id = {
        str(record.get("chunk_id") or ""): record
        for record in metadata_records
        if record.get("chunk_id")
    }
    results: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        row_index = int(hit["row_index"])
        if row_index >= len(metadata_records):
            raise IndexError(f"FAISS returned row {row_index}, but metadata has {len(metadata_records)} rows")
        metadata = metadata_records[row_index]
        context_text, context_chunk_ids, context_mode = build_context(
            metadata,
            by_chunk_id,
            context_max_units,
        )
        results.append(
            {
                "rank": rank,
                "score": round(float(hit["score"]), 6),
                "context_text": context_text,
                "context_chunk_ids": context_chunk_ids,
                "context_mode": context_mode,
                **metadata,
            }
        )
    return results


def print_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(f"[{result['rank']}] score={result['score']} {result.get('chunk_id', '')}")
        print(f"    {result.get('title', '')}")
        print(f"    {result.get('year', '')} | {result.get('journal', '')} | {result.get('section', '')}")
        text = str(result.get("context_text") or result.get("text") or "")
        preview = text[:600] + ("..." if len(text) > 600 else "")
        print(f"    context={result.get('context_mode', 'hit_only')} | {preview}")
        print()


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
# 这个脚本的目标是验证 retrieval 是否靠谱，还不是最终 RAG 问答。


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search semantic article chunks with BGE + FAISS.")
    parser.add_argument("query", help="Question or search text.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/index/combined/faiss.index"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/index/combined/chunk_metadata.jsonl"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--model-path", type=Path, help="Override embedding.local_model_path in config.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--context-max-units",
        type=int,
        default=512,
        help="LLM evidence 的最大近似 token 数；不会在子块或句子中间截断。",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--json-out", type=Path, help="Optional path to save search results as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model_path or model_path_from_config(args.config)
    if model_path is None:
        print("model path not provided and not found in config", file=sys.stderr)
        return 2
    for path in [args.index, args.metadata, model_path]:
        if not path.exists():
            print(f"input not found: {path}", file=sys.stderr)
            return 2

    device = resolve_device(args.device)
    index = faiss.read_index(str(args.index))
    metadata_records = read_metadata_jsonl(args.metadata)
    model = load_sentence_transformer(model_path, device)
    query_embedding = encode_query(model, args.query)
    hits = search_index(index, query_embedding, args.top_k)
    results = build_search_results(hits, metadata_records, args.context_max_units)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
