"""
用途：执行第一层 RAG：检索证据并拼接成可交给 LLM 的 prompt。
输入：用户问题、FAISS index、chunk_metadata.jsonl、embedding 模型配置。
输出：data/rag/answers/*_evidence.json 与 *_prompt.txt。
说明：本文件不直接调用 LLM；回答生成由 generate_answer.py 完成。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss

from rag_medical.build_embeddings import model_path_from_config, resolve_device
from rag_medical.build_faiss_index import read_metadata_jsonl
from rag_medical.search_chunks import (
    build_search_results,
    encode_query,
    load_sentence_transformer,
    search_index,
)


# -----------------------------------------------------------------------------
# Evidence 记录构造
# -----------------------------------------------------------------------------
# 第一层 RAG 不让 LLM 生成答案，只把检索结果整理成可追溯的 evidence package。
# 每条 evidence 都有稳定编号 E1/E2/...；后续 LLM 回答时必须引用这些编号。


def make_citation(record: dict[str, Any]) -> str:
    pmcid = str(record.get("pmcid") or "")
    year = str(record.get("year") or "")
    title = str(record.get("title") or "")
    section = str(record.get("section") or "")
    return " | ".join(part for part in [pmcid, year, title, section] if part)


def make_evidence_records(search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_records: list[dict[str, Any]] = []
    for index, result in enumerate(search_results, start=1):
        evidence = {
            "evidence_id": f"E{index}",
            "rank": result.get("rank", index),
            "score": result.get("score", ""),
            "chunk_id": result.get("chunk_id", ""),
            "source_type": result.get("source_type", ""),
            "pmcid": result.get("pmcid", ""),
            "pmid": result.get("pmid", ""),
            "doi": result.get("doi", ""),
            "title": result.get("title", ""),
            "journal": result.get("journal", ""),
            "year": result.get("year", ""),
            "section": result.get("section", ""),
            "source_url": result.get("source_url", ""),
            "citation": make_citation(result),
            "text": result.get("text", ""),
        }
        evidence_records.append(evidence)
    return evidence_records


# -----------------------------------------------------------------------------
# RAG Prompt 构造
# -----------------------------------------------------------------------------
# 这是给第二层 LLM 用的 prompt。这里刻意写入医学安全约束：只能基于 evidence、
# 证据不足要说明不足、不得给具体患者治疗决策。现在只生成 prompt，不调用模型。


def build_rag_prompt(question: str, evidence_records: list[dict[str, Any]]) -> str:
    evidence_blocks = []
    for evidence in evidence_records:
        evidence_blocks.append(
            "\n".join(
                [
                    f"[{evidence['evidence_id']}] {evidence.get('citation', '')}",
                    f"Score: {evidence.get('score', '')}",
                    f"Chunk: {evidence.get('chunk_id', '')}",
                    f"Text: {evidence.get('text', '')}",
                ]
            )
        )

    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence retrieved."
    return f"""你是医学文献 RAG 助手。请严格遵守以下规则：
1. 只允许基于下面给定的 Evidence 回答，不要使用未给出的外部知识。
2. 每个关键结论后必须引用证据编号，例如 [E1]、[E2]。
3. 证据不足时明确说证据不足，不要补充未给出的内容。
4. 不要给具体患者做诊断或治疗决策，只能总结文献证据。
5. 如果证据之间存在不一致，要说明不确定性。

Question:
{question}

Evidence:
{evidence_text}

请输出：
- 简短结论
- 证据依据
- 不确定性或证据不足之处
"""


# -----------------------------------------------------------------------------
# 输出文件
# -----------------------------------------------------------------------------
# evidence JSON 用于程序读取；prompt txt 用于人工检查或之后直接交给 LLM。


def slugify_query(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", text).strip("_")
    return slug[:max_length] or "rag_query"


def write_rag_package(
    output_dir: Path,
    question: str,
    evidence_records: list[dict[str, Any]],
    prompt: str,
    query_slug: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / f"{query_slug}_evidence.json"
    prompt_path = output_dir / f"{query_slug}_prompt.txt"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "evidence_count": len(evidence_records),
        "evidence": evidence_records,
    }
    evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    return {"evidence_path": evidence_path, "prompt_path": prompt_path}


# -----------------------------------------------------------------------------
# 检索编排
# -----------------------------------------------------------------------------
# 这里把 query -> BGE query embedding -> FAISS top-k -> evidence package 串起来。
# 注意：这一步仍然不是“回答生成”，只是为后续 LLM 准备可靠上下文。


def retrieve_evidence(
    question: str,
    index_path: Path,
    metadata_path: Path,
    model_path: Path,
    top_k: int,
    device: str,
) -> list[dict[str, Any]]:
    index = faiss.read_index(str(index_path))
    metadata_records = read_metadata_jsonl(metadata_path)
    model = load_sentence_transformer(model_path, device)
    query_embedding = encode_query(model, question)
    hits = search_index(index, query_embedding, top_k)
    search_results = build_search_results(hits, metadata_records)
    return make_evidence_records(search_results)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
# 默认输出到 data/rag/answers；该目录会被 .gitignore 忽略，只在服务器本地保留。


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build first-layer RAG evidence package and prompt.")
    parser.add_argument("question", help="Question to retrieve evidence for.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--index", type=Path, default=Path("data/index/faiss.index"))
    parser.add_argument("--metadata", type=Path, default=Path("data/index/chunk_metadata.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--model-path", type=Path, help="Override embedding.local_model_path in config.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/rag/answers"))
    parser.add_argument("--query-slug", help="Stable output filename prefix. Defaults to a slug from question.")
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
    evidence_records = retrieve_evidence(
        question=args.question,
        index_path=args.index,
        metadata_path=args.metadata,
        model_path=model_path,
        top_k=args.top_k,
        device=device,
    )
    prompt = build_rag_prompt(args.question, evidence_records)
    query_slug = args.query_slug or slugify_query(args.question)
    outputs = write_rag_package(args.output_dir, args.question, evidence_records, prompt, query_slug)

    print(f"evidence_count={len(evidence_records)}")
    print(f"evidence_path={outputs['evidence_path']}")
    print(f"prompt_path={outputs['prompt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
