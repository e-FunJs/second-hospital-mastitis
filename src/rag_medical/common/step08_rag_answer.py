"""
用途：执行第一层 RAG，检索并重排中英文证据。
输入：用户问题、固定的中文/英文 strict FAISS index、chunk_metadata.jsonl、模型配置。
输出：默认 data/rag/answers/bilingual/*_evidence.json、retrieval trace 及 query plan。
说明：本文件不再构造 prompt；证据先由 step09 清洗，再由 step10 构造报告 prompt。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import yaml

from rag_medical.common.expansion.aliases import load_aliases
from rag_medical.common.expansion.models import QueryRecord, RetrievalStage
from rag_medical.common.expansion.query_expand import build_query_plan
from rag_medical.common.expansion.query_plan_io import write_query_plan
from rag_medical.common.step03_build_embeddings import model_path_from_config, resolve_device
from rag_medical.common.step04_build_faiss_index import read_metadata_jsonl
from rag_medical.common.step11_generate_answer import generate_with_local_llm, load_llm_config
from rag_medical.common.step06_search_chunks import (
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


def make_evidence_records(
    search_results: list[dict[str, Any]],
    retrieval_query: str = "",
) -> list[dict[str, Any]]:
    evidence_records: list[dict[str, Any]] = []
    retrieval_stage: RetrievalStage = RetrievalStage.R0
    query_index: int = 0
    for index, result in enumerate(search_results, start=1):
        # 语言必须来自索引 metadata；缺失时保留空值，不能根据索引名称或文本猜测。
        language: str = str(result.get("language") or "")
        evidence = {
            "evidence_id": f"E{index}",
            "rank": result.get("rank", index),
            "score": result.get("score", ""),
            "dense_score": result.get("dense_score", ""),
            "rerank_score": result.get("rerank_score", ""),
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
            # matched_text 保留 BGE 实际命中的精确子块；text 则是给 Qwen 阅读的
            # 同父块上下文。两者同时保存，医生可追溯“为何命中”和“如何补全”。
            "matched_text": result.get("text", ""),
            "context_mode": result.get("context_mode", "hit_only"),
            "context_chunk_ids": result.get("context_chunk_ids", []),
            "text": result.get("context_text") or result.get("text", ""),
            # 当前检索流程只有原始检索阶段，因此这里固定记录 R0 和查询序号 0。
            "retrieval_stage": retrieval_stage.value,
            "retrieval_query": retrieval_query,
            "query_index": query_index,
            "language": language,
        }
        evidence_records.append(evidence)
    return evidence_records


# -----------------------------------------------------------------------------
# 输出文件
# -----------------------------------------------------------------------------
# step08 保存 evidence JSON 和同源的精简 trace，仍不生成未经 step09 清洗的 prompt。


RETRIEVAL_TRACE_FIELDS: tuple[str, ...] = (
    "evidence_id",
    "retrieval_stage",
    "retrieval_query",
    "query_index",
    "language",
    "chunk_id",
    "score",
)


def slugify_query(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", text).strip("_")
    return slug[:max_length] or "rag_query"


def write_retrieval_trace(path: Path, evidence_records: list[dict[str, Any]]) -> None:
    """覆盖写入单个问题的精简检索轨迹，每行直接投影自同一条 evidence。"""
    trace_lines = [
        json.dumps(
            {field_name: evidence[field_name] for field_name in RETRIEVAL_TRACE_FIELDS},
            ensure_ascii=False,
        )
        for evidence in evidence_records
    ]
    content = "\n".join(trace_lines)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def write_rag_package(
    output_dir: Path,
    question: str,
    evidence_records: list[dict[str, Any]],
    query_slug: str,
    query_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / f"{query_slug}_evidence.json"
    trace_path = output_dir / f"{query_slug}_retrieval_trace.jsonl"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "evidence_count": len(evidence_records),
        "evidence": evidence_records,
    }
    if query_metadata:
        payload["query_metadata"] = query_metadata
    evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_retrieval_trace(trace_path, evidence_records)
    return {"evidence_path": evidence_path}


def _try_write_query_plan(
    question: str,
    output_dir: Path,
    query_slug: str,
) -> Path | None:
    """独立生成查询计划；失败只记录警告，不影响既有 evidence 和 trace。"""
    final_path = output_dir / f"{query_slug}_query_plan.json"
    try:
        config = yaml.safe_load(
            Path("configs/evidence_expansion.yaml").read_text(encoding="utf-8")
        )
        expansion_config = config["query_expansion"]
        aliases = load_aliases(Path(expansion_config["aliases_path"]))
        query_record = QueryRecord(
            query_id="Q-" + hashlib.sha256(question.encode("utf-8")).hexdigest(),
            query_text=question,
        )
        queries = build_query_plan(
            query_record,
            aliases,
            expansion_config["max_queries"],
            min_combination_drugs=expansion_config["min_combination_drugs"],
            max_combination_drugs=expansion_config["max_combination_drugs"],
        )
        with tempfile.TemporaryDirectory(
            prefix=".query_plan_tmp_",
            dir=output_dir,
        ) as temporary_directory:
            temporary_dir = Path(temporary_directory)
            generated_path = write_query_plan(query_record, queries, temporary_dir)
            staged_path = temporary_dir / final_path.name
            if generated_path != staged_path:
                os.replace(generated_path, staged_path)
            os.replace(staged_path, final_path)
        return final_path
    except Exception:
        logging.warning("query plan generation failed", exc_info=True)
        return None


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
    candidate_k: int,
    reranker: Any,
    device: str,
    context_max_units: int,
) -> list[dict[str, Any]]:
    index = faiss.read_index(str(index_path))
    metadata_records = read_metadata_jsonl(metadata_path)
    model = load_sentence_transformer(model_path, device)
    query_embedding = encode_query(model, question)
    hits = search_index(index, query_embedding, candidate_k)

    # BGE-M3 负责扩大召回；reranker 只比较问题与实际命中的子块正文。
    # 先重排、后扩展父块，可避免父块中的无关内容反过来影响相关性判断。
    pairs = [
        (question, str(metadata_records[int(hit["row_index"])].get("text") or ""))
        for hit in hits
    ]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
    for hit, rerank_score in zip(hits, rerank_scores):
        hit["dense_score"] = float(hit["score"])
        hit["rerank_score"] = float(rerank_score)
        hit["score"] = float(rerank_score)
    hits = sorted(hits, key=lambda item: item["rerank_score"], reverse=True)[:top_k]

    search_results = build_search_results(hits, metadata_records, context_max_units)
    for result, hit in zip(search_results, hits):
        result["dense_score"] = round(float(hit["dense_score"]), 6)
        result["rerank_score"] = round(float(hit["rerank_score"]), 6)
    return make_evidence_records(search_results, question)


# -----------------------------------------------------------------------------
# 双语查询编排
# -----------------------------------------------------------------------------
# BGE-M3 能跨语言编码，但不会生成翻译文本。该项目固定检索中英文 strict 语料：
# 原问题负责同语言库，Qwen 翻译后的问题负责另一语言库，最后统一返回两边证据。


def detect_query_language(question: str) -> str:
    """本项目当前只处理中文或英文医学问题；含 CJK 字符即按中文处理。"""

    return "zh" if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", question) else "en"


def translate_query(question: str, llm_config_path: Path) -> tuple[str, str]:
    """使用本地 Qwen 翻译为另一语言；不让模型回答医学问题或补充外部事实。"""

    source_language = detect_query_language(question)
    target_language = "English" if source_language == "zh" else "Chinese"
    system_prompt = (
        "You are a clinical information-retrieval query translator. "
        "Translate only; do not answer the question or add medical facts. "
        "Preserve drug names, numbers, doses, units, disease subtypes, and uncertainty exactly. "
        "Return only the translated query without a label, explanation, quotation marks, or Markdown."
    )
    prompt = f"Translate the following medical retrieval query into {target_language}:\n{question}"
    llm_config = load_llm_config(llm_config_path)
    translated = generate_with_local_llm(
        prompt,
        Path(llm_config.model_path),
        llm_config,
        system_prompt=system_prompt,
        # 检索式通常只有一行；限制长度可防止基础模型持续生成解释性文本。
        max_new_tokens=128,
    ).strip()
    if not translated:
        raise ValueError("local Qwen returned an empty translated query")

    # 翻译完成后释放 Qwen 的 GPU 缓存，后续 BGE 检索才能稳定加载。
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - CPU 环境不影响后续检索。
        pass
    return translated, ("en" if source_language == "zh" else "zh")


def merge_bilingual_evidence(
    primary_evidence: list[dict[str, Any]],
    translated_evidence: list[dict[str, Any]],
    primary_language: str,
    translated_language: str,
) -> list[dict[str, Any]]:
    """将两套 strict 检索结果直接合并，并重新编号为一个可引用的 evidence 列表。"""

    merged = primary_evidence + translated_evidence
    for index, evidence in enumerate(merged, start=1):
        evidence["evidence_id"] = f"E{index}"
        evidence["retrieval_language"] = (
            primary_language if index <= len(primary_evidence) else translated_language
        )
    return merged


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
# 固定双语检索：索引路径有默认值，允许命令行覆盖路径但不允许关闭任一语言库。


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the first-layer bilingual RAG evidence package.")
    parser.add_argument("question", help="Question to retrieve evidence for.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=20, help="每个语言索引送入 reranker 的候选数。")
    parser.add_argument(
        "--context-max-units",
        type=int,
        default=512,
        help="每条 evidence 返回给 LLM 的最大近似 token 数。",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--model-path", type=Path, help="Override embedding.local_model_path in config.")
    parser.add_argument(
        "--reranker-path",
        type=Path,
        default=Path("models/reranker/bge-reranker-v2-m3"),
        help="本地多语言 reranker 模型路径。",
    )
    parser.add_argument(
        "--chinese-index",
        type=Path,
        default=Path("data/index/chinese/strict/faiss.index"),
        help="中文 strict FAISS 索引。",
    )
    parser.add_argument(
        "--chinese-metadata",
        type=Path,
        default=Path("data/index/chinese/strict/chunk_metadata.jsonl"),
        help="中文 strict 索引的 metadata。",
    )
    parser.add_argument(
        "--english-index",
        type=Path,
        default=Path("data/index/english/strict/faiss.index"),
        help="英文 strict FAISS 索引。",
    )
    parser.add_argument(
        "--english-metadata",
        type=Path,
        default=Path("data/index/english/strict/chunk_metadata.jsonl"),
        help="英文 strict 索引的 metadata。",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=Path("configs/llm.yaml"),
        help="本地 Qwen 配置；每次固定双语检索都用于翻译问题。",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rag/answers/bilingual"),
    )
    parser.add_argument("--query-slug", help="Stable output filename prefix. Defaults to a slug from question.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model_path or model_path_from_config(args.config)
    if model_path is None:
        print("model path not provided and not found in config", file=sys.stderr)
        return 2
    required_paths = [
        args.chinese_index,
        args.chinese_metadata,
        args.english_index,
        args.english_metadata,
        args.llm_config,
        model_path,
        args.reranker_path,
    ]
    for path in required_paths:
        if not path.exists():
            print(f"input not found: {path}", file=sys.stderr)
            return 2

    if args.candidate_k < args.top_k:
        print("candidate-k must be greater than or equal to top-k", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    original_language = detect_query_language(args.question)
    translated_question, translated_language = translate_query(args.question, args.llm_config)
    chinese_question = args.question if original_language == "zh" else translated_question
    english_question = args.question if original_language == "en" else translated_question

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(str(args.reranker_path), device=device)

    # 两个 strict 索引始终都会检索；提问语言只决定哪边直接使用原问题。
    chinese_evidence = retrieve_evidence(
        question=chinese_question,
        index_path=args.chinese_index,
        metadata_path=args.chinese_metadata,
        model_path=model_path,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        reranker=reranker,
        device=device,
        context_max_units=args.context_max_units,
    )
    english_evidence = retrieve_evidence(
        question=english_question,
        index_path=args.english_index,
        metadata_path=args.english_metadata,
        model_path=model_path,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        reranker=reranker,
        device=device,
        context_max_units=args.context_max_units,
    )
    evidence_records = merge_bilingual_evidence(chinese_evidence, english_evidence, "zh", "en")
    query_metadata: dict[str, Any] = {
        "original_query": args.question,
        "original_language": original_language,
        "chinese_query": chinese_question,
        "english_query": english_question,
        "translation_language": translated_language,
        "chinese_index": str(args.chinese_index),
        "english_index": str(args.english_index),
        "candidate_k_per_language": args.candidate_k,
        "final_top_k_per_language": args.top_k,
        "reranker_path": str(args.reranker_path),
    }
    query_slug = args.query_slug or slugify_query(args.question)
    outputs = write_rag_package(
        args.output_dir,
        args.question,
        evidence_records,
        query_slug,
        query_metadata,
    )
    _try_write_query_plan(args.question, args.output_dir, query_slug)

    print(f"evidence_count={len(evidence_records)}")
    print(f"chinese_query={chinese_question}")
    print(f"english_query={english_question}")
    print(f"evidence_path={outputs['evidence_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
