"""
用途：使用真实中文医学问题对已构建的 BGE + FAISS 索引执行冒烟验证。

默认输入：
    configs/chinese_retrieval_tests.yaml
    data/index/chinese/strict/faiss.index
    data/index/chinese/strict/chunk_metadata.jsonl

默认输出：
    data/index/chinese/strict/retrieval_smoke_report.json

说明：
    每个测试问题配置一组预期关键词。程序只要求 top-k 证据中至少命中一个
    关键词，并检查来源路径和页码是否完整；这不是医学准确性终评。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import yaml

from rag_medical.common.build_embeddings import model_path_from_config, resolve_device
from rag_medical.common.build_faiss_index import read_metadata_jsonl
from rag_medical.common.search_chunks import (
    build_search_results,
    encode_query,
    load_sentence_transformer,
    search_index,
)


def read_query_config(path: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = config.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"queries must be a non-empty list: {path}")
    return queries


def evaluate_retrieval_result(
    query_spec: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_keywords = [str(value) for value in query_spec.get("expected_any", []) if str(value)]
    searchable_text = "\n".join(
        f"{result.get('title', '')}\n{result.get('section', '')}\n{result.get('text', '')}"
        for result in results
    ).casefold()
    matched_keywords = [
        keyword for keyword in expected_keywords if keyword.casefold() in searchable_text
    ]
    missing_traceability = [
        str(result.get("chunk_id") or result.get("row_index"))
        for result in results
        if not result.get("source_path") or not result.get("source_pages")
    ]

    top_hits = [
        {
            "rank": result.get("rank"),
            "score": result.get("score"),
            "chunk_id": result.get("chunk_id"),
            "title": result.get("title"),
            "section": result.get("section"),
            "source_path": result.get("source_path"),
            "source_pages": result.get("source_pages"),
            "text_preview": str(result.get("text") or "")[:300],
        }
        for result in results[:5]
    ]
    passed = bool(results) and bool(matched_keywords) and not missing_traceability
    return {
        "id": query_spec.get("id", ""),
        "query": query_spec.get("query", ""),
        "expected_any": expected_keywords,
        "matched_keywords": matched_keywords,
        "result_count": len(results),
        "missing_traceability": missing_traceability,
        "passed": passed,
        "top_hits": top_hits,
    }


def run_retrieval_smoke(
    queries: list[dict[str, Any]],
    index: faiss.Index,
    metadata_records: list[dict[str, Any]],
    model: Any,
    default_top_k: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for query_spec in queries:
        query = str(query_spec.get("query") or "").strip()
        if not query:
            raise ValueError(f"empty query in test: {query_spec.get('id', '')}")
        top_k = int(query_spec.get("top_k") or default_top_k)
        query_embedding = encode_query(model, query)
        hits = search_index(index, query_embedding, top_k)
        search_results = build_search_results(hits, metadata_records)
        evaluation = evaluate_retrieval_result(query_spec, search_results)
        results.append(evaluation)
        print(
            f"[{evaluation['id']}] passed={evaluation['passed']} "
            f"matched={evaluation['matched_keywords']}"
        )

    passed_count = sum(bool(result["passed"]) for result in results)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(results) - passed_count,
        "all_passed": passed_count == len(results),
        "results": results,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured semantic retrieval smoke tests.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("configs/chinese_retrieval_tests.yaml"),
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data/index/chinese/strict"))
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model_path or model_path_from_config(args.embedding_config)
    report_path = args.report or args.index_dir / "retrieval_smoke_report.json"
    index_path = args.index_dir / "faiss.index"
    metadata_path = args.index_dir / "chunk_metadata.jsonl"

    required_paths = [args.queries, index_path, metadata_path]
    if model_path is not None:
        required_paths.append(model_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if model_path is None or missing:
        print(f"missing retrieval inputs: {missing}", file=sys.stderr)
        return 2

    queries = read_query_config(args.queries)
    index = faiss.read_index(str(index_path))
    metadata_records = read_metadata_jsonl(metadata_path)
    model = load_sentence_transformer(model_path, resolve_device(args.device))
    report = run_retrieval_smoke(
        queries,
        index,
        metadata_records,
        model,
        args.top_k,
    )
    write_report(report_path, report)

    print(f"queries={report['query_count']}")
    print(f"passed={report['passed_count']}")
    print(f"failed={report['failed_count']}")
    print(f"report={report_path}")
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
