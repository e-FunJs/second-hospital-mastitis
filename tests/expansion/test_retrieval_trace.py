"""
用途：验证 R0 evidence 溯源字段和逐问题 retrieval trace 的数据一致性。
输入：测试内构造的检索结果及查询文本。
输出：pytest 断言结果。
不做什么：不加载 BGE、FAISS、reranker、Qwen 或真实索引。
"""

import json

from rag_medical.common.step08_rag_answer import make_evidence_records, write_rag_package


def make_search_results() -> list[dict]:
    """构造顺序、排名与分数均固定的检索结果。"""
    return [
        {
            "rank": 1,
            "score": 0.91,
            "dense_score": 0.72,
            "rerank_score": 0.91,
            "chunk_id": "CNKI-001::001",
            "language": "zh",
            "text": "异烟肼治疗相关证据。",
        },
        {
            "rank": 2,
            "score": 0.83,
            "dense_score": 0.68,
            "rerank_score": 0.83,
            "chunk_id": "PMC001::001",
            "text": "Evidence without language metadata.",
        },
    ]


def test_retrieval_trace_matches_evidence_without_changing_ranking(tmp_path) -> None:
    search_results = make_search_results()
    evidence_records = make_evidence_records(search_results, "异烟肼治疗乳腺炎")

    write_rag_package(
        output_dir=tmp_path,
        question="异烟肼治疗乳腺炎",
        evidence_records=evidence_records,
        query_slug="isoniazid",
    )

    evidence_payload = json.loads((tmp_path / "isoniazid_evidence.json").read_text())
    trace_records = [
        json.loads(line)
        for line in (tmp_path / "isoniazid_retrieval_trace.jsonl").read_text().splitlines()
    ]

    assert [item["chunk_id"] for item in evidence_payload["evidence"]] == [
        "CNKI-001::001",
        "PMC001::001",
    ]
    assert [item["score"] for item in evidence_payload["evidence"]] == [0.91, 0.83]
    assert [item["rank"] for item in evidence_payload["evidence"]] == [1, 2]
    assert trace_records == [
        {
            "evidence_id": item["evidence_id"],
            "retrieval_stage": item["retrieval_stage"],
            "retrieval_query": item["retrieval_query"],
            "query_index": item["query_index"],
            "language": item["language"],
            "chunk_id": item["chunk_id"],
            "score": item["score"],
        }
        for item in evidence_payload["evidence"]
    ]
    assert trace_records[0]["retrieval_stage"] == "R0"
    assert trace_records[0]["retrieval_query"] == "异烟肼治疗乳腺炎"
    assert trace_records[0]["query_index"] == 0
    assert trace_records[0]["language"] == "zh"
    assert trace_records[1]["language"] == ""


def test_retrieval_trace_overwrites_existing_file(tmp_path) -> None:
    evidence_records = make_evidence_records(make_search_results()[:1], "第二次检索")
    trace_path = tmp_path / "same_retrieval_trace.jsonl"
    trace_path.write_text('{"stale": true}\n', encoding="utf-8")

    write_rag_package(
        output_dir=tmp_path,
        question="第二次检索",
        evidence_records=evidence_records,
        query_slug="same",
    )

    trace_records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(trace_records) == 1
    assert trace_records[0]["retrieval_query"] == "第二次检索"
    assert "stale" not in trace_records[0]
