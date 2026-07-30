"""真实检索冒烟结果判定逻辑测试。"""

from __future__ import annotations

from rag_medical.common.retrieval_smoke import evaluate_retrieval_result


def test_expected_keyword_and_traceability_pass() -> None:
    query = {
        "id": "anti_tb",
        "query": "联合抗结核治疗",
        "expected_any": ["利福平", "异烟肼"],
    }
    results = [
        {
            "rank": 1,
            "score": 0.8,
            "chunk_id": "CNKI-1::SEC::001",
            "title": "三联药物治疗研究",
            "section": "治疗方法",
            "source_path": "data/articles/raw/chinese/cnki_pdf/a.pdf",
            "source_pages": [2],
            "text": "患者接受利福平联合治疗。",
        }
    ]

    evaluation = evaluate_retrieval_result(query, results)

    assert evaluation["passed"] is True
    assert evaluation["matched_keywords"] == ["利福平"]


def test_missing_page_traceability_fails() -> None:
    query = {"id": "test", "query": "治疗", "expected_any": ["治疗"]}
    results = [
        {
            "chunk_id": "CNKI-1::SEC::001",
            "title": "治疗研究",
            "source_path": "a.pdf",
            "source_pages": [],
            "text": "治疗有效。",
        }
    ]

    evaluation = evaluate_retrieval_result(query, results)

    assert evaluation["passed"] is False
    assert evaluation["missing_traceability"] == ["CNKI-1::SEC::001"]
