"""第一层 RAG 问答脚本测试。

这里不调用 LLM，只验证 step08 输出可追溯的 evidence package，且不提前生成 prompt。
"""

from __future__ import annotations

import json

from rag_medical.common.step08_rag_answer import make_evidence_records, write_rag_package


def sample_search_results() -> list[dict]:
    return [
        {
            "rank": 1,
            "score": 0.81,
            "chunk_id": "PMC1::Treatment::001",
            "source_type": "pmc_full_text",
            "pmcid": "PMC1",
            "pmid": "123",
            "doi": "10.1/example",
            "title": "Treatment study",
            "journal": "Journal",
            "year": "2025",
            "section": "Treatment > Corticosteroids",
            "text": "Steroids were associated with lesion reduction.",
        },
        {
            "rank": 2,
            "score": 0.75,
            "chunk_id": "PMC2::Ultrasound::001",
            "source_type": "pubmed_abstract",
            "pmcid": "PMC2",
            "pmid": "456",
            "doi": "10.2/example",
            "title": "Ultrasound study",
            "journal": "Imaging Journal",
            "year": "2026",
            "section": "Diagnosis > Ultrasonography",
            "text": "Ultrasound commonly showed irregular hypoechoic lesions.",
        },
    ]


def test_make_evidence_records_adds_stable_evidence_ids() -> None:
    evidence = make_evidence_records(sample_search_results())

    assert evidence[0]["evidence_id"] == "E1"
    assert evidence[1]["evidence_id"] == "E2"
    assert evidence[0]["chunk_id"] == "PMC1::Treatment::001"
    assert evidence[0]["source_type"] == "pmc_full_text"
    assert evidence[1]["source_type"] == "pubmed_abstract"
    assert evidence[1]["citation"] == "PMC2 | 2026 | Ultrasound study | Diagnosis > Ultrasonography"


def test_write_rag_package_writes_only_evidence_json(tmp_path) -> None:
    question = "How are corticosteroids discussed?"
    evidence = make_evidence_records(sample_search_results())

    outputs = write_rag_package(
        output_dir=tmp_path,
        question=question,
        evidence_records=evidence,
        query_slug="corticosteroids",
    )

    evidence_json = json.loads(outputs["evidence_path"].read_text())

    assert evidence_json["question"] == question
    assert evidence_json["evidence"][0]["evidence_id"] == "E1"
    assert outputs["evidence_path"].name == "corticosteroids_evidence.json"
    assert set(outputs) == {"evidence_path"}
    assert not list(tmp_path.glob("*_prompt.txt"))
