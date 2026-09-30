"""
用途：验证证据外扩基础模型的数据契约和 JSON 兼容性。
输入：测试内构造的原始问题、检索查询和证据样例。
输出：pytest 断言结果。
不做什么：不调用检索模型、生成模型、FAISS 或真实业务数据。
"""

import json
from enum import Enum

import pytest

from rag_medical.common.expansion.models import (
    AssertionRole,
    EvidenceItem,
    EvidenceState,
    QueryActivation,
    QueryGenerationKind,
    QueryRecord,
    RetrievalQuery,
    RetrievalStage,
    StopReason,
)


def make_evidence_item() -> EvidenceItem:
    """构造包含中英文和全角字符的最小完整证据。"""
    return EvidenceItem(
        evidence_id="E1",
        retrieval_stage=RetrievalStage.R0,
        retrieval_query="异烟肼治疗非哺乳期乳腺炎",
        query_index=0,
        rank=1,
        chunk_id="CNKI-001:2:3",
        matched_text="异烟肼＋rifampicin 治疗非哺乳期乳腺炎。",
        text="证据上下文：剂量为０．３ g，response was recorded.",
        score=0.91,
        language="zh",
        context_chunk_ids=["CNKI-001:2:2", "CNKI-001:2:3"],
    )


def test_query_record_round_trip_preserves_multilingual_text() -> None:
    record = QueryRecord(query_id="Q1", query_text="GLM 的表现与 treatment response？")

    assert QueryRecord.from_dict(record.to_dict()) == record


def test_retrieval_query_round_trip_preserves_stage() -> None:
    query = RetrievalQuery(
        retrieval_stage=RetrievalStage.R2,
        query_index=2,
        query_text="非哺乳期乳腺炎＋non-puerperal mastitis",
        query_language="mixed",
        generation_kind=QueryGenerationKind.ALIAS_SUBSTITUTION,
        source_query_index=0,
        concept_ids=["DIS_NPM"],
        omitted_concept_ids=[],
        activation=QueryActivation.ALWAYS,
    )

    assert RetrievalQuery.from_dict(query.to_dict()) == query


def test_evidence_item_requires_evidence_id() -> None:
    payload = make_evidence_item().to_dict()
    del payload["evidence_id"]

    with pytest.raises(TypeError):
        EvidenceItem.from_dict(payload)


def test_evidence_item_requires_retrieval_query() -> None:
    payload = make_evidence_item().to_dict()
    del payload["retrieval_query"]

    with pytest.raises(TypeError):
        EvidenceItem.from_dict(payload)


def test_unknown_retrieval_stage_is_rejected() -> None:
    payload = make_evidence_item().to_dict()
    payload["retrieval_stage"] = "R9"

    with pytest.raises(ValueError):
        EvidenceItem.from_dict(payload)


def test_evidence_item_round_trip_preserves_unicode() -> None:
    evidence = make_evidence_item()

    assert EvidenceItem.from_dict(evidence.to_dict()) == evidence


def test_evidence_item_to_dict_is_json_serializable() -> None:
    serialized = json.dumps(make_evidence_item().to_dict(), ensure_ascii=False)

    assert "异烟肼＋rifampicin" in serialized
    assert "０．３ g" in serialized


@pytest.mark.parametrize(
    ("enum_type", "value"),
    [
        (RetrievalStage, "R1"),
        (EvidenceState, "DIRECT_CANDIDATE"),
        (AssertionRole, "NEGATED_OR_EXCLUDED"),
        (StopReason, "STOP_SEARCH_SCOPE"),
    ],
)
def test_enum_values_convert_from_strings(enum_type: type[Enum], value: str) -> None:
    assert enum_type(value).value == value
