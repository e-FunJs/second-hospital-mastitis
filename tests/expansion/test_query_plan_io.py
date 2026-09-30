"""
用途：验证查询计划 JSON 的结构、字符编码、文件命名和覆盖写入规则。
输入：测试内构造的 QueryRecord、RetrievalQuery 列表和 pytest 临时目录。
输出：pytest 对序列化往返、固定 slug 契约及目录错误的断言结果。
不做什么：不调用查询生成、检索、BGE、FAISS、Qwen 或 JSON Schema。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_medical.common.expansion.models import (
    QueryActivation,
    QueryGenerationKind,
    QueryRecord,
    RetrievalQuery,
    RetrievalStage,
)
from rag_medical.common.expansion.query_plan_io import write_query_plan


def _query(
    *,
    query_index: int,
    query_text: str,
    generation_kind: QueryGenerationKind,
    activation: QueryActivation,
    concept_ids: list[str],
    omitted_concept_ids: list[str],
    source_query_index: int | None,
    query_language: str = "mixed",
) -> RetrievalQuery:
    return RetrievalQuery(
        retrieval_stage=(
            RetrievalStage.R0
            if generation_kind is QueryGenerationKind.ORIGINAL
            else RetrievalStage.R2
        ),
        query_index=query_index,
        query_text=query_text,
        query_language=query_language,
        generation_kind=generation_kind,
        source_query_index=source_query_index,
        concept_ids=concept_ids,
        omitted_concept_ids=omitted_concept_ids,
        activation=activation,
    )


def _three_drug_queries() -> list[RetrievalQuery]:
    return [
        _query(
            query_index=0,
            query_text="异烟肼+利福平+乙胺丁醇治疗GLM的疗效如何？",
            generation_kind=QueryGenerationKind.ORIGINAL,
            activation=QueryActivation.ALWAYS,
            concept_ids=["DRUG_INH", "DRUG_RFP", "DRUG_EMB", "DIS_GLM"],
            omitted_concept_ids=[],
            source_query_index=None,
        ),
        _query(
            query_index=1,
            query_text="异烟肼+利福平治疗GLM的疗效如何？",
            generation_kind=QueryGenerationKind.PARTIAL_COMBINATION,
            activation=QueryActivation.AFTER_FULL_SCHEME_GAP,
            concept_ids=["DRUG_INH", "DRUG_RFP", "DIS_GLM"],
            omitted_concept_ids=["DRUG_EMB"],
            source_query_index=0,
        ),
        _query(
            query_index=2,
            query_text="异烟肼+乙胺丁醇治疗GLM的疗效如何？",
            generation_kind=QueryGenerationKind.PARTIAL_COMBINATION,
            activation=QueryActivation.AFTER_FULL_SCHEME_GAP,
            concept_ids=["DRUG_INH", "DRUG_EMB", "DIS_GLM"],
            omitted_concept_ids=["DRUG_RFP"],
            source_query_index=0,
        ),
        _query(
            query_index=3,
            query_text="利福平+乙胺丁醇治疗GLM的疗效如何？",
            generation_kind=QueryGenerationKind.PARTIAL_COMBINATION,
            activation=QueryActivation.AFTER_FULL_SCHEME_GAP,
            concept_ids=["DRUG_RFP", "DRUG_EMB", "DIS_GLM"],
            omitted_concept_ids=["DRUG_INH"],
            source_query_index=0,
        ),
        _query(
            query_index=4,
            query_text="异烟肼治疗GLM的疗效如何？",
            generation_kind=QueryGenerationKind.SINGLE_DRUG,
            activation=QueryActivation.AFTER_PARTIAL_SCHEME_GAP,
            concept_ids=["DRUG_INH", "DIS_GLM"],
            omitted_concept_ids=["DRUG_RFP", "DRUG_EMB"],
            source_query_index=0,
        ),
        _query(
            query_index=5,
            query_text="利福平治疗GLM的疗效如何？",
            generation_kind=QueryGenerationKind.SINGLE_DRUG,
            activation=QueryActivation.AFTER_PARTIAL_SCHEME_GAP,
            concept_ids=["DRUG_RFP", "DIS_GLM"],
            omitted_concept_ids=["DRUG_INH", "DRUG_EMB"],
            source_query_index=0,
        ),
        _query(
            query_index=6,
            query_text="乙胺丁醇治疗GLM的疗效如何？",
            generation_kind=QueryGenerationKind.SINGLE_DRUG,
            activation=QueryActivation.AFTER_PARTIAL_SCHEME_GAP,
            concept_ids=["DRUG_EMB", "DIS_GLM"],
            omitted_concept_ids=["DRUG_INH", "DRUG_RFP"],
            source_query_index=0,
        ),
    ]


def test_query_plan_serialization_round_trip(tmp_path: Path) -> None:
    record = QueryRecord(
        query_id="Q-001",
        query_text="异烟肼+利福平+乙胺丁醇治疗GLM的疗效如何？",
    )
    queries = _three_drug_queries()

    output_path = write_query_plan(record, queries, tmp_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    restored_record = QueryRecord.from_dict(payload["query_record"])
    restored_queries = [
        RetrievalQuery.from_dict(item) for item in payload["retrieval_queries"]
    ]

    assert payload["schema_version"] == "1.0"
    assert restored_record == record
    assert restored_queries == queries


def test_written_content_is_valid_json_with_enum_values(tmp_path: Path) -> None:
    record = QueryRecord(query_id="Q-ENUM", query_text="GLM如何治疗？")
    query = _query(
        query_index=0,
        query_text=record.query_text,
        generation_kind=QueryGenerationKind.ORIGINAL,
        activation=QueryActivation.ALWAYS,
        concept_ids=["DIS_GLM"],
        omitted_concept_ids=[],
        source_query_index=None,
    )

    output_path = write_query_plan(record, [query], tmp_path)
    item = json.loads(output_path.read_text(encoding="utf-8"))["retrieval_queries"][0]

    assert item["retrieval_stage"] == "R0"
    assert item["generation_kind"] == "ORIGINAL"
    assert item["activation"] == "always"
    assert item["source_query_index"] is None


def test_unicode_text_is_written_without_ascii_escaping(tmp_path: Path) -> None:
    text = "ＧＬＭ与granulomatous lobular mastitis如何治疗？"
    record = QueryRecord(query_id="Q-UNICODE", query_text=text)
    query = _query(
        query_index=0,
        query_text=text,
        generation_kind=QueryGenerationKind.ORIGINAL,
        activation=QueryActivation.ALWAYS,
        concept_ids=["DIS_GLM"],
        omitted_concept_ids=[],
        source_query_index=None,
    )

    output_path = write_query_plan(record, [query], tmp_path)
    raw_content = output_path.read_text(encoding="utf-8")

    assert text in raw_content
    assert "\\u" not in raw_content


@pytest.mark.parametrize(
    ("query_text", "expected_name"),
    [
        ("GLM有哪些临床表现？", "GLM有哪些临床表现_query_plan.json"),
        ("A+B+C treatment?", "A_B_C_treatment_query_plan.json"),
        ("？！", "rag_query_query_plan.json"),
    ],
)
def test_query_slug_contract_is_stable(
    tmp_path: Path,
    query_text: str,
    expected_name: str,
) -> None:
    record = QueryRecord(query_id="Q-SLUG", query_text=query_text)

    first_path = write_query_plan(record, [], tmp_path)
    second_path = write_query_plan(record, [], tmp_path)

    assert first_path.name == expected_name
    assert second_path == first_path


def test_existing_query_plan_is_overwritten(tmp_path: Path) -> None:
    query_text = "GLM有哪些临床表现？"
    first_record = QueryRecord(query_id="Q-OLD", query_text=query_text)
    second_record = QueryRecord(query_id="Q-NEW", query_text=query_text)

    output_path = write_query_plan(first_record, [], tmp_path)
    write_query_plan(second_record, _three_drug_queries(), tmp_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["query_record"]["query_id"] == "Q-NEW"
    assert len(payload["retrieval_queries"]) == 7


def test_missing_output_directory_raises_without_creating_it(tmp_path: Path) -> None:
    missing_directory = tmp_path / "missing"
    record = QueryRecord(query_id="Q-MISSING", query_text="GLM如何治疗？")

    with pytest.raises(FileNotFoundError, match="输出目录不存在"):
        write_query_plan(record, [], missing_directory)

    assert not missing_directory.exists()


def test_output_path_that_is_not_directory_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("occupied", encoding="utf-8")
    record = QueryRecord(query_id="Q-FILE", query_text="GLM如何治疗？")

    with pytest.raises(NotADirectoryError, match="输出路径不是目录"):
        write_query_plan(record, [], file_path)
