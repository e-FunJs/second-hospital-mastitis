"""
用途：验证 R0 原查询和 canonical 别名替换查询的生成规则。
输入：测试内构造的 QueryRecord 与小型可信别名词典。
输出：pytest 对查询顺序、溯源字段、语言和数量上限的断言结果。
不做什么：不测试部分组合、单药外扩、检索、模型推理或文件写入。
"""

import pytest

from rag_medical.common.expansion.aliases import AliasEntry
from rag_medical.common.expansion.models import (
    QueryActivation,
    QueryGenerationKind,
    QueryRecord,
    RetrievalStage,
)
from rag_medical.common.expansion.query_expand import build_query_plan


def _alias(
    concept_id: str,
    category: str,
    language: str,
    term: str,
    term_kind: str,
) -> AliasEntry:
    return AliasEntry(
        concept_id=concept_id,
        category=category,
        language=language,
        term=term,
        term_kind=term_kind,
        source="trusted_manual",
    )


def _make_aliases() -> dict[str, list[AliasEntry]]:
    return {
        "DIS_GLM": [
            _alias("DIS_GLM", "disease", "zh", "肉芽肿性小叶性乳腺炎", "canonical"),
            _alias(
                "DIS_GLM",
                "disease",
                "en",
                "granulomatous lobular mastitis",
                "canonical",
            ),
            _alias("DIS_GLM", "disease", "en", "GLM", "abbreviation"),
        ],
        "DIS_PDM": [
            _alias("DIS_PDM", "disease", "zh", "导管周围乳腺炎", "canonical"),
            _alias("DIS_PDM", "disease", "en", "periductal mastitis", "canonical"),
            _alias("DIS_PDM", "disease", "en", "PDM", "abbreviation"),
        ],
        "DRUG_INH": [
            _alias("DRUG_INH", "drug", "zh", "异烟肼", "canonical"),
            _alias("DRUG_INH", "drug", "en", "isoniazid", "canonical"),
            _alias("DRUG_INH", "drug", "en", "INH", "abbreviation"),
        ],
        "DRUG_RFP": [
            _alias("DRUG_RFP", "drug", "zh", "利福平", "canonical"),
            _alias("DRUG_RFP", "drug", "en", "rifampicin", "canonical"),
            _alias("DRUG_RFP", "drug", "en", "RFP", "abbreviation"),
        ],
    }


def _record(text: str) -> QueryRecord:
    return QueryRecord(query_id="Q1", query_text=text)


def test_single_entity_generates_original_and_two_canonical_queries() -> None:
    plan = build_query_plan(
        _record("GLM有哪些临床表现？"),
        _make_aliases(),
        12,
        min_combination_drugs=3,
        max_combination_drugs=4,
    )

    assert [item.query_text for item in plan] == [
        "GLM有哪些临床表现？",
        "肉芽肿性小叶性乳腺炎有哪些临床表现？",
        "granulomatous lobular mastitis有哪些临床表现？",
    ]
    assert [item.query_index for item in plan] == [0, 1, 2]
    assert [item.query_language for item in plan] == ["mixed", "zh", "mixed"]
    assert plan[0].retrieval_stage is RetrievalStage.R0
    assert plan[0].generation_kind is QueryGenerationKind.ORIGINAL
    assert plan[0].source_query_index is None
    assert all(item.activation is QueryActivation.ALWAYS for item in plan)
    assert all(item.concept_ids == ["DIS_GLM"] for item in plan)
    assert all(item.omitted_concept_ids == [] for item in plan)
    assert all(
        item.generation_kind is QueryGenerationKind.ALIAS_SUBSTITUTION
        and item.source_query_index == 0
        and item.retrieval_stage is RetrievalStage.R2
        for item in plan[1:]
    )


def test_existing_chinese_canonical_only_generates_english_query() -> None:
    plan = build_query_plan(
        _record("肉芽肿性小叶性乳腺炎有哪些表现？"),
        _make_aliases(),
        12,
        min_combination_drugs=3,
        max_combination_drugs=4,
    )

    assert [item.query_text for item in plan] == [
        "肉芽肿性小叶性乳腺炎有哪些表现？",
        "granulomatous lobular mastitis有哪些表现？",
    ]
    assert [item.query_language for item in plan] == ["zh", "mixed"]


def test_multiple_entities_expand_one_concept_at_a_time() -> None:
    plan = build_query_plan(
        _record("GLM合并PDM如何治疗？"),
        _make_aliases(),
        12,
        min_combination_drugs=3,
        max_combination_drugs=4,
    )

    assert [item.query_text for item in plan] == [
        "GLM合并PDM如何治疗？",
        "肉芽肿性小叶性乳腺炎合并PDM如何治疗？",
        "GLM合并导管周围乳腺炎如何治疗？",
        "granulomatous lobular mastitis合并PDM如何治疗？",
        "GLM合并periductal mastitis如何治疗？",
    ]
    assert [item.query_index for item in plan] == [0, 1, 2, 3, 4]
    assert all(item.concept_ids == ["DIS_GLM", "DIS_PDM"] for item in plan)


def test_text_without_dictionary_entity_only_keeps_original() -> None:
    plan = build_query_plan(
        _record("用抗生素治疗乳腺炎有效吗？"),
        _make_aliases(),
        12,
        min_combination_drugs=3,
        max_combination_drugs=4,
    )

    assert len(plan) == 1
    assert plan[0].query_language == "zh"
    assert plan[0].concept_ids == []


def test_existing_english_canonical_skips_duplicate_target_language() -> None:
    plan = build_query_plan(
        _record("GLM（granulomatous lobular mastitis）如何治疗？"),
        _make_aliases(),
        12,
        min_combination_drugs=3,
        max_combination_drugs=4,
    )

    assert [item.query_text for item in plan] == [
        "GLM（granulomatous lobular mastitis）如何治疗？",
        "肉芽肿性小叶性乳腺炎（granulomatous lobular mastitis）如何治疗？",
    ]
    assert all(item.concept_ids == ["DIS_GLM"] for item in plan)


def test_max_queries_keeps_r0_and_truncates_in_stable_order() -> None:
    plan = build_query_plan(
        _record("GLM合并PDM，并使用INH与RFP如何治疗？"),
        _make_aliases(),
        3,
        min_combination_drugs=3,
        max_combination_drugs=4,
    )

    assert [item.query_text for item in plan] == [
        "GLM合并PDM，并使用INH与RFP如何治疗？",
        "肉芽肿性小叶性乳腺炎合并PDM，并使用INH与RFP如何治疗？",
        "GLM合并导管周围乳腺炎，并使用INH与RFP如何治疗？",
    ]
    expected_concepts = ["DIS_GLM", "DIS_PDM", "DRUG_INH", "DRUG_RFP"]
    assert all(item.concept_ids == expected_concepts for item in plan)


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        ("ＡＢＣ治疗方案", "mixed"),
        ("ＡＢＣ", "en"),
        ("患者 5 例如何治疗？", "zh"),
        ("GLM患者 5 例", "mixed"),
        ("α、β", "unknown"),
        ("α受体表达如何？", "zh"),
        ("β-blocker", "en"),
        ("12345", "unknown"),
        ("㐀", "zh"),
        ("𠀀", "unknown"),
    ],
)
def test_r0_language_detection(text: str, expected_language: str) -> None:
    plan = build_query_plan(
        _record(text),
        _make_aliases(),
        1,
        min_combination_drugs=3,
        max_combination_drugs=4,
    )

    assert len(plan) == 1
    assert plan[0].query_text == text
    assert plan[0].query_language == expected_language


def test_max_queries_must_leave_room_for_r0() -> None:
    with pytest.raises(ValueError, match="max_queries"):
        build_query_plan(
            _record("GLM有哪些表现？"),
            _make_aliases(),
            0,
            min_combination_drugs=3,
            max_combination_drugs=4,
        )
