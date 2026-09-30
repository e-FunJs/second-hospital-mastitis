"""
用途：验证联合用药派生文本进入查询计划后的排序、字段和截断行为。
输入：测试内构造的 QueryRecord 与最小可信别名词典。
输出：pytest 对完整 RetrievalQuery 列表和超限日志的断言结果。
不做什么：不接入检索、翻译、文件输出或真实医学语料。
"""

import logging

from rag_medical.common.expansion.aliases import AliasEntry
from rag_medical.common.expansion.models import (
    QueryActivation,
    QueryGenerationKind,
    QueryRecord,
    RetrievalQuery,
)
from rag_medical.common.expansion.query_expand import build_query_plan


def _entry(
    concept_id: str,
    category: str,
    language: str,
    term: str,
    term_kind: str = "canonical",
) -> AliasEntry:
    return AliasEntry(
        concept_id=concept_id,
        category=category,
        language=language,
        term=term,
        term_kind=term_kind,
        source="test",
    )


def _canonical_aliases(*terms: str) -> dict[str, list[AliasEntry]]:
    aliases = {
        f"DRUG_{term}": [_entry(f"DRUG_{term}", "drug", "en", term)]
        for term in terms
    }
    aliases["DIS_X"] = [_entry("DIS_X", "disease", "en", "X")]
    return aliases


def _aliases_with_glm(*terms: str) -> dict[str, list[AliasEntry]]:
    aliases = _canonical_aliases(*terms)
    aliases.pop("DIS_X")
    aliases["DIS_GLM"] = [
        _entry("DIS_GLM", "disease", "zh", "肉芽肿性小叶性乳腺炎"),
        _entry("DIS_GLM", "disease", "en", "GLM", "abbreviation"),
    ]
    return aliases


def _build_plan(
    text: str,
    aliases: dict[str, list[AliasEntry]],
    max_queries: int = 20,
    query_id: str = "Q1",
) -> list[RetrievalQuery]:
    return build_query_plan(
        QueryRecord(query_id=query_id, query_text=text),
        aliases,
        max_queries,
        min_combination_drugs=3,
        max_combination_drugs=4,
    )


def test_three_drug_plan_has_alias_partials_and_singles() -> None:
    plan = _build_plan(
        "A+B+C治疗GLM的疗效如何？",
        _aliases_with_glm("A", "B", "C"),
    )

    assert [item.query_text for item in plan] == [
        "A+B+C治疗GLM的疗效如何？",
        "A+B+C治疗肉芽肿性小叶性乳腺炎的疗效如何？",
        "A+B治疗GLM的疗效如何？",
        "A+C治疗GLM的疗效如何？",
        "B+C治疗GLM的疗效如何？",
        "A治疗GLM的疗效如何？",
        "B治疗GLM的疗效如何？",
        "C治疗GLM的疗效如何？",
    ]
    assert [item.query_index for item in plan] == list(range(8))
    assert [item.generation_kind for item in plan] == [
        QueryGenerationKind.ORIGINAL,
        QueryGenerationKind.ALIAS_SUBSTITUTION,
        QueryGenerationKind.PARTIAL_COMBINATION,
        QueryGenerationKind.PARTIAL_COMBINATION,
        QueryGenerationKind.PARTIAL_COMBINATION,
        QueryGenerationKind.SINGLE_DRUG,
        QueryGenerationKind.SINGLE_DRUG,
        QueryGenerationKind.SINGLE_DRUG,
    ]
    expected_concepts = [
        ["DRUG_A", "DRUG_B", "DRUG_C", "DIS_GLM"],
        ["DRUG_A", "DRUG_B", "DRUG_C", "DIS_GLM"],
        ["DRUG_A", "DRUG_B", "DIS_GLM"],
        ["DRUG_A", "DRUG_C", "DIS_GLM"],
        ["DRUG_B", "DRUG_C", "DIS_GLM"],
        ["DRUG_A", "DIS_GLM"],
        ["DRUG_B", "DIS_GLM"],
        ["DRUG_C", "DIS_GLM"],
    ]
    assert [item.concept_ids for item in plan] == expected_concepts
    assert [item.omitted_concept_ids for item in plan] == [
        [],
        [],
        ["DRUG_C"],
        ["DRUG_B"],
        ["DRUG_A"],
        ["DRUG_B", "DRUG_C"],
        ["DRUG_A", "DRUG_C"],
        ["DRUG_A", "DRUG_B"],
    ]
    assert all(item.source_query_index == 0 for item in plan[1:])
    assert all(
        item.activation is QueryActivation.AFTER_FULL_SCHEME_GAP
        for item in plan[2:5]
    )
    assert all(
        item.activation is QueryActivation.AFTER_PARTIAL_SCHEME_GAP
        for item in plan[5:]
    )


def test_two_drug_plan_has_no_partial_combinations() -> None:
    plan = _build_plan("A+B治疗GLM的疗效如何？", _aliases_with_glm("A", "B"))

    assert [item.query_text for item in plan] == [
        "A+B治疗GLM的疗效如何？",
        "A+B治疗肉芽肿性小叶性乳腺炎的疗效如何？",
        "A治疗GLM的疗效如何？",
        "B治疗GLM的疗效如何？",
    ]
    assert all(
        item.generation_kind is not QueryGenerationKind.PARTIAL_COMBINATION
        for item in plan
    )
    assert all(
        item.activation is QueryActivation.AFTER_FULL_SCHEME_GAP
        for item in plan[2:]
    )


def test_comparison_expression_only_keeps_original_and_aliases() -> None:
    aliases = {
        "DRUG_A": [
            _entry("DRUG_A", "drug", "en", "alpha"),
            _entry("DRUG_A", "drug", "en", "A", "abbreviation"),
        ],
        "DRUG_B": [
            _entry("DRUG_B", "drug", "en", "beta"),
            _entry("DRUG_B", "drug", "en", "B", "abbreviation"),
        ],
    }
    plan = _build_plan("A与B哪个更安全？", aliases)

    assert [item.query_text for item in plan] == [
        "A与B哪个更安全？",
        "alpha与B哪个更安全？",
        "A与beta哪个更安全？",
    ]
    assert all(
        item.generation_kind
        in {QueryGenerationKind.ORIGINAL, QueryGenerationKind.ALIAS_SUBSTITUTION}
        for item in plan
    )


def test_unknown_drug_does_not_create_regimen_queries() -> None:
    aliases = {
        "DRUG_A": [
            _entry("DRUG_A", "drug", "en", "alpha"),
            _entry("DRUG_A", "drug", "en", "A", "abbreviation"),
        ],
        "DIS_X": [_entry("DIS_X", "disease", "en", "X")],
    }
    plan = _build_plan("A+司美格鲁肽治疗X", aliases)

    assert [item.query_text for item in plan] == [
        "A+司美格鲁肽治疗X",
        "alpha+司美格鲁肽治疗X",
    ]
    assert all(
        item.generation_kind
        in {QueryGenerationKind.ORIGINAL, QueryGenerationKind.ALIAS_SUBSTITUTION}
        for item in plan
    )


def test_max_queries_prioritizes_partials_before_singles() -> None:
    plan = _build_plan(
        "A+B+C治疗X",
        _canonical_aliases("A", "B", "C"),
        max_queries=5,
    )

    assert [item.query_text for item in plan] == [
        "A+B+C治疗X",
        "A+B治疗X",
        "A+C治疗X",
        "B+C治疗X",
        "A治疗X",
    ]
    assert [item.query_index for item in plan] == [0, 1, 2, 3, 4]


def test_removed_middle_drug_never_leaves_double_connector() -> None:
    plan = _build_plan("A+B+C治疗X", _canonical_aliases("A", "B", "C"))

    middle_removed = next(
        item for item in plan if item.omitted_concept_ids == ["DRUG_B"]
    )
    assert middle_removed.query_text == "A+C治疗X"
    assert "++" not in middle_removed.query_text


def test_over_limit_regimen_logs_without_question_text(caplog) -> None:
    caplog.set_level(logging.WARNING)
    plan = _build_plan(
        "A+B+C+D+E治疗X",
        _canonical_aliases("A", "B", "C", "D", "E"),
        query_id="Q-LIMIT",
    )

    assert [item.query_text for item in plan] == ["A+B+C+D+E治疗X"]
    assert "query_id=Q-LIMIT" in caplog.text
    assert "drug_count=5" in caplog.text
    assert "max_drugs=4" in caplog.text
    assert "治疗X" not in caplog.text
