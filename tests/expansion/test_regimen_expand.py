"""
用途：独立验证联合用药分句识别、连接符重组和中立派生结果。
输入：测试内构造的原始问题与药物实体匹配结果。
输出：pytest 对 DerivedQuery 文本、类型和省略概念的断言结果。
不做什么：不构造 RetrievalQuery，不测试编号、激活条件或检索流程。
"""

import pytest

from rag_medical.common.expansion.aliases import AliasEntry
from rag_medical.common.expansion.entity_match import match_entities
from rag_medical.common.expansion.regimen_expand import (
    PARTIAL_COMBINATION,
    SINGLE_DRUG,
    DerivedQuery,
    RegimenExpansionResult,
    derive_regimen_queries,
)


def _make_aliases(*terms: str) -> dict[str, list[AliasEntry]]:
    return {
        f"DRUG_{term}": [
            AliasEntry(
                concept_id=f"DRUG_{term}",
                category="drug",
                language="en",
                term=term,
                term_kind="canonical",
                source="test",
            )
        ]
        for term in terms
    }


def _derive(
    text: str,
    *terms: str,
    min_drugs: int = 3,
    max_drugs: int = 4,
) -> RegimenExpansionResult:
    aliases = _make_aliases(*terms)
    entities = match_entities(text, aliases).entities
    return derive_regimen_queries(text, entities, min_drugs, max_drugs)


def _find_query(
    queries: tuple[DerivedQuery, ...],
    derivation_type: str,
    omitted_ids: tuple[str, ...],
) -> DerivedQuery:
    return next(
        query
        for query in queries
        if query.derivation_type == derivation_type
        and query.omitted_concept_ids == omitted_ids
    )


@pytest.mark.parametrize(
    ("text", "expected_text"),
    [
        ("A+B+C治疗X", "A+C治疗X"),
        ("A、B、C治疗X", "A、C治疗X"),
        ("A联合B联合C治疗X", "A联合C治疗X"),
        ("A plus B plus C治疗X", "A plus C治疗X"),
        ("A+B+C三药联合治疗X", "A+C联合治疗X"),
    ],
)
def test_partial_combination_preserves_connector(
    text: str,
    expected_text: str,
) -> None:
    result = _derive(text, "A", "B", "C")

    query = _find_query(
        result.queries,
        PARTIAL_COMBINATION,
        ("DRUG_B",),
    )
    assert query.query_text == expected_text


def test_single_drug_removes_joint_structure_and_count_cue() -> None:
    result = _derive("A+B+C三药联合治疗X", "A", "B", "C")

    query = _find_query(
        result.queries,
        SINGLE_DRUG,
        ("DRUG_B", "DRUG_C"),
    )
    assert query.query_text == "A治疗X"


def test_repeated_he_connector_is_joint_for_three_drugs() -> None:
    result = _derive("A和B和C", "A", "B", "C")

    query = _find_query(
        result.queries,
        PARTIAL_COMBINATION,
        ("DRUG_B",),
    )
    assert query.query_text == "A和C"


@pytest.mark.parametrize(
    "text",
    [
        "A与B哪个更安全？",
        "A与B的区别",
        "A或B治疗X",
        "A、B、C分别是三种药物",
        "先使用A，随后改用B",
        "A、B、C",
        "A、B、C分别治疗不同疾病",
    ],
)
def test_non_joint_expression_is_not_expanded(text: str) -> None:
    result = _derive(text, "A", "B", "C")

    assert result.queries == ()


def test_sentence_boundary_prevents_three_drug_scheme() -> None:
    result = _derive("先使用A。然后B联合C治疗X。", "A", "B", "C")

    assert result.queries
    assert all("DRUG_A" not in item.omitted_concept_ids for item in result.queries)
    assert all("A。" in item.query_text for item in result.queries)


def test_unknown_second_drug_does_not_form_known_scheme() -> None:
    result = _derive("A+司美格鲁肽治疗X", "A")

    assert result.queries == ()


def test_more_than_maximum_drugs_is_reported_without_queries() -> None:
    result = _derive("A+B+C+D+E治疗X", "A", "B", "C", "D", "E")

    assert result.queries == ()
    assert result.skipped_regimen_concept_ids == (
        ("DRUG_A", "DRUG_B", "DRUG_C", "DRUG_D", "DRUG_E"),
    )
