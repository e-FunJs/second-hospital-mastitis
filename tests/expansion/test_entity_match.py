"""
用途：验证实体精确匹配、重叠消解、歧义隔离和原文坐标映射。
输入：测试内构造的中英文文本和 AliasEntry 词典。
输出：pytest 断言结果。
不做什么：不测试查询生成、检索、否定判断或模型推理。
"""

from pathlib import Path

import pytest

from rag_medical.common.expansion.aliases import AliasEntry, load_aliases
from rag_medical.common.expansion.entity_match import match_entities


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ALIASES_PATH = PROJECT_ROOT / "resources/evidence_expansion/query_aliases.jsonl"


def make_alias(
    term: str,
    *,
    concept_id: str = "DIS_GLM",
    category: str = "disease",
    language: str = "zh",
    term_kind: str = "canonical",
) -> AliasEntry:
    """构造一条仅供匹配测试使用的词典条目。"""
    return AliasEntry(
        concept_id=concept_id,
        category=category,
        language=language,
        term=term,
        term_kind=term_kind,
        source="test_fixture",
    )


def group_aliases(*entries: AliasEntry) -> dict[str, list[AliasEntry]]:
    """将测试条目组装成 load_aliases() 的返回结构。"""
    grouped: dict[str, list[AliasEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.concept_id, []).append(entry)
    return grouped


def glm_aliases() -> dict[str, list[AliasEntry]]:
    """返回正式词典，用于验证真实 GLM 条目。"""
    return load_aliases(FORMAL_ALIASES_PATH)


def test_matches_glm_inside_chinese_question() -> None:
    result = match_entities("GLM有哪些临床表现", glm_aliases())

    assert [(entity.concept_id, entity.matched_term) for entity in result.entities] == [
        ("DIS_GLM", "GLM")
    ]


@pytest.mark.parametrize("spelling", ["GLM", "glm", "Glm"])
def test_english_matching_is_case_insensitive(spelling: str) -> None:
    result = match_entities(spelling, glm_aliases())

    assert len(result.entities) == 1
    assert result.entities[0].concept_id == "DIS_GLM"
    assert result.entities[0].matched_term == spelling


@pytest.mark.parametrize(
    ("text", "should_match"),
    [
        ("GLMX", False),
        ("XGLM", False),
        ("GLM2", False),
        ("GLM患者", True),
        ("GLM-related", True),
    ],
)
def test_english_abbreviation_uses_explicit_boundaries(
    text: str,
    should_match: bool,
) -> None:
    result = match_entities(text, glm_aliases())

    assert bool(result.entities) is should_match
    if should_match:
        assert (result.entities[0].start, result.entities[0].end) == (0, 3)


def test_longest_chinese_term_wins_for_same_concept_overlap() -> None:
    long_term = "肉芽肿性小叶性乳腺炎"
    aliases = group_aliases(
        make_alias(long_term),
        make_alias("小叶性乳腺炎", term_kind="synonym"),
    )

    result = match_entities(long_term, aliases)

    assert len(result.entities) == 1
    assert result.entities[0].matched_alias == long_term
    assert result.entities[0].matched_term == long_term


def test_same_concept_overlapping_english_terms_are_merged() -> None:
    text = "granulomatous lobular mastitis"
    aliases = group_aliases(
        make_alias(text, language="en"),
        make_alias("lobular mastitis", language="en", term_kind="synonym"),
    )

    result = match_entities(text, aliases)

    assert len(result.entities) == 1
    assert result.entities[0].matched_alias == text


def test_different_concepts_with_overlapping_terms_are_ambiguous() -> None:
    text = "肉芽肿性小叶性乳腺炎"
    aliases = group_aliases(
        make_alias(text, concept_id="DIS_GLM"),
        make_alias("小叶性乳腺炎", concept_id="DIS_OTHER"),
    )

    result = match_entities(text, aliases)

    assert result.entities == []
    assert result.ambiguous_spans == [(0, len(text), ["DIS_GLM", "DIS_OTHER"])]


def test_fullwidth_abbreviation_matches_original_span() -> None:
    text = "ＧＬＭ患者"

    result = match_entities(text, glm_aliases())

    assert len(result.entities) == 1
    assert result.entities[0].matched_term == "ＧＬＭ"
    assert (result.entities[0].start, result.entities[0].end) == (0, 3)
    assert result.entities[0].position_unreliable is False


def test_abbreviation_and_full_name_are_kept_as_separate_occurrences() -> None:
    text = "GLM and granulomatous lobular mastitis"

    result = match_entities(text, glm_aliases())

    assert [entity.concept_id for entity in result.entities] == ["DIS_GLM", "DIS_GLM"]
    assert [entity.matched_term for entity in result.entities] == [
        "GLM",
        "granulomatous lobular mastitis",
    ]
    assert [(entity.start, entity.end) for entity in result.entities] == [(0, 3), (8, 38)]


def test_fullwidth_position_mapping_uses_original_indices() -> None:
    text = "诊断：ＧＬＭ，建议复诊"
    expected_start = text.index("Ｇ")

    result = match_entities(text, glm_aliases())

    entity = result.entities[0]
    assert (entity.start, entity.end) == (expected_start, expected_start + 3)
    assert text[entity.start:entity.end] == "ＧＬＭ"


def test_nfkc_ligature_expansion_maps_to_one_original_character() -> None:
    aliases = group_aliases(make_alias("ffi", language="en"))
    text = "marker ﬃ result"
    expected_start = text.index("ﬃ")

    result = match_entities(text, aliases)

    entity = result.entities[0]
    assert (entity.start, entity.end) == (expected_start, expected_start + 1)
    assert entity.matched_term == "ﬃ"
    assert entity.position_unreliable is False


def test_collapsed_whitespace_maps_back_to_complete_original_phrase() -> None:
    alias = make_alias("granulomatous lobular mastitis", language="en")
    aliases = group_aliases(alias)
    text = "Result: granulomatous   lobular\tmastitis confirmed."
    expected_start = text.index("granulomatous")
    expected_end = text.index(" confirmed")

    result = match_entities(text, aliases)

    entity = result.entities[0]
    assert (entity.start, entity.end) == (expected_start, expected_end)
    assert entity.matched_term == "granulomatous   lobular\tmastitis"
    assert entity.position_unreliable is False
