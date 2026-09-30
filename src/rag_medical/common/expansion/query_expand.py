"""
用途：编排原查询、canonical 别名、部分用药组合和单药检索查询。
输入：QueryRecord、可信别名词典，以及查询数和联合用药数量限制。
输出：按激活优先级排序、带概念溯源字段的 RetrievalQuery 列表。
不做什么：不执行检索、翻译、否定或角色判断，也不读写查询计划文件。
语言判断仅检查 NFKC 后的影子文本。汉字范围为 CJK 基本区
（U+4E00-U+9FFF）和扩展 A 区（U+3400-U+4DBF），不含扩展 B+。
中文医学文献中扩展 B+ 几乎不出现，遗漏只影响语言判断精度，
不产生错误结论。Latin 字母为 [A-Za-z]；其他字符均为中性。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterator

from rag_medical.common.expansion.aliases import AliasEntry
from rag_medical.common.expansion.entity_match import MatchedEntity, match_entities
from rag_medical.common.expansion.models import (
    QueryActivation,
    QueryGenerationKind,
    QueryRecord,
    RetrievalQuery,
    RetrievalStage,
)
from rag_medical.common.expansion.regimen_expand import (
    PARTIAL_COMBINATION,
    SINGLE_DRUG,
    DerivedQuery,
    derive_regimen_queries,
)


_HAN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")
logger = logging.getLogger(__name__)


def _detect_query_language(text: str) -> str:
    """按限定的汉字和 Latin 字母范围判定查询语言形态。"""
    normalized_text = unicodedata.normalize("NFKC", text)
    has_han = _HAN_PATTERN.search(normalized_text) is not None
    has_latin = _LATIN_PATTERN.search(normalized_text) is not None
    if has_han and has_latin:
        return "mixed"
    if has_han:
        return "zh"
    if has_latin:
        return "en"
    return "unknown"


def _normalize_for_comparison(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized_text.split())


def _group_entities_by_concept(
    entities: list[MatchedEntity],
) -> dict[str, list[MatchedEntity]]:
    grouped_entities: dict[str, list[MatchedEntity]] = {}
    for entity in sorted(entities, key=lambda item: (item.start, item.end)):
        grouped_entities.setdefault(entity.concept_id, []).append(entity)
    return grouped_entities


def _collect_canonical_terms(
    concept_ids: list[str],
    aliases: dict[str, list[AliasEntry]],
) -> dict[str, dict[str, str]]:
    """为每个已识别概念选取词典中首个中英文 canonical。"""
    canonical_terms: dict[str, dict[str, str]] = {}
    for concept_id in concept_ids:
        terms_by_language: dict[str, str] = {}
        for entry in aliases.get(concept_id, []):
            if entry.term_kind == "canonical":
                terms_by_language.setdefault(entry.language, entry.term)
        canonical_terms[concept_id] = terms_by_language
    return canonical_terms


def _canonical_is_present(
    entities: list[MatchedEntity],
    canonical_term: str,
) -> bool:
    """确认同概念的某次有效命中已经使用目标 canonical。"""
    target = _normalize_for_comparison(canonical_term)
    return any(
        _normalize_for_comparison(entity.matched_term) == target
        for entity in entities
    )


def _make_original_query(
    query_record: QueryRecord,
    query_language: str,
    concept_ids: list[str],
) -> RetrievalQuery:
    """构造始终保留的 R0 原查询。"""
    return RetrievalQuery(
        retrieval_stage=RetrievalStage.R0,
        query_index=0,
        query_text=query_record.query_text,
        query_language=query_language,
        generation_kind=QueryGenerationKind.ORIGINAL,
        source_query_index=None,
        concept_ids=list(concept_ids),
        omitted_concept_ids=[],
        activation=QueryActivation.ALWAYS,
    )


def _make_derived_query(
    query_index: int,
    query_text: str,
    query_language: str,
    generation_kind: QueryGenerationKind,
    activation: QueryActivation,
    concept_ids: list[str],
    omitted_concept_ids: list[str],
) -> RetrievalQuery:
    """在编排层统一填充所有 R0 派生查询的数据契约。"""
    return RetrievalQuery(
        retrieval_stage=RetrievalStage.R2,
        query_index=query_index,
        query_text=query_text,
        query_language=query_language,
        generation_kind=generation_kind,
        source_query_index=0,
        concept_ids=list(concept_ids),
        omitted_concept_ids=list(omitted_concept_ids),
        activation=activation,
    )


def _iter_alias_queries(
    query_text: str,
    original_language: str,
    concept_ids: list[str],
    entities_by_concept: dict[str, list[MatchedEntity]],
    canonical_terms: dict[str, dict[str, str]],
) -> Iterator[RetrievalQuery]:
    """依确定顺序产生不重复的 canonical 替换查询。"""
    seen_query_texts = {query_text}
    query_index = 1
    target_languages = ("en", "zh") if original_language == "en" else ("zh", "en")
    for target_language in target_languages:
        for concept_id in concept_ids:
            canonical_term = canonical_terms[concept_id].get(target_language)
            if canonical_term is None:
                continue
            if _canonical_is_present(entities_by_concept[concept_id], canonical_term):
                continue
            entity = entities_by_concept[concept_id][0]
            variant_text = query_text[: entity.start] + canonical_term + query_text[entity.end :]
            variant_language = _detect_query_language(variant_text)
            if variant_language == "unknown" or variant_text in seen_query_texts:
                continue
            seen_query_texts.add(variant_text)
            yield _make_derived_query(
                query_index,
                variant_text,
                variant_language,
                QueryGenerationKind.ALIAS_SUBSTITUTION,
                QueryActivation.ALWAYS,
                concept_ids,
                [],
            )
            query_index += 1


def _resolve_regimen_metadata(
    derived_query: DerivedQuery,
    min_combination_drugs: int,
) -> tuple[QueryGenerationKind, QueryActivation]:
    """把中立文本派生类型映射为唯一的查询编排字段。"""
    if derived_query.derivation_type == PARTIAL_COMBINATION:
        return (
            QueryGenerationKind.PARTIAL_COMBINATION,
            QueryActivation.AFTER_FULL_SCHEME_GAP,
        )
    if derived_query.derivation_type != SINGLE_DRUG:
        raise ValueError(f"未知的方案派生类型：{derived_query.derivation_type!r}")
    original_drug_count = len(derived_query.omitted_concept_ids) + 1
    activation = (
        QueryActivation.AFTER_PARTIAL_SCHEME_GAP
        if original_drug_count >= min_combination_drugs
        else QueryActivation.AFTER_FULL_SCHEME_GAP
    )
    return QueryGenerationKind.SINGLE_DRUG, activation


def _append_regimen_queries(
    query_plan: list[RetrievalQuery],
    derived_queries: tuple[DerivedQuery, ...],
    concept_ids: list[str],
    min_combination_drugs: int,
    max_queries: int,
) -> bool:
    """去重并追加方案派生查询；达到查询上限时返回 True。"""
    seen_query_texts = {item.query_text for item in query_plan}
    for derived_query in derived_queries:
        query_language = _detect_query_language(derived_query.query_text)
        if query_language == "unknown" or derived_query.query_text in seen_query_texts:
            continue
        generation_kind, activation = _resolve_regimen_metadata(
            derived_query, min_combination_drugs
        )
        omitted_ids = list(derived_query.omitted_concept_ids)
        retained_ids = [item for item in concept_ids if item not in omitted_ids]
        query_plan.append(
            _make_derived_query(
                len(query_plan),
                derived_query.query_text,
                query_language,
                generation_kind,
                activation,
                retained_ids,
                omitted_ids,
            )
        )
        seen_query_texts.add(derived_query.query_text)
        if len(query_plan) >= max_queries:
            return True
    return False


def _log_skipped_regimens(
    query_id: str,
    skipped_regimens: tuple[tuple[str, ...], ...],
    max_combination_drugs: int,
) -> None:
    """记录超限概念而不把可能含患者信息的完整问题写入日志。"""
    for drug_ids in skipped_regimens:
        logger.warning(
            "跳过联合用药外扩：query_id=%s drug_count=%d max_drugs=%d "
            "drug_concept_ids=%s",
            query_id,
            len(drug_ids),
            max_combination_drugs,
            list(drug_ids),
        )


def build_query_plan(
    query_record: QueryRecord,
    aliases: dict[str, list[AliasEntry]],
    max_queries: int,
    *,
    min_combination_drugs: int,
    max_combination_drugs: int,
) -> list[RetrievalQuery]:
    """按固定优先级生成原查询、别名、部分组合和单药查询。"""
    if max_queries < 1:
        raise ValueError("max_queries 必须至少为 1，以确保保留 R0 原查询")
    if min_combination_drugs < 3:
        raise ValueError("min_combination_drugs 必须至少为 3")
    if max_combination_drugs < min_combination_drugs:
        raise ValueError("max_combination_drugs 不能小于 min_combination_drugs")

    match_result = match_entities(query_record.query_text, aliases)
    entities_by_concept = _group_entities_by_concept(match_result.entities)
    concept_ids = list(entities_by_concept)
    original_language = _detect_query_language(query_record.query_text)
    query_plan = [_make_original_query(query_record, original_language, concept_ids)]
    if len(query_plan) >= max_queries:
        return query_plan
    canonical_terms = _collect_canonical_terms(concept_ids, aliases)

    for alias_query in _iter_alias_queries(
        query_record.query_text,
        original_language,
        concept_ids,
        entities_by_concept,
        canonical_terms,
    ):
        query_plan.append(alias_query)
        if len(query_plan) >= max_queries:
            return query_plan
    regimen_result = derive_regimen_queries(
        query_record.query_text,
        match_result.entities,
        min_combination_drugs,
        max_combination_drugs,
    )
    _log_skipped_regimens(
        query_record.query_id,
        regimen_result.skipped_regimen_concept_ids,
        max_combination_drugs,
    )
    _append_regimen_queries(
        query_plan,
        regimen_result.queries,
        concept_ids,
        min_combination_drugs,
        max_queries,
    )
    return query_plan
