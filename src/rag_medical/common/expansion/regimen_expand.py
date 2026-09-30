"""
用途：识别问题中的显式联合用药方案，并生成部分组合和单药文本。
输入：原始问题、实体精确匹配结果，以及联合用药数量上下限。
输出：中立的 DerivedQuery 文本列表和因药物数超限而跳过的方案。
不做什么：不构造 RetrievalQuery，不分配索引、激活状态或检索阶段，
也不执行检索、翻译、日志和文件读写。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

from rag_medical.common.expansion.entity_match import MatchedEntity


PARTIAL_COMBINATION = "partial_combination"
SINGLE_DRUG = "single_drug"

_SENTENCE_PATTERN = re.compile(r"[^。！？；.!?;]+")
_BLOCKER_PATTERN = re.compile(
    r"哪个|比较|区别|差异|或|或者|分别|各自|(?<![A-Za-z])or(?![A-Za-z])",
    re.I,
)
_JOINT_CUE_PATTERN = re.compile(
    r"(?:[二三四2-4]\s*药\s*联合|[二三四2-4]\s*联|联合|联用|合用)"
)
_LEADING_CUE_PATTERN = re.compile(
    r"^(\s*)(?:[二三四2-4]\s*药\s*联合|[二三四2-4]\s*联|联合|联用|合用)\s*"
)
_CONNECTORS = {
    "+": "+",
    "＋": "＋",
    "plus": " plus ",
    "联合": "联合",
    "联用": "联用",
    "合用": "合用",
    "、": "、",
    ",": ",",
    "，": "，",
    "和": "和",
    "与": "与",
}
_STRONG_CONNECTORS = {"+", "＋", " plus ", "联合", "联用", "合用"}


@dataclass(frozen=True)
class DerivedQuery:
    """一次可靠局部重组得到的查询文本及被省略药物。"""

    query_text: str
    derivation_type: str
    omitted_concept_ids: tuple[str, ...]


@dataclass(frozen=True)
class RegimenExpansionResult:
    """联合用药文本派生结果，以及需要由编排层记录的超限方案。"""

    queries: tuple[DerivedQuery, ...]
    skipped_regimen_concept_ids: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _DrugScheme:
    start: int
    end: int
    drugs: tuple[MatchedEntity, ...]
    connector: str


def _parse_connector(gap: str) -> str | None:
    """只接受由空白和一个已确认连接词组成的实体间文本。"""
    return _CONNECTORS.get(re.sub(r"\s+", "", gap).lower())


def _make_scheme(
    text: str,
    drugs: list[MatchedEntity],
    connector: str,
    sentence_end: int,
) -> _DrugScheme | None:
    """确认共享联合结构或治疗谓词，并按概念去除重复药物。"""
    unique_drugs: list[MatchedEntity] = []
    seen_concepts: set[str] = set()
    for drug in drugs:
        if drug.concept_id not in seen_concepts:
            unique_drugs.append(drug)
            seen_concepts.add(drug.concept_id)
    if len(unique_drugs) < 2:
        return None
    suffix = text[drugs[-1].end : sentence_end]
    has_joint_meaning = (
        connector in _STRONG_CONNECTORS
        or (connector == "和" and len(unique_drugs) >= 3)
        or _JOINT_CUE_PATTERN.search(suffix) is not None
        or "治疗" in suffix
    )
    if not has_joint_meaning:
        return None
    return _DrugScheme(
        start=drugs[0].start,
        end=drugs[-1].end,
        drugs=tuple(unique_drugs),
        connector=connector,
    )


def _find_schemes(
    text: str,
    entities: list[MatchedEntity],
) -> Iterator[_DrugScheme]:
    """按句末标点切分，并寻找连接符一致的已知药物序列。"""
    for sentence_match in _SENTENCE_PATTERN.finditer(text):
        if _BLOCKER_PATTERN.search(sentence_match.group()) is not None:
            continue
        drugs = [
            entity
            for entity in entities
            if entity.category == "drug"
            and entity.start >= sentence_match.start()
            and entity.end <= sentence_match.end()
        ]
        if len(drugs) < 2:
            continue
        connectors = [
            _parse_connector(text[left.end : right.start])
            for left, right in zip(drugs, drugs[1:])
        ]
        connector = connectors[0]
        if connector is None or any(item != connector for item in connectors):
            continue
        scheme = _make_scheme(text, drugs, connector, sentence_match.end())
        if scheme is not None:
            yield scheme


def _adjust_suffix(suffix: str, remaining_drug_count: int) -> str:
    """单药删除联合提示；部分组合把药物数量提示泛化为“联合”。"""
    match = _LEADING_CUE_PATTERN.match(suffix)
    if match is None:
        return suffix
    replacement = "联合" if remaining_drug_count >= 2 else ""
    return match.group(1) + replacement + suffix[match.end() :]


def _render_query(
    text: str,
    scheme: _DrugScheme,
    kept_concept_ids: tuple[str, ...],
) -> str:
    """只替换已确认的方案区间，避免清洗问题中的其他内容。"""
    kept = [drug for drug in scheme.drugs if drug.concept_id in kept_concept_ids]
    replacement = scheme.connector.join(text[item.start : item.end] for item in kept)
    suffix = _adjust_suffix(text[scheme.end :], len(kept))
    return text[: scheme.start] + replacement + suffix


def _derive_scheme_queries(
    text: str,
    scheme: _DrugScheme,
    min_combination_drugs: int,
) -> Iterator[DerivedQuery]:
    """依次生成 combinations 顺序的部分组合和原文顺序的单药。"""
    drug_ids = tuple(drug.concept_id for drug in scheme.drugs)
    if len(drug_ids) >= min_combination_drugs:
        for kept_ids in combinations(drug_ids, len(drug_ids) - 1):
            omitted = tuple(item for item in drug_ids if item not in kept_ids)
            yield DerivedQuery(
                _render_query(text, scheme, kept_ids),
                PARTIAL_COMBINATION,
                omitted,
            )
    for kept_id in drug_ids:
        omitted = tuple(item for item in drug_ids if item != kept_id)
        yield DerivedQuery(
            _render_query(text, scheme, (kept_id,)),
            SINGLE_DRUG,
            omitted,
        )


def derive_regimen_queries(
    text: str,
    entities: list[MatchedEntity],
    min_combination_drugs: int,
    max_combination_drugs: int,
) -> RegimenExpansionResult:
    """生成中立派生文本，并单独返回超过药物数量上限的方案。"""
    queries: list[DerivedQuery] = []
    skipped_regimens: list[tuple[str, ...]] = []
    for scheme in _find_schemes(text, entities):
        drug_ids = tuple(drug.concept_id for drug in scheme.drugs)
        if len(drug_ids) > max_combination_drugs:
            skipped_regimens.append(drug_ids)
            continue
        queries.extend(_derive_scheme_queries(text, scheme, min_combination_drugs))
    return RegimenExpansionResult(tuple(queries), tuple(skipped_regimens))
