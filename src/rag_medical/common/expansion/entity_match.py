"""
用途：在一段原始文本中精确识别可信词典已收录的疾病和药物实体。
输入：原始文本，以及 load_aliases() 返回的按 concept_id 分组的词典。
输出：带原文坐标的 MatchedEntity 列表和跨概念歧义区间。
不做什么：不生成查询、不判断实体角色，不做否定、模糊或模型匹配。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator

from rag_medical.common.expansion.aliases import AliasEntry


@dataclass(frozen=True)
class MatchedEntity:
    """原始文本中的一次确定实体出现。"""
    concept_id: str
    category: str
    matched_term: str
    start: int
    end: int
    matched_alias: str
    term_kind: str
    position_unreliable: bool = False


@dataclass
class MatchResult:
    """实体匹配结果和不应自动扩展的歧义区间。"""
    entities: list[MatchedEntity]
    ambiguous_spans: list[tuple[int, int, list[str]]]

@dataclass(frozen=True)
class _MappedCharacter:
    value: str
    original_start: int
    original_end: int
    starts_source: bool
    ends_source: bool


@dataclass(frozen=True)
class _MatchCandidate:
    normalized_start: int
    normalized_end: int
    entity: MatchedEntity


def _make_segment(
    value: str,
    original_start: int,
    original_end: int,
    starts_source: bool = True,
    ends_source: bool = True,
) -> list[_MappedCharacter]:
    return [
        _MappedCharacter(
            character,
            original_start,
            original_end,
            starts_source and index == 0,
            ends_source and index == len(value) - 1,
        )
        for index, character in enumerate(value)
    ]


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    limit = min(len(left), len(right))
    while length < limit and left[length] == right[length]:
        length += 1
    return length


def _normalize_nfkc_by_prefix(text: str) -> list[_MappedCharacter]:
    characters: list[_MappedCharacter] = []
    previous = ""
    for original_index in range(len(text)):
        current = unicodedata.normalize("NFKC", text[: original_index + 1])
        stable_length = _common_prefix_length(previous, current)
        # 不在一个原字符的展开结果中间切断，否则无法回到可靠的原文边界。
        while stable_length and not characters[stable_length - 1].ends_source:
            stable_length -= 1
        changed = characters[stable_length:]
        changed_start = min([original_index] + [item.original_start for item in changed])
        characters = characters[:stable_length]
        characters.extend(_make_segment(current[stable_length:], changed_start, original_index + 1))
        previous = current
    return characters


def _normalize_nfkc(text: str) -> list[_MappedCharacter]:
    characters: list[_MappedCharacter] = []
    for original_index, character in enumerate(text):
        characters.extend(
            _make_segment(
                unicodedata.normalize("NFKC", character),
                original_index,
                original_index + 1,
            )
        )
    normalized = "".join(character.value for character in characters)
    if normalized == unicodedata.normalize("NFKC", text):
        return characters
    # 只有组合字符等跨字符 NFKC 情况才需要较慢的前缀重建。
    return _normalize_nfkc_by_prefix(text)


def _lower_characters(characters: list[_MappedCharacter]) -> list[_MappedCharacter]:
    lowered: list[_MappedCharacter] = []
    for character in characters:
        lowered.extend(
            _make_segment(
                character.value.lower(),
                character.original_start,
                character.original_end,
                character.starts_source,
                character.ends_source,
            )
        )
    return lowered


def _fold_whitespace(characters: list[_MappedCharacter]) -> list[_MappedCharacter]:
    folded: list[_MappedCharacter] = []
    previous_was_whitespace = False
    for character in characters:
        is_whitespace = character.value.isspace()
        if is_whitespace and not previous_was_whitespace:
            # 规范要求折叠空格仍映射到第一个原始空白。
            folded.append(
                _MappedCharacter(
                    " ",
                    character.original_start,
                    character.original_end,
                    character.starts_source,
                    character.ends_source,
                )
            )
        elif not is_whitespace:
            folded.append(character)
        previous_was_whitespace = is_whitespace
    if folded and folded[0].value == " ":
        folded = folded[1:]
    if folded and folded[-1].value == " ":
        folded = folded[:-1]
    return folded


def _normalize_with_mapping(text: str) -> tuple[str, tuple[_MappedCharacter, ...]]:
    characters = _lower_characters(_normalize_nfkc(text))
    characters = _fold_whitespace(characters)
    normalized = "".join(character.value for character in characters)
    return normalized, tuple(characters)


def _prepare_aliases(
    aliases: dict[str, list[AliasEntry]],
) -> list[tuple[AliasEntry, str]]:
    prepared = [
        (entry, _normalize_with_mapping(entry.term)[0])
        for entries in aliases.values()
        for entry in entries
    ]
    prepared = [(entry, term) for entry, term in prepared if term]
    return sorted(prepared, key=lambda item: (-len(item[1]), item[0].concept_id, item[0].term))


def _find_spans(text: str, entry: AliasEntry, term: str) -> Iterator[tuple[int, int]]:
    if entry.language == "en":
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])")
        for match in pattern.finditer(text):
            yield match.span()
        return
    search_start = 0
    while True:
        match_start = text.find(term, search_start)
        if match_start < 0:
            return
        yield match_start, match_start + len(term)
        search_start = match_start + 1


def _make_candidate(
    original_text: str,
    characters: tuple[_MappedCharacter, ...],
    entry: AliasEntry,
    normalized_start: int,
    normalized_end: int,
) -> _MatchCandidate:
    first = characters[normalized_start]
    last = characters[normalized_end - 1]
    original_start, original_end = first.original_start, last.original_end
    covered = characters[normalized_start:normalized_end]
    unreliable = not first.starts_source or not last.ends_source or any(
        item.original_start < original_start or item.original_end > original_end
        for item in covered
    )
    entity = MatchedEntity(
        concept_id=entry.concept_id,
        category=entry.category,
        matched_term=original_text[original_start:original_end],
        start=original_start,
        end=original_end,
        matched_alias=entry.term,
        term_kind=entry.term_kind,
        position_unreliable=unreliable,
    )
    return _MatchCandidate(normalized_start, normalized_end, entity)


def _collect_candidates(
    text: str,
    normalized_text: str,
    characters: tuple[_MappedCharacter, ...],
    aliases: dict[str, list[AliasEntry]],
) -> list[_MatchCandidate]:
    candidates: list[_MatchCandidate] = []
    for entry, term in _prepare_aliases(aliases):
        for normalized_start, normalized_end in _find_spans(normalized_text, entry, term):
            candidates.append(
                _make_candidate(text, characters, entry, normalized_start, normalized_end)
            )
    return candidates


def _overlaps(left: _MatchCandidate, right: _MatchCandidate) -> bool:
    return (
        left.normalized_start < right.normalized_end
        and right.normalized_start < left.normalized_end
    )


def _overlap_components(candidates: list[_MatchCandidate]) -> list[list[_MatchCandidate]]:
    ordered = sorted(candidates, key=lambda item: (item.normalized_start, item.normalized_end))
    components: list[list[_MatchCandidate]] = []
    current: list[_MatchCandidate] = []
    current_end = -1
    for candidate in ordered:
        if current and candidate.normalized_start >= current_end:
            components.append(current)
            current = []
        current.append(candidate)
        current_end = max(current_end, candidate.normalized_end)
    if current:
        components.append(current)
    return components


def _select_longest(component: list[_MatchCandidate]) -> list[MatchedEntity]:
    ordered = sorted(
        component,
        key=lambda item: (
            -(item.normalized_end - item.normalized_start),
            item.normalized_start,
            item.entity.matched_alias,
        ),
    )
    selected: list[_MatchCandidate] = []
    for candidate in ordered:
        if not any(_overlaps(candidate, existing) for existing in selected):
            selected.append(candidate)
    return [candidate.entity for candidate in selected]


def _resolve_candidates(candidates: list[_MatchCandidate]) -> MatchResult:
    entities: list[MatchedEntity] = []
    ambiguous_spans: list[tuple[int, int, list[str]]] = []
    for component in _overlap_components(candidates):
        concept_ids = sorted({candidate.entity.concept_id for candidate in component})
        if len(concept_ids) == 1:
            entities.extend(_select_longest(component))
            continue
        ambiguous_spans.append(
            (
                min(candidate.entity.start for candidate in component),
                max(candidate.entity.end for candidate in component),
                concept_ids,
            )
        )
    entities.sort(key=lambda entity: (entity.start, entity.end, entity.concept_id))
    ambiguous_spans.sort(key=lambda span: (span[0], span[1]))
    return MatchResult(entities, ambiguous_spans)


def match_entities(
    text: str,
    aliases: dict[str, list[AliasEntry]],
) -> MatchResult:
    """在原始文本中精确匹配词典实体，并返回原文坐标。"""
    normalized_text, characters = _normalize_with_mapping(text)
    if not normalized_text:
        return MatchResult([], [])
    candidates = _collect_candidates(text, normalized_text, characters, aliases)
    return _resolve_candidates(candidates)
