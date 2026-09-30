"""加载并查询只读医学术语词典。

输入：``canonical_terms.jsonl`` 与 ``aliases.jsonl`` 两个正式词典文件。
输出：内存中的精确索引、Trie、冲突清单，以及经过语言/类别/n-gram 限定的
      模糊候选；本模块不修改文本，也不写回任何词典文件。

设计约束：
1. 别名通过稳定的 ``concept_id`` 指向规范词，不直接指向字符串。
2. ``compact_key`` 在载入时生成，不在数据文件中人工维护。
3. 同一 compact_key 指向不同概念时隔离冲突，禁止静默覆盖。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


COMPACT_SEPARATORS_RE = re.compile(r"[\s\-_‐‑‒–—−]+")


def compact_key(text: str) -> str:
    """生成匹配键：统一字符宽度与大小写，仅忽略空白和连字符差异。"""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return COMPACT_SEPARATORS_RE.sub("", normalized)


def character_ngrams(text: str, size: int = 3) -> frozenset[str]:
    """生成字符 n-gram；短字符串作为一个整体，不退化为全词典扫描。"""

    if not text:
        return frozenset()
    if len(text) <= size:
        return frozenset({text})
    return frozenset(text[index : index + size] for index in range(len(text) - size + 1))


def bounded_levenshtein(left: str, right: str, max_distance: int) -> int:
    """计算带上限的 Levenshtein 距离，超过上限时提前结束。"""

    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    if len(left) > len(right):
        left, right = right, left

    previous = list(range(len(left) + 1))
    for row_index, right_char in enumerate(right, start=1):
        current = [row_index]
        row_minimum = row_index
        for column_index, left_char in enumerate(left, start=1):
            value = min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_char != right_char),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class CanonicalTerm:
    concept_id: str
    canonical: str
    language: str
    category: str
    source: str
    confidence: float


@dataclass(frozen=True)
class AliasTerm:
    alias: str
    concept_id: str
    language: str
    category: str
    source: str
    confidence: float
    auto_replace: bool


@dataclass(frozen=True)
class ResolvedTerm:
    """可供规范化器使用的只读词条。"""

    key: str
    canonical: str
    concept_id: str
    language: str
    category: str
    source: str
    confidence: float
    match_source: str
    auto_replace: bool


@dataclass(frozen=True)
class DictionaryConflict:
    language: str
    key: str
    concept_ids: tuple[str, ...]
    forms: tuple[str, ...]


@dataclass(frozen=True)
class FuzzyMatch:
    term: ResolvedTerm
    distance: int
    similarity: float
    shared_ngram_count: int


class _TrieNode:
    __slots__ = ("children", "term")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.term: ResolvedTerm | None = None


class CompactTrie:
    """按 compact_key 建立的前缀树，用于枚举短语候选。"""

    def __init__(self, terms: Iterable[ResolvedTerm]) -> None:
        self.root = _TrieNode()
        for term in terms:
            node = self.root
            for char in term.key:
                node = node.children.setdefault(char, _TrieNode())
            node.term = term

    def iter_matches(self, text: str, start: int = 0) -> Iterator[tuple[int, ResolvedTerm]]:
        """返回从 start 开始命中的 ``(结束位置, 词条)``，结束位置为开区间。"""

        node = self.root
        for index in range(start, len(text)):
            node = node.children.get(text[index])
            if node is None:
                return
            if node.term is not None:
                yield index + 1, node.term


class TermDictionary:
    """不可变的医学术语查询索引。"""

    def __init__(
        self,
        canonical_terms: Sequence[CanonicalTerm],
        aliases: Sequence[AliasTerm],
        *,
        ngram_size: int = 3,
    ) -> None:
        self.canonical_terms = tuple(canonical_terms)
        self.aliases = tuple(aliases)
        self.ngram_size = ngram_size
        self._canonical_by_concept: dict[tuple[str, str], CanonicalTerm] = {}

        for term in self.canonical_terms:
            identity = (term.concept_id, term.language)
            existing = self._canonical_by_concept.get(identity)
            if existing is not None and existing.canonical != term.canonical:
                raise ValueError(
                    "multiple canonical terms for "
                    f"concept_id={term.concept_id!r}, language={term.language!r}"
                )
            self._canonical_by_concept[identity] = term

        grouped: dict[tuple[str, str], list[ResolvedTerm]] = defaultdict(list)
        for term in self.canonical_terms:
            key = compact_key(term.canonical)
            if not key:
                raise ValueError(f"empty compact_key for canonical term {term.canonical!r}")
            grouped[(term.language, key)].append(
                ResolvedTerm(
                    key=key,
                    canonical=term.canonical,
                    concept_id=term.concept_id,
                    language=term.language,
                    category=term.category,
                    source=term.source,
                    confidence=term.confidence,
                    match_source="canonical",
                    auto_replace=True,
                )
            )

        for alias in self.aliases:
            canonical = self._canonical_by_concept.get((alias.concept_id, alias.language))
            if canonical is None:
                raise ValueError(
                    f"alias {alias.alias!r} refers to an unknown concept/language: "
                    f"{alias.concept_id!r}/{alias.language!r}"
                )
            if alias.category != canonical.category:
                raise ValueError(
                    f"alias {alias.alias!r} category {alias.category!r} does not match "
                    f"canonical category {canonical.category!r}"
                )
            key = compact_key(alias.alias)
            if not key:
                raise ValueError(f"empty compact_key for alias {alias.alias!r}")
            grouped[(alias.language, key)].append(
                ResolvedTerm(
                    key=key,
                    canonical=canonical.canonical,
                    concept_id=canonical.concept_id,
                    language=canonical.language,
                    category=canonical.category,
                    source=alias.source,
                    confidence=min(alias.confidence, canonical.confidence),
                    match_source="alias",
                    auto_replace=alias.auto_replace,
                )
            )

        self.conflicts: tuple[DictionaryConflict, ...]
        conflicts: list[DictionaryConflict] = []
        self._resolved: dict[tuple[str, str], ResolvedTerm] = {}
        for identity, candidates in grouped.items():
            concept_ids = {candidate.concept_id for candidate in candidates}
            if len(concept_ids) > 1:
                conflicts.append(
                    DictionaryConflict(
                        language=identity[0],
                        key=identity[1],
                        concept_ids=tuple(sorted(concept_ids)),
                        forms=tuple(sorted({candidate.canonical for candidate in candidates})),
                    )
                )
                continue
            # 同一概念的 canonical 与 alias 共用 compact_key 时，优先 canonical 来源。
            self._resolved[identity] = sorted(
                candidates,
                key=lambda item: (item.match_source != "canonical", -item.confidence),
            )[0]
        self.conflicts = tuple(sorted(conflicts, key=lambda item: (item.language, item.key)))

        terms_by_language: dict[str, list[ResolvedTerm]] = defaultdict(list)
        self._ngram_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._keys_by_language: dict[str, set[str]] = defaultdict(set)
        for (language, key), term in self._resolved.items():
            terms_by_language[language].append(term)
            self._keys_by_language[language].add(key)
            for ngram in character_ngrams(key, self.ngram_size):
                self._ngram_index[(language, ngram)].add(key)
        self._tries = {
            language: CompactTrie(terms)
            for language, terms in terms_by_language.items()
        }

    @property
    def term_count(self) -> int:
        return len(self._resolved)

    def canonical_for(self, concept_id: str, language: str) -> CanonicalTerm | None:
        return self._canonical_by_concept.get((concept_id, language))

    def resolve_compact(
        self,
        text: str,
        language: str,
        allowed_categories: Iterable[str] | None = None,
    ) -> ResolvedTerm | None:
        """精确解析一个文本形式；冲突 key 和禁用自动替换的别名不会返回。"""

        term = self._resolved.get((language, compact_key(text)))
        if term is None or not term.auto_replace:
            return None
        if allowed_categories is not None and term.category not in set(allowed_categories):
            return None
        return term

    def iter_compact_matches(
        self,
        compact_text: str,
        language: str,
        start: int = 0,
        allowed_categories: Iterable[str] | None = None,
    ) -> Iterator[tuple[int, ResolvedTerm]]:
        """从 compact 文本指定位置枚举 Trie 精确命中。"""

        trie = self._tries.get(language)
        if trie is None:
            return
        categories = set(allowed_categories) if allowed_categories is not None else None
        for end, term in trie.iter_matches(compact_text, start):
            if term.auto_replace and (categories is None or term.category in categories):
                yield end, term

    def fuzzy_matches(
        self,
        text: str,
        language: str,
        *,
        allowed_categories: Iterable[str] | None,
        max_distance: int,
        min_similarity: float,
        limit: int = 8,
        shortlist_limit: int = 128,
    ) -> list[FuzzyMatch]:
        """返回严格受限的模糊候选，不在整个词典上逐项扫描。

        候选首先来自字符 n-gram 倒排索引，再按语言、类别和长度过滤。调用方仍需
        检查第一名与第二名的差距，才能决定是否自动替换。
        """

        observed = compact_key(text)
        if not observed or max_distance < 0:
            return []
        categories = set(allowed_categories) if allowed_categories is not None else None
        overlap_counts: dict[str, int] = defaultdict(int)
        for ngram in character_ngrams(observed, self.ngram_size):
            for key in self._ngram_index.get((language, ngram), ()):
                overlap_counts[key] += 1
        if not overlap_counts:
            return []

        shortlist = sorted(
            overlap_counts,
            key=lambda key: (-overlap_counts[key], abs(len(key) - len(observed)), key),
        )[:shortlist_limit]
        matches: list[FuzzyMatch] = []
        for key in shortlist:
            if abs(len(key) - len(observed)) > max_distance:
                continue
            term = self._resolved[(language, key)]
            if not term.auto_replace:
                continue
            if categories is not None and term.category not in categories:
                continue
            distance = bounded_levenshtein(observed, key, max_distance)
            if distance > max_distance:
                continue
            similarity = 1.0 - distance / max(len(observed), len(key), 1)
            if similarity < min_similarity:
                continue
            matches.append(
                FuzzyMatch(
                    term=term,
                    distance=distance,
                    similarity=similarity,
                    shared_ngram_count=overlap_counts[key],
                )
            )
        return sorted(
            matches,
            key=lambda item: (
                -item.similarity,
                item.distance,
                -item.shared_ngram_count,
                item.term.canonical,
            ),
        )[:limit]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(record)
    return records


def _required_string(record: dict[str, object], field: str, path: Path) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty {field!r} in {path}")
    return value.strip()


def load_dictionary(
    canonical_path: Path,
    aliases_path: Path,
    *,
    ngram_size: int = 3,
) -> TermDictionary:
    """从正式 JSONL 资源创建只读词典；仅执行载入和校验，不写文件。"""

    canonical_terms: list[CanonicalTerm] = []
    for record in _read_jsonl(canonical_path):
        canonical_terms.append(
            CanonicalTerm(
                concept_id=_required_string(record, "concept_id", canonical_path),
                canonical=_required_string(record, "canonical", canonical_path),
                language=_required_string(record, "language", canonical_path),
                category=_required_string(record, "category", canonical_path),
                source=_required_string(record, "source", canonical_path),
                confidence=float(record.get("confidence", 1.0)),
            )
        )

    aliases: list[AliasTerm] = []
    for record in _read_jsonl(aliases_path):
        alias = _required_string(record, "alias", aliases_path)
        aliases.append(
            AliasTerm(
                alias=alias,
                concept_id=_required_string(record, "concept_id", aliases_path),
                language=_required_string(record, "language", aliases_path),
                category=_required_string(record, "category", aliases_path),
                source=_required_string(record, "source", aliases_path),
                confidence=float(record.get("confidence", 1.0)),
                auto_replace=bool(record.get("auto_replace", len(compact_key(alias)) >= 6)),
            )
        )
    return TermDictionary(canonical_terms, aliases, ngram_size=ngram_size)
