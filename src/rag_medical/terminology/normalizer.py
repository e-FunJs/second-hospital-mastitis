"""使用只读医学词典执行保守的 OCR 术语规范化。

输入：原始文本、已加载的 ``TermDictionary``、规范化配置，以及可选语言/类别上下文。
输出：``NormalizationResult``，包含规范化文本、逐项修改记录和未自动处理的可疑片段。
      本模块是纯计算逻辑，不读取或写入候选日志，也不会修改正式词典。

处理顺序：NFKC -> 词典精确匹配与全覆盖分词 -> 严格模糊匹配。正确文本若不需要
规范化则原样保留；数字、剂量、单位和低置信度候选不会被自动改写。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rag_medical.terminology.dictionary import (
    FuzzyMatch,
    ResolvedTerm,
    TermDictionary,
    compact_key,
    load_dictionary,
)


LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
JOINABLE_GAP_RE = re.compile(r"(?:[ \t]+|[\-‐‑‒–—−][ \t]*|[\-‐‑‒–—−][ \t]*\n[ \t]*)")
DIGIT_RE = re.compile(r"\d")
DOSE_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:mg|g|kg|ml|l|mmol|mol|μg|ug|iu|u|qd|bid|tid|qid|q\d+h|%)(?![A-Za-z])",
    re.IGNORECASE,
)
COMMON_SHORT_WORDS = {
    "a",
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "if",
    "in",
    "is",
    "it",
    "me",
    "no",
    "of",
    "on",
    "or",
    "so",
    "to",
    "up",
    "us",
    "we",
}


@dataclass(frozen=True)
class FuzzyThreshold:
    min_length: int
    max_length: int
    max_distance: int
    min_similarity: float


@dataclass(frozen=True)
class NormalizerSettings:
    enabled_languages: tuple[str, ...] = ("en",)
    min_fuzzy_length: int = 6
    top_margin: float = 0.05
    max_fuzzy_candidates: int = 8
    max_shortlist_size: int = 128
    max_fragment_window: int = 10
    max_compact_span_length: int = 64
    fuzzy_thresholds: tuple[FuzzyThreshold, ...] = (
        FuzzyThreshold(6, 8, 1, 0.83),
        FuzzyThreshold(9, 15, 1, 0.88),
        FuzzyThreshold(16, 64, 2, 0.875),
    )

    def threshold_for(self, length: int) -> FuzzyThreshold | None:
        for threshold in self.fuzzy_thresholds:
            if threshold.min_length <= length <= threshold.max_length:
                return threshold
        return None


@dataclass(frozen=True)
class TermReplacement:
    before: str
    after: str
    start: int
    end: int
    concept_id: str
    category: str
    match_type: str
    score: float


@dataclass(frozen=True)
class UnresolvedSpan:
    text: str
    start: int
    end: int
    reason: str
    candidates: tuple[str, ...] = ()
    candidate_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class NormalizationResult:
    original_text: str
    normalized_text: str
    replacements: tuple[TermReplacement, ...] = ()
    unresolved_spans: tuple[UnresolvedSpan, ...] = ()
    actions: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.original_text != self.normalized_text


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    replacement: str
    terms: tuple[ResolvedTerm, ...]
    match_type: str
    score: float


def _allowed_categories(context: Mapping[str, Any] | None) -> set[str] | None:
    if not context:
        return None
    raw = context.get("allowed_categories")
    if raw is None:
        return None
    if isinstance(raw, str):
        return {raw}
    return {str(item) for item in raw}


def _is_joinable_gap(gap: str) -> bool:
    """普通空格和明确的行末连字符可连接；裸换行是边界，不能猜测合并。"""

    return bool(gap and JOINABLE_GAP_RE.fullmatch(gap))


def _layout_equal(left: str, right: str) -> bool:
    """忽略大小写比较版式；不会仅因句首大写而改写正确术语。"""

    return unicodedata.normalize("NFKC", left).casefold() == right.casefold()


def _exact_window_candidates(
    text: str,
    dictionary: TermDictionary,
    settings: NormalizerSettings,
    categories: set[str] | None,
) -> list[_Candidate]:
    words = list(LATIN_WORD_RE.finditer(text))
    candidates: list[_Candidate] = []
    for start_index, first in enumerate(words):
        end_limit = min(len(words), start_index + settings.max_fragment_window)
        for end_index in range(start_index, end_limit):
            current = words[end_index]
            if end_index > start_index:
                previous = words[end_index - 1]
                if not _is_joinable_gap(text[previous.end() : current.start()]):
                    break
            before = text[first.start() : current.end()]
            key = compact_key(before)
            if not key or len(key) > settings.max_compact_span_length:
                break
            term = dictionary.resolve_compact(before, "en", categories)
            if term is None or _layout_equal(before, term.canonical):
                continue
            candidates.append(
                _Candidate(
                    start=first.start(),
                    end=current.end(),
                    replacement=term.canonical,
                    terms=(term,),
                    match_type="compact_exact",
                    # 平方长度让完整医学短语优先于其内部较短词条。
                    score=float(len(term.key) ** 2),
                )
            )
    return candidates


def _segment_compact_token(
    token: str,
    dictionary: TermDictionary,
    categories: set[str] | None,
) -> tuple[ResolvedTerm, ...] | None:
    """用 DP 寻找完整覆盖单个黏连词的最优多术语路径。"""

    key = compact_key(token)
    if not key:
        return None
    # dp[position] = (累计分数, 词条序列)；不允许跳过字符，因此不会部分猜测。
    dp: list[tuple[float, tuple[ResolvedTerm, ...]] | None] = [None] * (len(key) + 1)
    dp[0] = (0.0, ())
    for start in range(len(key)):
        state = dp[start]
        if state is None:
            continue
        for end, term in dictionary.iter_compact_matches(key, "en", start, categories):
            if len(term.key) < 6:
                continue
            candidate = (state[0] + len(term.key) ** 2, state[1] + (term,))
            existing = dp[end]
            if existing is None or candidate[0] > existing[0]:
                dp[end] = candidate
    final = dp[-1]
    if final is None or len(final[1]) < 2:
        return None
    return final[1]


def _segmentation_candidates(
    text: str,
    dictionary: TermDictionary,
    settings: NormalizerSettings,
    categories: set[str] | None,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for match in LATIN_WORD_RE.finditer(text):
        token = match.group(0)
        if not 12 <= len(token) <= settings.max_compact_span_length:
            continue
        terms = _segment_compact_token(token, dictionary, categories)
        if terms is None:
            continue
        replacement = " ".join(term.canonical for term in terms)
        if _layout_equal(token, replacement):
            continue
        candidates.append(
            _Candidate(
                start=match.start(),
                end=match.end(),
                replacement=replacement,
                terms=terms,
                match_type="dictionary_segmentation",
                score=float(sum(len(term.key) ** 2 for term in terms)),
            )
        )
    return candidates


def _select_non_overlapping(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    """用加权区间 DP 选取全局最优且互不重叠的替换。"""

    ordered = sorted(candidates, key=lambda item: (item.end, item.start, -item.score))
    if not ordered:
        return []
    previous: list[int] = []
    for index, candidate in enumerate(ordered):
        compatible = index - 1
        while compatible >= 0 and ordered[compatible].end > candidate.start:
            compatible -= 1
        previous.append(compatible)

    scores = [0.0] * (len(ordered) + 1)
    chosen: list[tuple[int, ...]] = [()] * (len(ordered) + 1)
    for index, candidate in enumerate(ordered, start=1):
        include_from = previous[index - 1] + 1
        include_score = scores[include_from] + candidate.score
        if include_score > scores[index - 1]:
            scores[index] = include_score
            chosen[index] = chosen[include_from] + (index - 1,)
        else:
            scores[index] = scores[index - 1]
            chosen[index] = chosen[index - 1]
    return sorted((ordered[index] for index in chosen[-1]), key=lambda item: item.start)


def _apply_candidates(
    text: str,
    candidates: Sequence[_Candidate],
) -> tuple[str, list[TermReplacement]]:
    result = text
    replacements: list[TermReplacement] = []
    for candidate in reversed(candidates):
        before = result[candidate.start : candidate.end]
        result = result[: candidate.start] + candidate.replacement + result[candidate.end :]
        first_term = candidate.terms[0]
        replacements.append(
            TermReplacement(
                before=before,
                after=candidate.replacement,
                start=candidate.start,
                end=candidate.end,
                concept_id="+".join(term.concept_id for term in candidate.terms),
                category=first_term.category,
                match_type=candidate.match_type,
                score=candidate.score,
            )
        )
    return result, sorted(replacements, key=lambda item: item.start)


def _safe_fuzzy_choice(matches: Sequence[FuzzyMatch], top_margin: float) -> FuzzyMatch | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if matches[0].similarity - matches[1].similarity < top_margin:
        return None
    return matches[0]


def _contains_protected_measurement(text: str) -> bool:
    return bool(DIGIT_RE.search(text) or DOSE_UNIT_RE.search(text))


def _fuzzy_candidates(
    text: str,
    dictionary: TermDictionary,
    settings: NormalizerSettings,
    categories: set[str] | None,
) -> tuple[list[_Candidate], list[UnresolvedSpan]]:
    candidates: list[_Candidate] = []
    unresolved: list[UnresolvedSpan] = []
    for match in LATIN_WORD_RE.finditer(text):
        observed = match.group(0)
        key = compact_key(observed)
        threshold = settings.threshold_for(len(key))
        if (
            threshold is None
            or len(key) < settings.min_fuzzy_length
            or _contains_protected_measurement(observed)
            or dictionary.resolve_compact(observed, "en", categories) is not None
        ):
            continue
        matches = dictionary.fuzzy_matches(
            observed,
            "en",
            allowed_categories=categories,
            max_distance=threshold.max_distance,
            min_similarity=threshold.min_similarity,
            limit=settings.max_fuzzy_candidates,
            shortlist_limit=settings.max_shortlist_size,
        )
        choice = _safe_fuzzy_choice(matches, settings.top_margin)
        if choice is None:
            if matches:
                unresolved.append(
                    UnresolvedSpan(
                        text=observed,
                        start=match.start(),
                        end=match.end(),
                        reason="ambiguous_fuzzy_candidates",
                        candidates=tuple(item.term.canonical for item in matches[:3]),
                        candidate_scores=tuple(item.similarity for item in matches[:3]),
                    )
                )
            continue
        if observed.casefold() == choice.term.canonical.casefold():
            continue
        candidates.append(
            _Candidate(
                start=match.start(),
                end=match.end(),
                replacement=choice.term.canonical,
                terms=(choice.term,),
                match_type="strict_fuzzy",
                score=choice.similarity,
            )
        )
    return candidates, unresolved


def _fragment_unresolved_spans(
    text: str,
    settings: NormalizerSettings,
) -> list[UnresolvedSpan]:
    """仅记录高密度异常短片段，避免把正常英文功能词记成候选。"""

    words = list(LATIN_WORD_RE.finditer(text))
    unresolved: list[UnresolvedSpan] = []
    index = 0
    while index < len(words):
        end = index
        while end + 1 < len(words) and _is_joinable_gap(text[words[end].end() : words[end + 1].start()]):
            end += 1
        run = words[index : end + 1]
        short_fragments = [item.group(0).casefold() for item in run if len(item.group(0)) <= 2]
        before = text[run[0].start() : run[-1].end()] if run else ""
        if (
            4 <= len(run) <= settings.max_fragment_window
            and len(short_fragments) >= 3
            and len(short_fragments) / len(run) >= 0.60
            and any(fragment not in COMMON_SHORT_WORDS for fragment in short_fragments)
            and len(compact_key(before)) <= 40
            and not _contains_protected_measurement(before)
        ):
            unresolved.append(
                UnresolvedSpan(
                    text=before,
                    start=run[0].start(),
                    end=run[-1].end(),
                    reason="unknown_fragmented_latin",
                )
            )
        index = end + 1
    return unresolved


def normalize_text(
    text: str,
    *,
    dictionary: TermDictionary,
    settings: NormalizerSettings | None = None,
    language: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> NormalizationResult:
    """纯函数式公开接口；调用方无需接触 Trie、compact_key 或模糊索引。"""

    del language  # 文档语言仅作未来扩展；中文文献中的英文药名仍必须使用英文词典。
    settings = settings or NormalizerSettings()
    original = str(text or "")
    normalized = unicodedata.normalize("NFKC", original)
    actions: list[str] = []
    if normalized != original:
        actions.append("unicode_nfkc_normalized")
    if "en" not in settings.enabled_languages:
        return NormalizationResult(original, normalized, actions=tuple(actions))

    categories = _allowed_categories(context)
    exact_candidates = _exact_window_candidates(normalized, dictionary, settings, categories)
    exact_candidates.extend(
        _segmentation_candidates(normalized, dictionary, settings, categories)
    )
    selected_exact = _select_non_overlapping(exact_candidates)
    after_exact, replacements = _apply_candidates(normalized, selected_exact)
    if selected_exact:
        actions.append("terminology_exact_repaired")

    fuzzy_candidates, unresolved = _fuzzy_candidates(
        after_exact, dictionary, settings, categories
    )
    selected_fuzzy = _select_non_overlapping(fuzzy_candidates)
    final_text, fuzzy_replacements = _apply_candidates(after_exact, selected_fuzzy)
    replacements.extend(fuzzy_replacements)
    if selected_fuzzy:
        actions.append("terminology_fuzzy_repaired")

    occupied = [(item.start, item.end) for item in unresolved]
    for item in _fragment_unresolved_spans(final_text, settings):
        if not any(item.start < end and item.end > start for start, end in occupied):
            unresolved.append(item)
    return NormalizationResult(
        original_text=original,
        normalized_text=final_text,
        replacements=tuple(replacements),
        unresolved_spans=tuple(unresolved),
        actions=tuple(actions),
    )


def settings_from_mapping(payload: Mapping[str, Any]) -> NormalizerSettings:
    normalization = payload.get("normalization", {})
    if not isinstance(normalization, Mapping):
        raise ValueError("terminology config 'normalization' must be a mapping")
    thresholds = tuple(
        FuzzyThreshold(
            min_length=int(item["min_length"]),
            max_length=int(item["max_length"]),
            max_distance=int(item["max_distance"]),
            min_similarity=float(item["min_similarity"]),
        )
        for item in normalization.get("fuzzy_thresholds", [])
    )
    defaults = NormalizerSettings()
    return NormalizerSettings(
        enabled_languages=tuple(normalization.get("enabled_languages", defaults.enabled_languages)),
        min_fuzzy_length=int(normalization.get("min_fuzzy_length", defaults.min_fuzzy_length)),
        top_margin=float(normalization.get("top_margin", defaults.top_margin)),
        max_fuzzy_candidates=int(
            normalization.get("max_fuzzy_candidates", defaults.max_fuzzy_candidates)
        ),
        max_shortlist_size=int(
            normalization.get("max_shortlist_size", defaults.max_shortlist_size)
        ),
        max_fragment_window=int(
            normalization.get("max_fragment_window", defaults.max_fragment_window)
        ),
        max_compact_span_length=int(
            normalization.get("max_compact_span_length", defaults.max_compact_span_length)
        ),
        fuzzy_thresholds=thresholds or defaults.fuzzy_thresholds,
    )


def load_normalizer_resources(
    config_path: Path,
) -> tuple[TermDictionary, NormalizerSettings, dict[str, Any]]:
    """按配置加载只读词典与参数；第三个返回值供候选维护脚本读取路径配置。"""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("terminology config root must be a mapping")
    dictionary_config = payload.get("dictionary", {})
    if not isinstance(dictionary_config, Mapping):
        raise ValueError("terminology config 'dictionary' must be a mapping")
    project_root = config_path.resolve().parent.parent

    def resolve_path(value: object, field_name: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing dictionary config field: {field_name}")
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    dictionary = load_dictionary(
        resolve_path(dictionary_config.get("canonical_path"), "canonical_path"),
        resolve_path(dictionary_config.get("aliases_path"), "aliases_path"),
        ngram_size=int(dictionary_config.get("ngram_size", 3)),
    )
    return dictionary, settings_from_mapping(payload), payload
