"""定位模型引句并恢复清洗后证据中的实际字符串。

输入：提交给模型的文本及模型返回的引句。输出：原文子串和匹配级别。
只允许严格匹配或 NFKC、空白折叠、首尾去空白后的匹配；不改变大小写，
不额外转换标点，不做语义判断、模糊匹配、文件读写或模型调用。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from enum import Enum


class QuoteMatchKind(str, Enum):
    """由程序确定的引句定位级别。"""

    STRICT = "strict"
    NORMALIZED = "normalized"


@dataclass(frozen=True)
class LocatedQuote:
    text: str  # 恢复出的原文实际字符串。
    match_kind: QuoteMatchKind  # 严格或归一化匹配级别。


@dataclass(frozen=True)
class _MappedCharacter:
    value: str  # 一个归一化字符。
    start: int  # 对应原文区间的起点。
    end: int  # 对应原文区间的终点，不含该位置。
    starts_source: bool  # 是否覆盖对应原文单元的开头。
    ends_source: bool  # 是否覆盖对应原文单元的结尾。


def _make_segment(value: str, start: int, end: int) -> list[_MappedCharacter]:
    return [
        _MappedCharacter(character, start, end, index == 0, index == len(value) - 1)
        for index, character in enumerate(value)
    ]


def _find_common_prefix(left: str, right: str) -> int:
    for index, (left_character, right_character) in enumerate(zip(left, right)):
        if left_character != right_character:
            return index
    return min(len(left), len(right))


def _map_composed_nfkc(text: str) -> list[_MappedCharacter]:
    characters: list[_MappedCharacter] = []
    previous = ""
    for original_index in range(len(text)):
        current = unicodedata.normalize("NFKC", text[: original_index + 1])
        stable_length = _find_common_prefix(previous, current)
        # 组合字符可能回改前缀；不可在一个连字展开单元的中间保留旧映射。
        while stable_length and not characters[stable_length - 1].ends_source:
            stable_length -= 1
        changed = characters[stable_length:]
        original_start = min([original_index] + [character.start for character in changed])
        characters = characters[:stable_length]
        characters.extend(_make_segment(current[stable_length:], original_start, original_index + 1))
        previous = current
    return characters


def _map_nfkc(text: str) -> list[_MappedCharacter]:
    characters = [
        character
        for index, original in enumerate(text)
        for character in _make_segment(unicodedata.normalize("NFKC", original), index, index + 1)
    ]
    if "".join(character.value for character in characters) == unicodedata.normalize("NFKC", text):
        return characters
    return _map_composed_nfkc(text)


def _fold_whitespace(characters: list[_MappedCharacter]) -> list[_MappedCharacter]:
    folded: list[_MappedCharacter] = []
    for character in characters:
        if character.value.isspace():
            if folded and folded[-1].value == " ":
                folded[-1] = replace(folded[-1], end=character.end, ends_source=character.ends_source)
            else:
                folded.append(replace(character, value=" "))
        else:
            folded.append(character)
    if folded and folded[0].value == " ":
        folded = folded[1:]
    if folded and folded[-1].value == " ":
        folded = folded[:-1]
    return folded


def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _restore_quote(text: str, quote: str) -> LocatedQuote:
    characters = _fold_whitespace(_map_nfkc(text))
    normalized_text = "".join(character.value for character in characters)
    normalized_quote = _normalize_quote(quote)
    # 前瞻保留重叠命中；多个归一化位置不能任意选第一个作为引用锚点。
    positions = [match.start() for match in re.finditer(f"(?={re.escape(normalized_quote)})", normalized_text)]
    if not positions:
        raise ValueError("quote_not_found")
    if len(positions) != 1:
        raise ValueError("ambiguous_normalized_quote")
    start = positions[0]
    end = start + len(normalized_quote)
    first, last = characters[start], characters[end - 1]
    restored = text[first.start:last.end]
    # 禁止把连字的一部分冒充完整原文；回验也覆盖组合字符的保守区间映射。
    if not first.starts_source or not last.ends_source or _normalize_quote(restored) != normalized_quote:
        raise ValueError("quote_position_unrecoverable")
    return LocatedQuote(restored, QuoteMatchKind.NORMALIZED)


def locate_quote(text: str, quote: str) -> LocatedQuote:
    """严格定位优先，否则唯一归一化定位；失败或无法恢复时抛 ValueError。

    严格匹配多次时原文字符串仍然相同，可直接保存；归一化匹配多次则拒收。
    原文指调用方提供的清洗后文本，不是 PDF，也不是模型改写后的字符串。
    """
    if not isinstance(text, str) or not isinstance(quote, str) or not _normalize_quote(quote):
        raise ValueError("empty_or_invalid_quote")
    if quote in text:
        return LocatedQuote(quote, QuoteMatchKind.STRICT)
    return _restore_quote(text, quote)
