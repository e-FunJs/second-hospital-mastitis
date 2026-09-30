"""
用途：加载人工审定的查询别名 JSONL，并验证其基础数据契约。
输入：每行一个术语条目的 JSONL 文件路径。
输出：按 concept_id 分组的 AliasEntry 列表。
不做什么：不做实体匹配、文本归一化、模糊检索或查询生成。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union


_REQUIRED_FIELDS = {
    "concept_id",
    "category",
    "language",
    "term",
    "term_kind",
    "source",
}
_ALLOWED_CATEGORIES = {"disease", "drug"}
_ALLOWED_LANGUAGES = {"zh", "en"}
_ALLOWED_TERM_KINDS = {"canonical", "abbreviation", "synonym"}


@dataclass(frozen=True)
class AliasEntry:
    """词典中的一条可信术语记录。"""

    concept_id: str  # 同一医学概念的稳定标识。
    category: str  # 概念类别，只能是 disease 或 drug。
    language: str  # 术语语言，只能是 zh 或 en。
    term: str  # 保留原样的术语文本。
    term_kind: str  # 术语是规范名、缩写还是同义词。
    source: str  # 术语来源或审定方式。


def _raise_line_error(line_number: int, message: str) -> None:
    """在所有加载错误中统一保留 JSONL 行号，便于直接修复资源文件。"""
    raise ValueError(f"第 {line_number} 行：{message}")


def _parse_record(line: str, line_number: int) -> Mapping[str, Any]:
    """解析单行 JSON，并确保根结构是对象。"""
    if not line.strip():
        _raise_line_error(line_number, "不允许空行")
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        _raise_line_error(line_number, f"JSON 格式错误：{error.msg}")
    if not isinstance(record, dict):
        _raise_line_error(line_number, "每行 JSON 必须是对象")
    return record


def _validate_fields(record: Mapping[str, Any], line_number: int) -> None:
    """验证字段集合和基础字符串类型。"""
    missing_fields = _REQUIRED_FIELDS - set(record)
    extra_fields = set(record) - _REQUIRED_FIELDS
    if missing_fields:
        _raise_line_error(line_number, f"缺少字段：{', '.join(sorted(missing_fields))}")
    if extra_fields:
        _raise_line_error(line_number, f"存在未定义字段：{', '.join(sorted(extra_fields))}")
    non_string_fields = [name for name in _REQUIRED_FIELDS if not isinstance(record[name], str)]
    if non_string_fields:
        _raise_line_error(line_number, f"字段必须是字符串：{', '.join(sorted(non_string_fields))}")


def _validate_values(record: Mapping[str, Any], line_number: int) -> None:
    """验证必填值和三个受限字段的取值范围。"""
    if not record["concept_id"].strip():
        _raise_line_error(line_number, "concept_id 不能为空")
    if not record["term"].strip():
        _raise_line_error(line_number, "term 不能为空")
    if record["category"] not in _ALLOWED_CATEGORIES:
        _raise_line_error(line_number, f"category 非法：{record['category']!r}")
    if record["language"] not in _ALLOWED_LANGUAGES:
        _raise_line_error(line_number, f"language 非法：{record['language']!r}")
    if record["term_kind"] not in _ALLOWED_TERM_KINDS:
        _raise_line_error(line_number, f"term_kind 非法：{record['term_kind']!r}")


def _make_alias_entry(record: Mapping[str, Any], line_number: int) -> AliasEntry:
    """将通过校验的原始记录转换为不可变数据对象。"""
    _validate_fields(record, line_number)
    _validate_values(record, line_number)
    return AliasEntry(**{field_name: record[field_name] for field_name in _REQUIRED_FIELDS})


def load_aliases(path: Union[str, Path]) -> dict[str, list[AliasEntry]]:
    """从 JSONL 加载可信别名，并按 concept_id 分组。"""
    aliases_by_concept: dict[str, list[AliasEntry]] = {}
    concept_by_term: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            entry = _make_alias_entry(_parse_record(line, line_number), line_number)
            existing_concept_id = concept_by_term.get(entry.term)
            if existing_concept_id is not None and existing_concept_id != entry.concept_id:
                _raise_line_error(
                    line_number,
                    f"term {entry.term!r} 已指向 {existing_concept_id!r}，"
                    f"不能再指向 {entry.concept_id!r}",
                )
            concept_by_term[entry.term] = entry.concept_id
            aliases_by_concept.setdefault(entry.concept_id, []).append(entry)
    return aliases_by_concept
