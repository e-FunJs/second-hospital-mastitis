"""
用途：验证查询别名词典的加载、值域校验和术语冲突检查。
输入：测试临时目录中的小型 JSONL 词典。
输出：pytest 断言结果。
不做什么：不测试实体匹配、查询生成或任何模型调用。
"""

import json
from pathlib import Path
from typing import Any

import pytest

from rag_medical.common.expansion.aliases import load_aliases


def make_record(**overrides: Any) -> dict[str, str]:
    """构造一条默认合法的术语记录。"""
    record = {
        "concept_id": "DIS_GLM",
        "category": "disease",
        "language": "zh",
        "term": "肉芽肿性小叶性乳腺炎",
        "term_kind": "canonical",
        "source": "trusted_manual",
    }
    record.update(overrides)
    return record


def write_records(path: Path, records: list[dict[str, str]]) -> None:
    """将测试记录写成与正式资源相同的 JSONL 格式。"""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_aliases_groups_entries_and_preserves_count(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "aliases.jsonl"
    write_records(
        dictionary_path,
        [
            make_record(),
            make_record(language="en", term="granulomatous lobular mastitis"),
            make_record(concept_id="DRUG_INH", category="drug", term="异烟肼"),
        ],
    )

    aliases = load_aliases(dictionary_path)

    assert set(aliases) == {"DIS_GLM", "DRUG_INH"}
    assert sum(len(entries) for entries in aliases.values()) == 3


def test_invalid_category_reports_line_number(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "aliases.jsonl"
    write_records(dictionary_path, [make_record(), make_record(category="procedure")])

    with pytest.raises(ValueError, match=r"第 2 行.*category"):
        load_aliases(dictionary_path)


def test_invalid_term_kind_is_rejected(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "aliases.jsonl"
    write_records(dictionary_path, [make_record(term_kind="variant")])

    with pytest.raises(ValueError, match="term_kind"):
        load_aliases(dictionary_path)


def test_same_term_cannot_point_to_different_concepts(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "aliases.jsonl"
    write_records(
        dictionary_path,
        [make_record(), make_record(concept_id="DIS_PDM")],
    )

    with pytest.raises(ValueError, match=r"第 2 行.*已指向.*不能再指向"):
        load_aliases(dictionary_path)


def test_empty_term_is_rejected(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "aliases.jsonl"
    write_records(dictionary_path, [make_record(term="")])

    with pytest.raises(ValueError, match=r"第 1 行.*term 不能为空"):
        load_aliases(dictionary_path)


def test_duplicate_term_for_same_concept_is_allowed(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "aliases.jsonl"
    write_records(
        dictionary_path,
        [make_record(source="trusted_manual"), make_record(source="clinical_review")],
    )

    aliases = load_aliases(dictionary_path)

    assert len(aliases["DIS_GLM"]) == 2
    assert [entry.source for entry in aliases["DIS_GLM"]] == [
        "trusted_manual",
        "clinical_review",
    ]
