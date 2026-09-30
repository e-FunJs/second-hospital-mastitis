"""只读医学词典测试。

输入：测试临时目录中的小型 canonical/alias JSONL 与项目正式词典。
输出：无文件输出；验证解析、冲突隔离、类别限制和 n-gram 模糊候选行为。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_medical.terminology.dictionary import compact_key, load_dictionary


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def load_project_dictionary():
    return load_dictionary(
        ROOT / "resources/terminology/canonical_terms.jsonl",
        ROOT / "resources/terminology/aliases.jsonl",
    )


def test_compact_key_only_removes_spacing_and_hyphen_variants() -> None:
    assert compact_key("Ｎｏｎ－lactating mastitis") == "nonlactatingmastitis"
    assert compact_key("0. 3 mg") == "0.3mg"
    assert compact_key("TNF-α") == "tnfα"


def test_loads_project_terms_and_resolves_alias_via_concept_id() -> None:
    dictionary = load_project_dictionary()

    resolved = dictionary.resolve_compact("nonpuerperal mastitis", "en")
    assert resolved is not None
    assert resolved.canonical == "non-puerperal mastitis"
    assert resolved.concept_id == "DIS_NPM"
    assert dictionary.canonical_for("DIS_NPM", "zh").canonical == "非哺乳期乳腺炎"
    assert dictionary.resolve_compact("IGM", "en") is None
    assert dictionary.term_count >= 40


def test_same_compact_key_for_different_concepts_is_isolated(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.jsonl"
    aliases_path = tmp_path / "aliases.jsonl"
    write_jsonl(
        canonical_path,
        [
            {
                "concept_id": "A",
                "canonical": "anti-inflammatory",
                "language": "en",
                "category": "drug",
                "source": "test",
            },
            {
                "concept_id": "B",
                "canonical": "anti inflammatory",
                "language": "en",
                "category": "disease",
                "source": "test",
            },
        ],
    )
    write_jsonl(aliases_path, [])

    dictionary = load_dictionary(canonical_path, aliases_path)

    assert dictionary.resolve_compact("antiinflammatory", "en") is None
    assert len(dictionary.conflicts) == 1
    assert dictionary.conflicts[0].concept_ids == ("A", "B")


def test_rejects_multiple_canonical_forms_for_same_concept_and_language(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.jsonl"
    aliases_path = tmp_path / "aliases.jsonl"
    write_jsonl(
        canonical_path,
        [
            {
                "concept_id": "A",
                "canonical": "first form",
                "language": "en",
                "category": "disease",
                "source": "test",
            },
            {
                "concept_id": "A",
                "canonical": "second form",
                "language": "en",
                "category": "disease",
                "source": "test",
            },
        ],
    )
    write_jsonl(aliases_path, [])

    with pytest.raises(ValueError, match="multiple canonical terms"):
        load_dictionary(canonical_path, aliases_path)


def test_fuzzy_candidates_are_language_category_and_distance_limited() -> None:
    dictionary = load_project_dictionary()

    drug_matches = dictionary.fuzzy_matches(
        "pyrazinamid",
        "en",
        allowed_categories={"drug"},
        max_distance=1,
        min_similarity=0.88,
    )
    wrong_category = dictionary.fuzzy_matches(
        "pyrazinamid",
        "en",
        allowed_categories={"disease"},
        max_distance=1,
        min_similarity=0.88,
    )
    wrong_language = dictionary.fuzzy_matches(
        "pyrazinamid",
        "zh",
        allowed_categories={"drug"},
        max_distance=1,
        min_similarity=0.88,
    )

    assert drug_matches[0].term.canonical == "pyrazinamide"
    assert drug_matches[0].distance == 1
    assert wrong_category == []
    assert wrong_language == []


def test_trie_returns_longer_phrase_as_a_candidate() -> None:
    dictionary = load_project_dictionary()
    compact = "nonlactatingmastitiscellwallstructure"
    first_matches = list(dictionary.iter_compact_matches(compact, "en", 0))

    assert any(term.canonical == "non-lactating mastitis" for _, term in first_matches)
