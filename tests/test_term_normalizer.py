"""医学术语纯规范化器测试。

输入：项目正式词典和人工构造的 OCR 断裂、黏连、拼写误差及正确文本。
输出：无文件输出；重点验证修复目标、保守性以及数字/剂量保护。
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_medical.terminology.dictionary import load_dictionary
from rag_medical.terminology.normalizer import (
    NormalizerSettings,
    load_normalizer_resources,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/terminology.yaml"


def project_resources():
    dictionary, settings, _ = load_normalizer_resources(CONFIG)
    return dictionary, settings


def test_repairs_fragmented_and_concatenated_medical_terms() -> None:
    dictionary, settings = project_resources()

    fragmented = normalize_text(
        "Drug therapy used ｐｙ ｒａｚｉ ｎａｍｉ ｄｅ.",
        dictionary=dictionary,
        settings=settings,
    )
    disease = normalize_text(
        "Diagnosis: non-l actat i ng ma s tit is.",
        dictionary=dictionary,
        settings=settings,
    )
    concatenated = normalize_text(
        "nonlactatingmastitiscellwallstructure",
        dictionary=dictionary,
        settings=settings,
    )

    assert fragmented.normalized_text == "Drug therapy used pyrazinamide."
    assert disease.normalized_text == "Diagnosis: non-lactating mastitis."
    assert concatenated.normalized_text == "non-lactating mastitis cell wall structure"
    assert all(item.match_type != "strict_fuzzy" for item in fragmented.replacements)


def test_correct_terms_general_english_and_doses_remain_unchanged() -> None:
    dictionary, settings = project_resources()
    samples = [
        "The cell wall structure remained intact.",
        "Non-lactating mastitis was diagnosed.",
        "Rifampicin 0. 45 g qd and isoniazid 0. 3 g qd were prescribed.",
        "The patient was discharged after clinical improvement.",
        "TNF-alpha, DNA and RNA were measured.",
    ]

    for sample in samples:
        result = normalize_text(sample, dictionary=dictionary, settings=settings)
        assert result.normalized_text == sample
        assert result.replacements == ()


def test_strict_fuzzy_repairs_one_edit_but_keeps_unknown_word() -> None:
    dictionary, settings = project_resources()

    typo = normalize_text(
        "Treatment included pyrazinamid.", dictionary=dictionary, settings=settings
    )
    unknown = normalize_text(
        "Treatment included experimentalcompound.", dictionary=dictionary, settings=settings
    )

    assert typo.normalized_text == "Treatment included pyrazinamide."
    assert typo.replacements[0].match_type == "strict_fuzzy"
    assert unknown.normalized_text == "Treatment included experimentalcompound."


def test_category_context_prevents_cross_category_replacement() -> None:
    dictionary, settings = project_resources()

    result = normalize_text(
        "pyrazinamid",
        dictionary=dictionary,
        settings=settings,
        context={"allowed_categories": ["disease"]},
    )

    assert result.normalized_text == "pyrazinamid"
    assert result.replacements == ()


def test_line_end_hyphen_requires_an_actual_hyphen() -> None:
    dictionary, settings = project_resources()

    explicit = normalize_text(
        "granu-\nlomatous mastitis", dictionary=dictionary, settings=settings
    )
    bare_break = normalize_text(
        "granu\nlomatous mastitis", dictionary=dictionary, settings=settings
    )

    assert explicit.normalized_text == "granulomatous mastitis"
    assert bare_break.normalized_text == "granu\nlomatous mastitis"


def test_incomplete_fragment_is_not_guessed() -> None:
    dictionary, settings = project_resources()

    result = normalize_text(
        "The report says idiopathicgranu.", dictionary=dictionary, settings=settings
    )

    assert result.normalized_text == "The report says idiopathicgranu."
    assert result.replacements == ()


def test_normal_english_short_words_do_not_create_false_observations() -> None:
    dictionary, settings = project_resources()
    normal = normalize_text(
        "As a result of treatment, the patient was followed up for six months.",
        dictionary=dictionary,
        settings=settings,
    )
    suspicious = normalize_text(
        "Unknown term: xeno bi o lo gy.", dictionary=dictionary, settings=settings
    )

    assert normal.unresolved_spans == ()
    assert suspicious.unresolved_spans[0].reason == "unknown_fragmented_latin"


def test_ambiguous_fuzzy_candidates_are_not_replaced(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.jsonl"
    aliases_path = tmp_path / "aliases.jsonl"
    records = [
        {
            "concept_id": "A",
            "canonical": "abcdefghi",
            "language": "en",
            "category": "disease",
            "source": "test",
        },
        {
            "concept_id": "B",
            "canonical": "abcdefgki",
            "language": "en",
            "category": "disease",
            "source": "test",
        },
    ]
    canonical_path.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    aliases_path.write_text("", encoding="utf-8")
    dictionary = load_dictionary(canonical_path, aliases_path)

    result = normalize_text(
        "abcdefgji", dictionary=dictionary, settings=NormalizerSettings()
    )

    assert result.normalized_text == "abcdefgji"
    assert result.replacements == ()
    assert result.unresolved_spans[0].reason == "ambiguous_fuzzy_candidates"
