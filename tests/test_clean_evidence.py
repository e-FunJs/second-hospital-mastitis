"""RAG evidence 技术清洗测试。

测试聚焦于保守清洗、严重乱码拒绝、证据编号稳定和原始字段可追溯，不涉及医学相关性。
"""

from __future__ import annotations

from pathlib import Path

from rag_medical.common.step09_clean_evidence import (
    clean_evidence_payload,
    clean_text,
    should_join_fullwidth_fragments,
)
from rag_medical.terminology.normalizer import load_normalizer_resources


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    """模拟本地 BGE tokenizer 的关键行为，避免单元测试加载 2.2 GB 模型。"""

    pieces = {
        "pyrazi": ["py", "razi"],
        "razi": ["r", "azi"],
        "nami": ["nami"],
        "de": ["de"],
        "pyrazinamide": ["py", "raz", "inami", "de"],
        "py razi nami de": ["py", "r", "azi", "n", "ami", "de"],
        "pyrazi nami de": ["py", "ra", "zi", "n", "ami", "de"],
        "in": ["in"],
        "the": ["the"],
        "study": ["study"],
        "in the study": ["in", "the", "study"],
        "inthestudy": ["in", "the", "stu", "dy"],
    }

    def tokenize(self, text: str) -> list[str]:
        return self.pieces.get(text, [text])


def test_repairs_fragmented_fullwidth_word_but_keeps_normal_phrase() -> None:
    tokenizer = FakeTokenizer()
    broken = clean_text("药物为ｐｙｒａｚｉ ｎａｍｉ ｄｅ。", tokenizer)
    broken_at_start = clean_text("ｐｙ ｒａｚｉ ｎａｍｉ ｄｅ；", tokenizer)
    normal = clean_text("结果来自ｉｎ ｔｈｅ ｓｔｕｄｙ。", tokenizer)
    long_phrase = clean_text("诊断为ｎｏｎ－ｌ ａｃｔａｔ ｉ ｎｇ ｍａ ｓ ｔｉｔ ｉｓ。", tokenizer)
    hyphenated = clean_text("术语为ｉｄ ｉｏｐａｔｈｉ ｃｇ ｒ ａｎｕ－", tokenizer)

    assert "pyrazinamide" in broken.text
    assert broken_at_start.text == "pyrazinamide;"
    assert "fragmented_fullwidth_latin_repaired" in broken.actions
    assert "in the study" in normal.text
    assert "inthestudy" not in normal.text
    assert "lactatingmastitis" not in long_phrase.text
    assert "idiopathicgranu" not in hyphenated.text
    assert "idiopathicgr" not in hyphenated.text
    assert not should_join_fullwidth_fragments(["cell", "wall", "structure"], tokenizer)


def test_nfkc_does_not_globally_remove_dose_spacing() -> None:
    result = clean_text("异烟肼０． ３ｑｄ，利福平０． ４５ｇ。", FakeTokenizer())

    assert "0. 3qd" in result.text
    assert "0. 45g" in result.text


def test_removes_only_explicit_publication_artifacts() -> None:
    text = (
        "文章编号：１００４－４３３７（２０２１）０５－０７３３－０３ "
        "中图分类号：Ｒ６５５．８ 文献标识码：Ａ ■■■■■■ "
        "异烟肼联合利福平治疗后病灶缩小。"
    )
    result = clean_text(text, FakeTokenizer())

    assert "文章编号" not in result.text
    assert "中图分类号" not in result.text
    assert "文献标识码" not in result.text
    assert "■■" not in result.text
    assert "病灶缩小" in result.text
    assert result.reject_reason == ""


def test_rejects_high_repetition_ocr_noise() -> None:
    result = clean_text(("邓马屯了也卫" * 20), FakeTokenizer())

    assert result.reject_reason == "high_repetition_noise"


def test_rejects_reference_list_dominant_evidence() -> None:
    text = " ".join(
        f"[{number}] 作者. 乳腺炎研究[J]. 医学杂志, 20{number:02d}, 12(3): 1-8."
        for number in range(10, 16)
    )
    result = clean_text(text, FakeTokenizer())

    assert result.reject_reason == "reference_list_dominant"


def test_payload_preserves_original_text_and_stable_ids() -> None:
    payload = {
        "question": "治疗效果如何？",
        "evidence_count": 3,
        "evidence": [
            {"evidence_id": "E1", "matched_text": "正文。", "text": "正文。"},
            {
                "evidence_id": "E2",
                "matched_text": "邓马屯了也卫" * 20,
                "text": "邓马屯了也卫" * 20,
            },
            {
                "evidence_id": "E3",
                "matched_text": "疗效为９０％。",
                "text": "疗效为９０％。",
            },
        ],
    }

    cleaned, report = clean_evidence_payload(payload, FakeTokenizer(), "input.json")

    assert [item["evidence_id"] for item in cleaned["evidence"]] == ["E1", "E3"]
    assert cleaned["evidence"][1]["text"] == "疗效为９０％。"
    assert cleaned["evidence"][1]["cleaned_text"] == "疗效为90%。"
    assert report["rejected_evidence"][0]["evidence_id"] == "E2"
    assert report["rejected_evidence"][0]["reason"] == "high_repetition_noise"


def test_dictionary_normalizer_is_optional_and_repairs_terms_when_enabled() -> None:
    dictionary, settings, _ = load_normalizer_resources(ROOT / "configs/terminology.yaml")
    text = "Diagnosis: non-l actat i ng ma s tit is; drug: pyrazinamid."

    baseline = clean_text(text, FakeTokenizer())
    enhanced = clean_text(
        text,
        FakeTokenizer(),
        terminology_dictionary=dictionary,
        terminology_settings=settings,
    )

    assert baseline.text == text
    assert enhanced.text == (
        "Diagnosis: non-lactating mastitis; drug: pyrazinamide."
    )
    assert "terminology_exact_repaired" in enhanced.actions
    assert "terminology_fuzzy_repaired" in enhanced.actions
