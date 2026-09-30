"""step10 清洗证据到报告 prompt 的测试。

输入：人工构造的 step09 cleaned evidence payload。
输出：测试临时目录中的 prompt.txt；验证只使用清洗字段、保留类比证据和稳定编号。
"""

from __future__ import annotations

import json

import pytest

from rag_medical.common.step10_build_prompt import (
    build_report_prompt,
    main,
    output_path_for_evidence,
    select_evidence_records,
)


def cleaned_payload() -> dict:
    return {
        "question": "利福平、异烟肼和乙胺丁醇联合治疗非哺乳期乳腺炎的疗效如何？",
        "evidence_count": 2,
        "evidence": [
            {
                "evidence_id": "E1",
                "citation": "2021 | 三联疗法治疗非哺乳期乳腺炎 | 结果",
                "retrieval_language": "zh",
                "text": "未经清洗且不应进入prompt的原文",
                "cleaned_matched_text": "异烟肼、利福平和吡嗪酰胺治疗。",
                "cleaned_text": "研究使用异烟肼、利福平和吡嗪酰胺联合治疗并报告疗效。",
            },
            {
                "evidence_id": "E7",
                "citation": "PMC1 | 2023 | Related treatment",
                "retrieval_language": "en",
                "text": "RAW-NOISE",
                "cleaned_matched_text": "Rifampicin was investigated.",
                "cleaned_text": "Rifampicin was investigated.",
            },
        ],
    }


def test_prompt_uses_cleaned_fields_and_preserves_analogous_evidence() -> None:
    prompt = build_report_prompt(cleaned_payload())

    assert "异烟肼、利福平和吡嗪酰胺联合治疗并报告疗效" in prompt
    assert "未经清洗且不应进入prompt的原文" not in prompt
    assert "RAW-NOISE" not in prompt
    assert "[E1]" in prompt and "[E7]" in prompt
    assert "不要因为文献中的疾病亚型、药物组合或治疗条件与问题不完全一致" in prompt
    assert "不得把相近方案的疗效直接表述为目标方案已经得到验证" in prompt
    assert "不能把“方案不完全一致”本身视为噪声" in prompt
    assert "首先直接回答用户问题" in prompt
    assert "不要按 E1、E2、E3 的顺序逐篇摘要" in prompt
    assert "以下三个部分必须出现" in prompt
    assert "仅在至少一条证据明确提供相应信息时选用" in prompt
    assert "不得把甲方案的剂量、疗程或结局移植给乙方案" in prompt
    assert "一句话中的这些信息必须来自同一个 E" in prompt
    assert "单药研究必须明确称为单药研究" in prompt
    assert "只要任一证据已经给出该信息，就不得声称缺失" in prompt
    assert "研究计划和纳入标准不是实际观察结果" in prompt
    assert "该段内直接引用支持结论的证据" in prompt
    assert "不要求使用完所有证据" in prompt
    assert "不得为了增加引用数量再用类比证据重复该内容" in prompt
    assert "找不到就删除，不能凭标题、常识或其他 E 补齐" in prompt


def test_prompt_does_not_duplicate_identical_match_and_context() -> None:
    prompt = build_report_prompt(cleaned_payload())

    assert prompt.count("Rifampicin was investigated.") == 1


def test_duplicate_evidence_ids_are_rejected() -> None:
    payload = cleaned_payload()
    payload["evidence"][1]["evidence_id"] = "E1"

    with pytest.raises(ValueError, match="duplicate evidence_id"):
        build_report_prompt(payload)


def test_evidence_limit_keeps_both_languages_and_stable_ids() -> None:
    payload = cleaned_payload()
    payload["evidence"] = []
    for index in range(1, 13):
        payload["evidence"].append(
            {
                "evidence_id": f"E{index}",
                "citation": f"source-{index}",
                "retrieval_language": "zh" if index <= 6 else "en",
                "cleaned_text": f"cleaned evidence {index}",
            }
        )

    selected = select_evidence_records(payload["evidence"], max_evidence=10)
    prompt = build_report_prompt(payload, max_evidence=10)

    assert [record["evidence_id"] for record in selected] == [
        "E1",
        "E7",
        "E2",
        "E8",
        "E3",
        "E9",
        "E4",
        "E10",
        "E5",
        "E11",
    ]
    assert prompt.count("### [E") == 10
    assert "### [E6]" not in prompt
    assert "### [E7]" in prompt
    assert "### [E12]" not in prompt


def test_non_positive_evidence_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        build_report_prompt(cleaned_payload(), max_evidence=0)


def test_cli_writes_prompt_with_stable_filename(tmp_path) -> None:
    evidence_path = tmp_path / "example_cleaned_evidence.json"
    evidence_path.write_text(
        json.dumps(cleaned_payload(), ensure_ascii=False), encoding="utf-8"
    )

    status = main(["--evidence", str(evidence_path), "--max-evidence", "1"])
    prompt_path = tmp_path / "example_prompt.txt"

    assert status == 0
    assert output_path_for_evidence(evidence_path) == prompt_path
    assert prompt_path.exists()
    prompt = prompt_path.read_text(encoding="utf-8")
    assert "## 用户问题" in prompt
    assert prompt.count("### [E") == 1
