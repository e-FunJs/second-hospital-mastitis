"""LLM-as-judge 评估辅助函数测试。"""

from __future__ import annotations

from rag_medical.eval_judge import build_judge_prompt, parse_judge_response


def sample_evidence() -> list[dict]:
    return [
        {"evidence_id": "E1", "citation": "PMC1 | 2025 | Study", "text": "Ultrasound showed irregular lesions."},
        {"evidence_id": "E2", "citation": "PMC2 | 2026 | Study", "text": "Skin fistula may be present."},
    ]


def test_build_judge_prompt_requires_json_and_evidence_grounding() -> None:
    prompt = build_judge_prompt(
        question="What are common ultrasound findings?",
        evidence_records=sample_evidence(),
        answer="低回声病灶 [E1]。",
    )

    assert "只输出 JSON" in prompt
    assert "faithfulness_score" in prompt
    assert "[E1]" in prompt
    assert "低回声病灶" in prompt


def test_parse_judge_response_reads_fenced_json() -> None:
    raw = """```json
{
  "status": "warning",
  "faithfulness_score": 4,
  "groundedness_score": 3,
  "safety_score": 5,
  "comments": "部分结论引用不够具体",
  "unsupported_claims": ["治疗预测价值不足"]
}
```"""

    parsed = parse_judge_response(raw)

    assert parsed["status"] == "warning"
    assert parsed["faithfulness_score"] == 4
    assert parsed["unsupported_claims"] == ["治疗预测价值不足"]


def test_parse_judge_response_returns_warning_on_invalid_json() -> None:
    parsed = parse_judge_response("无法解析")

    assert parsed["status"] == "warning"
    assert parsed["parse_error"] is True
