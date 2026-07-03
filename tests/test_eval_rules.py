"""RAG 答案规则评估测试。"""

from __future__ import annotations

from rag_medical.eval_rules import (
    evaluate_rules,
    extract_citation_ids,
    infer_evidence_path,
    max_status,
)


def sample_evidence() -> list[dict]:
    return [
        {"evidence_id": "E1", "text": "Ultrasound showed irregular hypoechoic lesions."},
        {"evidence_id": "E2", "text": "Skin fistula and lymph nodes can be observed."},
    ]


def test_extract_citation_ids_deduplicates_in_order() -> None:
    answer = "低回声病灶 [E2]，皮肤瘘管 [E1]，再次提到 [E2]。"

    assert extract_citation_ids(answer) == ["E2", "E1"]


def test_evaluate_rules_passes_grounded_answer() -> None:
    answer = "超声常见表现包括低回声病灶 [E1]，也可见皮肤瘘管和淋巴结改变 [E2]。"

    result = evaluate_rules(answer, sample_evidence(), min_citations=2, min_chars=10)

    assert result["status"] == "pass"
    assert result["citation_count"] == 2
    assert result["unknown_citations"] == []


def test_evaluate_rules_fails_unknown_citation_and_think_leak() -> None:
    answer = "<think>分析过程</think> 结论来自 [E9]。"

    result = evaluate_rules(answer, sample_evidence(), min_citations=1, min_chars=10)

    assert result["status"] == "fail"
    assert "E9" in result["unknown_citations"]
    assert any(issue["code"] == "think_leak" for issue in result["issues"])


def test_evaluate_rules_warns_on_low_citation_count_and_risk_terms() -> None:
    answer = "这个方案一定有效，建议患者使用某药继续治疗 [E1]。"

    result = evaluate_rules(answer, sample_evidence(), min_citations=2, min_chars=10)

    assert result["status"] == "fail"
    assert result["citation_count"] == 1
    assert "一定有效" in result["risk_terms"]
    assert any(issue["code"] == "clinical_advice" for issue in result["issues"])


def test_infer_evidence_path_from_answer_path() -> None:
    assert infer_evidence_path("data/rag/answers/demo_answer.json").name == "demo_evidence.json"
    assert infer_evidence_path("data/rag/answers/demo_answer.md").name == "demo_evidence.json"


def test_max_status_uses_fail_over_warning_over_pass() -> None:
    assert max_status(["pass", "warning"]) == "warning"
    assert max_status(["pass", "fail", "warning"]) == "fail"
