"""RAG 答案评估入口测试。"""

from __future__ import annotations

import json

from rag_medical.evaluate_answer import (
    build_evaluation_report,
    load_answer_text,
    output_path_for_answer,
    should_run_judge,
)


def sample_evidence_payload() -> dict:
    return {
        "question": "What are ultrasound findings?",
        "evidence": [
            {"evidence_id": "E1", "text": "Ultrasound showed irregular lesions."},
            {"evidence_id": "E2", "text": "Skin fistula may be observed."},
        ],
    }


def test_load_answer_text_reads_answer_json(tmp_path) -> None:
    path = tmp_path / "demo_answer.json"
    path.write_text(json.dumps({"answer": "结论 [E1]"}, ensure_ascii=False), encoding="utf-8")

    assert load_answer_text(path) == "结论 [E1]"


def test_output_path_for_answer_reuses_answer_prefix() -> None:
    assert output_path_for_answer("data/rag/answers/demo_answer.json").name == "demo_eval.json"


def test_should_run_judge_modes() -> None:
    assert should_run_judge("judge", "pass", False) is True
    assert should_run_judge("all", "pass", False) is True
    assert should_run_judge("rules", "fail", False) is False
    assert should_run_judge("all", "pass", True) is False
    assert should_run_judge("all", "warning", True) is True


def test_build_evaluation_report_merges_rule_status() -> None:
    report = build_evaluation_report(
        answer_path="demo_answer.json",
        evidence_path="demo_evidence.json",
        answer="超声可见低回声病灶 [E1]，也可见皮肤瘘管 [E2]。",
        evidence_payload=sample_evidence_payload(),
        mode="rules",
        judge_result=None,
        min_chars=10,
    )

    assert report["overall_status"] == "pass"
    assert report["rule_eval"]["citation_count"] == 2
    assert report["llm_judge"]["status"] == "skipped"
