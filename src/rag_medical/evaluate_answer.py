"""
用途：统一运行 RAG 回答评估，可选择规则评估、LLM-as-judge 或二者结合。
输入：*_answer.json 或 *_answer.md，以及对应 *_evidence.json。
输出：默认生成同名前缀的 *_eval.json。
说明：这是回答质量控制入口，用于检查引用是否可靠、是否越过证据给医学建议。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_medical.eval_judge import run_llm_judge
from rag_medical.eval_rules import evaluate_rules, infer_evidence_path, load_json, max_status
from rag_medical.generate_answer import LLMConfig, load_llm_config


# -----------------------------------------------------------------------------
# 统一评估入口
# -----------------------------------------------------------------------------
# 本文件只做编排：读取 answer/evidence，先跑规则评估，再按 mode 决定是否调用
# LLM-as-judge。规则和 judge 的细节分别放在 eval_rules.py / eval_judge.py。


def load_answer_text(answer_path: Path) -> str:
    if answer_path.suffix == ".json":
        payload = load_json(answer_path)
        answer = payload.get("answer")
        if not isinstance(answer, str):
            raise ValueError(f"answer field missing or not string: {answer_path}")
        return answer.strip()
    return answer_path.read_text(encoding="utf-8").strip()


def output_path_for_answer(answer_path: str | Path) -> Path:
    path = Path(answer_path)
    stem = path.stem
    if stem.endswith("_answer"):
        stem = stem[: -len("_answer")]
    return path.with_name(f"{stem}_eval.json")


def should_run_judge(mode: str, rule_status: str, judge_on_warning: bool) -> bool:
    if mode == "judge":
        return True
    if mode == "rules":
        return False
    if judge_on_warning:
        return rule_status in {"warning", "fail"}
    return mode == "all"


def skipped_judge_result(reason: str = "mode did not request judge") -> dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def normalize_judge_status(judge_result: dict[str, Any]) -> str:
    status = str(judge_result.get("status", "warning"))
    return status if status in {"pass", "warning", "fail"} else "warning"


def build_evaluation_report(
    answer_path: str,
    evidence_path: str,
    answer: str,
    evidence_payload: dict[str, Any],
    mode: str,
    judge_result: dict[str, Any] | None,
    min_citations: int = 2,
    min_chars: int = 80,
) -> dict[str, Any]:
    evidence_records = evidence_payload.get("evidence", [])
    if not isinstance(evidence_records, list):
        raise ValueError("evidence payload must contain an evidence list")

    rule_eval = evaluate_rules(
        answer=answer,
        evidence_records=evidence_records,
        min_citations=min_citations,
        min_chars=min_chars,
    )
    llm_judge = judge_result if judge_result is not None else skipped_judge_result()
    statuses = [rule_eval["status"]]
    if llm_judge.get("status") != "skipped":
        statuses.append(normalize_judge_status(llm_judge))

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_path": answer_path,
        "evidence_path": evidence_path,
        "question": evidence_payload.get("question", ""),
        "mode": mode,
        "overall_status": max_status(statuses),
        "rule_eval": rule_eval,
        "llm_judge": llm_judge,
    }


def build_judge_config(config_path: Path, max_new_tokens: int | None) -> LLMConfig:
    llm_config = load_llm_config(config_path)
    if max_new_tokens is None:
        # judge 输出 JSON，默认不需要像正式回答那样长。
        return replace(llm_config, generation=replace(llm_config.generation, max_new_tokens=512))
    return replace(llm_config, generation=replace(llm_config.generation, max_new_tokens=max_new_tokens))


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a generated RAG answer with rules and optional LLM judge.")
    parser.add_argument("--answer", type=Path, required=True, help="Path to *_answer.json or *_answer.md.")
    parser.add_argument("--evidence", type=Path, help="Path to *_evidence.json. Defaults from answer filename.")
    parser.add_argument("--output", type=Path, help="Path to output *_eval.json.")
    parser.add_argument("--mode", choices=["rules", "judge", "all"], default="rules")
    parser.add_argument("--judge-on-warning", action="store_true", help="In all mode, run judge only when rules warn/fail.")
    parser.add_argument("--config", type=Path, default=Path("configs/llm.yaml"))
    parser.add_argument("--model-path", type=Path, help="Override llm.local_model_path in config for judge.")
    parser.add_argument("--max-new-tokens", type=int, help="Override judge generation length.")
    parser.add_argument("--min-citations", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evidence_path = args.evidence or infer_evidence_path(args.answer)
    output_path = args.output or output_path_for_answer(args.answer)

    for path in [args.answer, evidence_path]:
        if not path.exists():
            print(f"input not found: {path}", file=sys.stderr)
            return 2

    answer = load_answer_text(args.answer)
    evidence_payload = load_json(evidence_path)
    evidence_records = evidence_payload.get("evidence", [])
    if not isinstance(evidence_records, list):
        print("evidence payload must contain an evidence list", file=sys.stderr)
        return 2

    # 先跑规则评估，再决定是否启动 LLM；这样 --judge-on-warning 可以节省显存和时间。
    preliminary_rule_eval = evaluate_rules(answer, evidence_records, args.min_citations, args.min_chars)
    judge_result: dict[str, Any] | None = None
    if should_run_judge(args.mode, preliminary_rule_eval["status"], args.judge_on_warning):
        llm_config = build_judge_config(args.config, args.max_new_tokens)
        model_path = args.model_path or Path(llm_config.model_path)
        if not model_path.exists():
            print(f"model path not found: {model_path}", file=sys.stderr)
            return 2
        judge_result = run_llm_judge(
            question=str(evidence_payload.get("question", "")),
            evidence_records=evidence_records,
            answer=answer,
            model_path=model_path,
            llm_config=llm_config,
        )
    elif args.mode == "all" and args.judge_on_warning:
        judge_result = skipped_judge_result("rules passed and --judge-on-warning was set")

    report = build_evaluation_report(
        answer_path=str(args.answer),
        evidence_path=str(evidence_path),
        answer=answer,
        evidence_payload=evidence_payload,
        mode=args.mode,
        judge_result=judge_result,
        min_citations=args.min_citations,
        min_chars=args.min_chars,
    )
    write_report(report, output_path)

    print(f"eval_path={output_path}")
    print(f"overall_status={report['overall_status']}")
    print(f"rule_status={report['rule_eval']['status']}")
    print(f"judge_status={report['llm_judge'].get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
