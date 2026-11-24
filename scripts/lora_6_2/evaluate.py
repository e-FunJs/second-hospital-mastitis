"""Evaluate Qwen3 assertion extraction with strict 6.2-compatible parsing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ROLES = (
    "STUDY_INTERVENTION",
    "COMPARATOR",
    "NEGATED_OR_EXCLUDED",
    "NOT_DIRECTLY_RELATED",
    "UNCLEAR",
)
TOP_LEVEL_FIELDS = {"question", "concepts", "evidence_text", "assertions"}
CONCEPT_FIELDS = {"concept_id", "names", "category"}
ASSERTION_FIELDS = {"concept_id", "claim_quote", "relation_quote", "role"}
RELATION_OVERRIDE = (
    "\n本次LoRA协议覆盖上述关系引句规则：relation_quote固定输出空字符串，"
    "不抽取关系片段。"
)


def _f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _counter_scores(gold: Counter[Any], predicted: Counter[Any]) -> dict[str, float | int]:
    true_positive = sum((gold & predicted).values())
    return _f1(
        true_positive,
        sum((predicted - gold).values()),
        sum((gold - predicted).values()),
    )


def _validate_record(record: dict[str, Any], sample_id: str) -> list[dict[str, str]]:
    if not isinstance(record, dict) or set(record) != TOP_LEVEL_FIELDS:
        raise ValueError(f"{sample_id}: invalid top-level fields")
    if not isinstance(record["question"], str) or not isinstance(record["evidence_text"], str):
        raise ValueError(f"{sample_id}: question and evidence_text must be strings")
    if not isinstance(record["concepts"], list) or not isinstance(record["assertions"], list):
        raise ValueError(f"{sample_id}: concepts and assertions must be lists")

    concepts: dict[str, dict[str, Any]] = {}
    for concept in record["concepts"]:
        if not isinstance(concept, dict) or set(concept) != CONCEPT_FIELDS:
            raise ValueError(f"{sample_id}: invalid concept fields")
        concept_id = concept["concept_id"]
        if not isinstance(concept_id, str) or concept_id in concepts:
            raise ValueError(f"{sample_id}: invalid or duplicate concept_id")
        if concept["category"] not in {"drug", "disease"}:
            raise ValueError(f"{sample_id}: invalid concept category for {concept_id}")
        if not isinstance(concept["names"], list) or not concept["names"]:
            raise ValueError(f"{sample_id}: missing concept names for {concept_id}")
        concepts[concept_id] = concept

    assertions: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for position, assertion in enumerate(record["assertions"], 1):
        if not isinstance(assertion, dict) or set(assertion) != ASSERTION_FIELDS:
            raise ValueError(f"{sample_id}: invalid assertion fields at {position}")
        if any(not isinstance(value, str) for value in assertion.values()):
            raise ValueError(f"{sample_id}: assertion fields must be strings at {position}")
        concept_id = assertion["concept_id"]
        claim = assertion["claim_quote"]
        role = assertion["role"]
        if concept_id not in concepts or concepts[concept_id]["category"] != "drug":
            raise ValueError(f"{sample_id}: assertion concept outside drug whitelist: {concept_id}")
        if role not in ROLES:
            raise ValueError(f"{sample_id}: invalid role: {role}")
        if not claim or claim not in record["evidence_text"]:
            raise ValueError(f"{sample_id}: claim_quote is not an exact evidence substring")
        identity = (concept_id, claim, role)
        if identity in seen:
            raise ValueError(f"{sample_id}: duplicate gold assertion: {identity}")
        seen.add(identity)
        assertions.append(
            {
                "concept_id": concept_id,
                "claim_quote": claim,
                "relation_quote": "",
                "role": role,
            }
        )
    return assertions


def prepare_records(raw_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for index, record in enumerate(raw_records, 1):
        sample_id = f"S{index:04d}"
        prepared.append(
            {
                "sample_id": sample_id,
                "record": record,
                "gold_assertions": _validate_record(record, sample_id),
            }
        )
    return prepared


def load_records(data_path: Path, split_manifest: Path | None = None) -> list[dict[str, Any]]:
    raw_records = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError("dataset top level must be a list")
    prepared = prepare_records(raw_records)
    if split_manifest is None:
        return prepared

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    validation_ids = manifest.get("validation_sample_ids")
    if not isinstance(validation_ids, list) or any(not isinstance(x, str) for x in validation_ids):
        raise ValueError("split manifest must contain validation_sample_ids: list[str]")
    selected = {item["sample_id"]: item for item in prepared}
    missing = sorted(set(validation_ids) - selected.keys())
    if missing:
        raise ValueError(f"unknown validation sample ids: {missing}")
    return [selected[sample_id] for sample_id in validation_ids]


def _prediction_assertions(prediction: dict[str, Any]) -> list[dict[str, str]]:
    assertions = prediction.get("assertions", []) if prediction.get("accepted") else []
    if not isinstance(assertions, list):
        raise ValueError("prediction assertions must be a list")
    return assertions


def score_predictions(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    *,
    val_loss: float | None = None,
) -> dict[str, Any]:
    prediction_map: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        sample_id = prediction.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in prediction_map:
            raise ValueError("prediction sample_id must be unique strings")
        prediction_map[sample_id] = prediction

    expected_ids = {item["sample_id"] for item in records}
    if set(prediction_map) != expected_ids:
        raise ValueError("prediction sample ids do not match evaluation records")

    gold_concepts: set[tuple[str, str]] = set()
    predicted_concepts: set[tuple[str, str]] = set()
    gold_by_role = {role: Counter() for role in ROLES}
    predicted_by_role = {role: Counter() for role in ROLES}
    json_valid = accepted = emitted = located = 0
    per_sample = []

    for item in records:
        sample_id = item["sample_id"]
        prediction = prediction_map[sample_id]
        gold = item["gold_assertions"]
        predicted = _prediction_assertions(prediction)
        json_valid += int(bool(prediction.get("json_valid")))
        accepted += int(bool(prediction.get("accepted")))
        emitted += int(prediction.get("emitted_assertion_count", 0))
        located += int(prediction.get("located_claim_count", 0))

        for assertion in gold:
            concept_id, claim, role = (
                assertion["concept_id"],
                assertion["claim_quote"],
                assertion["role"],
            )
            gold_concepts.add((sample_id, concept_id))
            gold_by_role[role][(sample_id, concept_id, claim)] += 1
        for assertion in predicted:
            concept_id, claim, role = (
                assertion["concept_id"],
                assertion["claim_quote"],
                assertion["role"],
            )
            predicted_concepts.add((sample_id, concept_id))
            predicted_by_role[role][(sample_id, concept_id, claim)] += 1

        per_sample.append(
            {
                "sample_id": sample_id,
                "accepted": bool(prediction.get("accepted")),
                "error": prediction.get("error"),
                "gold_count": len(gold),
                "predicted_count": len(predicted),
            }
        )

    concept_scores = _f1(
        len(gold_concepts & predicted_concepts),
        len(predicted_concepts - gold_concepts),
        len(gold_concepts - predicted_concepts),
    )
    role_scores = {
        role: _counter_scores(gold_by_role[role], predicted_by_role[role]) for role in ROLES
    }
    supported_roles = [role for role in ROLES if sum(gold_by_role[role].values())]
    role_macro_f1 = (
        sum(float(role_scores[role]["f1"]) for role in supported_roles) / len(supported_roles)
        if supported_roles
        else 0.0
    )
    return {
        "sample_count": len(records),
        "role_macro_f1": role_macro_f1,
        "concept_micro_f1": concept_scores["f1"],
        "claim_locate_rate": located / emitted if emitted else 1.0,
        "json_valid_rate": json_valid / len(records) if records else 0.0,
        "accepted_rate": accepted / len(records) if records else 0.0,
        "val_loss": val_loss,
        "concept": concept_scores,
        "roles": role_scores,
        "roles_missing_from_gold": [role for role in ROLES if role not in supported_roles],
        "per_sample": per_sample,
    }


def metric_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    loss = metrics.get("val_loss")
    return (
        float(metrics["role_macro_f1"]),
        float(metrics["concept_micro_f1"]),
        float(metrics["claim_locate_rate"]),
        float(metrics["json_valid_rate"]),
        -float(loss) if loss is not None else float("-inf"),
    )


def is_better(candidate: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    return current_best is None or metric_key(candidate) > metric_key(current_best)


def _strict_parse(raw_output: str, record: dict[str, Any]) -> dict[str, Any]:
    from rag_medical.common.expansion.assertion_extract import _parse_output
    from rag_medical.common.expansion.assertion_quote import locate_quote

    result: dict[str, Any] = {
        "raw_output": raw_output,
        "assertions": [],
        "accepted": False,
        "json_valid": False,
        "emitted_assertion_count": 0,
        "located_claim_count": 0,
        "error": None,
    }
    try:
        payload = json.loads(raw_output)
        if not isinstance(payload, dict) or set(payload) != {"assertions"}:
            raise ValueError("invalid_output_schema")
        values = payload["assertions"]
        if not isinstance(values, list):
            raise ValueError("invalid_output_schema")
        result["json_valid"] = True
        result["emitted_assertion_count"] = len(values)
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("claim_quote"), str):
                try:
                    locate_quote(record["evidence_text"], value["claim_quote"])
                    result["located_claim_count"] += 1
                except ValueError:
                    pass
            if not isinstance(value, dict) or value.get("relation_quote") != "":
                raise ValueError("relation_quote_must_be_empty")

        drug_ids = {
            concept["concept_id"]
            for concept in record["concepts"]
            if concept["category"] == "drug"
        }
        parsed = _parse_output(raw_output, record["evidence_text"], drug_ids)
        result["assertions"] = [
            {
                "concept_id": assertion.concept_id,
                "claim_quote": assertion.claim_quote,
                "relation_quote": "",
                "role": assertion.role.value,
            }
            for assertion in parsed
        ]
        result["accepted"] = True
    except Exception as error:  # Keep the model output and exact rejection reason.
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _messages_for_record(record: dict[str, Any]) -> list[dict[str, str]]:
    from rag_medical.common.expansion.assertion_extract import SYSTEM_PROMPT, USER_PROMPT
    from rag_medical.common.step11_generate_answer import build_messages

    concepts = {
        concept["concept_id"]: {
            "names": concept["names"],
            "category": concept["category"],
        }
        for concept in record["concepts"]
    }
    payload = {
        "原始问题": record["question"],
        "完整目标白名单": list(concepts),
        "目标概念及名称、类别": concepts,
        "本次待标注药物白名单": [
            key for key, value in concepts.items() if value["category"] == "drug"
        ],
        "证据原文": record["evidence_text"],
    }
    return build_messages(
        USER_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False)),
        SYSTEM_PROMPT + RELATION_OVERRIDE,
    )


def evaluate_loaded_model(
    model: Any,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    *,
    max_seq_length: int = 2048,
    max_new_tokens: int = 2048,
    val_loss: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from rag_medical.common.step11_generate_answer import extract_model_inputs

    predictions = []
    model.eval()
    for item in records:
        record = item["record"]
        encoded = tokenizer.apply_chat_template(
            _messages_for_record(record),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        input_ids, attention_mask = extract_model_inputs(encoded)
        if input_ids.shape[-1] > max_seq_length:
            raise ValueError(
                f"{item['sample_id']}: input tokens {input_ids.shape[-1]} exceed {max_seq_length}"
            )
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw_output = tokenizer.decode(
            generated[0][input_ids.shape[-1] :], skip_special_tokens=True
        ).strip()
        prediction = _strict_parse(raw_output, record)
        prediction["sample_id"] = item["sample_id"]
        prediction["input_token_count"] = input_ids.shape[-1]
        predictions.append(prediction)
    return score_predictions(records, predictions, val_loss=val_loss), predictions


def load_runtime(model_path: Path, adapter_path: Path | None = None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES=0; exactly one GPU must be visible")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=False, local_files_only=True
        )
    return tokenizer, model


def _selftest() -> None:
    claim = "实验组使用利福平，对照组也使用利福平。"
    raw = [
        {
            "question": "示例问题",
            "concepts": [
                {"concept_id": "DRUG_RFP", "names": ["利福平"], "category": "drug"}
            ],
            "evidence_text": claim,
            "assertions": [
                {
                    "concept_id": "DRUG_RFP",
                    "claim_quote": claim,
                    "relation_quote": "",
                    "role": "STUDY_INTERVENTION",
                },
                {
                    "concept_id": "DRUG_RFP",
                    "claim_quote": claim,
                    "relation_quote": "",
                    "role": "COMPARATOR",
                },
            ],
        }
    ]
    records = prepare_records(raw)
    partial = [
        {
            "sample_id": "S0001",
            "accepted": True,
            "json_valid": True,
            "emitted_assertion_count": 1,
            "located_claim_count": 1,
            "error": None,
            "assertions": [raw[0]["assertions"][0]],
        }
    ]
    scores = score_predictions(records, partial, val_loss=0.1)
    assert scores["concept_micro_f1"] == 1.0
    assert scores["roles"]["STUDY_INTERVENTION"]["f1"] == 1.0
    assert scores["roles"]["COMPARATOR"]["fn"] == 1
    perfect = partial[0] | {
        "emitted_assertion_count": 2,
        "located_claim_count": 2,
        "assertions": raw[0]["assertions"],
    }
    perfect_scores = score_predictions(records, [perfect], val_loss=0.2)
    assert perfect_scores["role_macro_f1"] == 1.0
    assert is_better(perfect_scores, scores)
    worse_role = perfect_scores | {"role_macro_f1": 0.5, "val_loss": 0.01}
    assert not is_better(worse_role, perfect_scores)
    duplicate = json.loads(json.dumps(raw))
    duplicate[0]["assertions"].append(duplicate[0]["assertions"][0])
    try:
        prepare_records(duplicate)
    except ValueError as error:
        assert "duplicate gold assertion" in str(error)
    else:
        raise AssertionError("duplicate gold assertion was accepted")
    print("selftest: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run metric tests without a model")
    parser.add_argument("--data-path", type=Path, default=Path("datasets/LoRA/6-2_LoRA_datas.json"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/llm/qwen3-8b"))
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--val-loss", type=float)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.selftest:
        _selftest()
        return
    records = load_records(args.data_path, args.split_manifest)
    tokenizer, model = load_runtime(args.model_path, args.adapter_path)
    metrics, predictions = evaluate_loaded_model(
        model,
        tokenizer,
        records,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        val_loss=args.val_loss,
    )
    output_dir = args.output_dir or Path("outputs/lora_6_2") / (
        "eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
