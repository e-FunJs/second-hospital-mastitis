"""Train the single-GPU Qwen3-8B LoRA adapter for 6.2 assertion extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import uuid
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate as evaluation  # noqa: E402


TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _canonical_record(item: dict[str, Any]) -> str:
    record = item["record"]
    payload = {
        "question": record["question"],
        "concepts": record["concepts"],
        "evidence_text": record["evidence_text"],
        "assertions": item["gold_assertions"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _roles_in(records: Sequence[dict[str, Any]]) -> set[str]:
    return {
        assertion["role"]
        for item in records
        for assertion in item["gold_assertions"]
    }


def split_records(
    records: Sequence[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(records) < 2 or not 0 < validation_ratio < 1:
        raise ValueError("dataset needs at least two records and 0 < validation_ratio < 1")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        fingerprint = hashlib.sha256(_canonical_record(item).encode("utf-8")).hexdigest()
        grouped.setdefault(fingerprint, []).append(item)

    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    target = max(1, min(len(records) - 1, round(len(records) * validation_ratio)))
    selected: set[int] = set()
    selected_count = 0

    role_totals = Counter(
        assertion["role"]
        for item in records
        for assertion in item["gold_assertions"]
    )
    for role in sorted(role_totals, key=lambda value: (role_totals[value], value)):
        candidates = [
            index
            for index, group in enumerate(groups)
            if index not in selected and role in _roles_in(group)
            and selected_count + len(group) < len(records)
        ]
        if not candidates:
            continue
        index = min(
            candidates,
            key=lambda value: (
                abs(target - selected_count - len(groups[value])),
                len(groups[value]),
                value,
            ),
        )
        selected.add(index)
        selected_count += len(groups[index])

    while selected_count < target:
        candidates = [
            index
            for index, group in enumerate(groups)
            if index not in selected and selected_count + len(group) < len(records)
        ]
        if not candidates:
            break
        index = min(
            candidates,
            key=lambda value: (
                selected_count + len(groups[value]) > target,
                abs(target - selected_count - len(groups[value])),
                len(groups[value]),
                value,
            ),
        )
        selected.add(index)
        selected_count += len(groups[index])

    validation = [item for index in selected for item in groups[index]]
    training = [item for index, group in enumerate(groups) if index not in selected for item in group]
    training.sort(key=lambda item: item["sample_id"])
    validation.sort(key=lambda item: item["sample_id"])
    if not training or not validation:
        raise ValueError("split produced an empty training or validation set")

    validation_roles = _roles_in(validation)
    missing_roles = sorted(set(evaluation.ROLES) - validation_roles)
    if missing_roles:
        warnings.warn(
            f"validation split is missing roles: {', '.join(missing_roles)}",
            stacklevel=2,
        )
    manifest = {
        "seed": seed,
        "validation_ratio_requested": validation_ratio,
        "record_count": len(records),
        "training_count": len(training),
        "validation_count": len(validation),
        "validation_ratio_actual": len(validation) / len(records),
        "duplicate_group_count": sum(len(group) > 1 for group in groups),
        "training_roles": sorted(_roles_in(training)),
        "validation_roles": sorted(validation_roles),
        "missing_validation_roles": missing_roles,
        "training_sample_ids": [item["sample_id"] for item in training],
        "validation_sample_ids": [item["sample_id"] for item in validation],
    }
    return training, validation, manifest


def load_split(
    records: Sequence[dict[str, Any]], manifest_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {item["sample_id"]: item for item in records}
    train_ids = manifest.get("training_sample_ids")
    validation_ids = manifest.get("validation_sample_ids")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise ValueError("split manifest needs training_sample_ids and validation_sample_ids")
    if any(not isinstance(value, str) for value in train_ids + validation_ids):
        raise ValueError("split manifest sample ids must be strings")
    if set(train_ids) & set(validation_ids) or set(train_ids + validation_ids) != set(by_id):
        raise ValueError("split manifest must cover every sample exactly once")
    training = [by_id[sample_id] for sample_id in train_ids]
    validation = [by_id[sample_id] for sample_id in validation_ids]
    if not training or not validation:
        raise ValueError("split manifest contains an empty partition")
    return training, validation, manifest


def _assistant_text(item: dict[str, Any]) -> str:
    return json.dumps(
        {"assertions": item["gold_assertions"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _template_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) for token in value):
        raise TypeError(f"unsupported tokenized template type: {type(value)!r}")
    return value


def _assistant_only_labels(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("chat template did not preserve the assistant-generation prefix")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]


def encode_record(tokenizer: Any, item: dict[str, Any], max_seq_length: int) -> dict[str, Any]:
    messages = evaluation._messages_for_record(item["record"])
    common = {"tokenize": True, "enable_thinking": False}
    prompt_ids = _template_ids(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, **common)
    )
    full_ids = _template_ids(
        tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": _assistant_text(item)}],
            add_generation_prompt=False,
            **common,
        )
    )
    if len(full_ids) > max_seq_length:
        position = int(item["sample_id"][1:])
        raise ValueError(
            f"{item['sample_id']} (dataset position {position}) has {len(full_ids)} tokens; "
            f"max_seq_length={max_seq_length}; truncation is disabled"
        )
    return {
        "sample_id": item["sample_id"],
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": _assistant_only_labels(prompt_ids, full_ids),
    }


class TokenDataset:
    def __init__(self, examples: Sequence[dict[str, Any]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in examples)
        input_ids, attention_mask, labels = [], [], []
        for item in examples:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _check_single_gpu(torch: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_RANK", "-1")) > 0:
        raise RuntimeError("DDP, DeepSpeed, and multi-process training are not supported")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible CUDA device, found {torch.cuda.device_count()}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the visible CUDA device does not support bf16")


def _load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_base_model(model_path: Path) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )


def _attach_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _validation_loss(model: Any, dataset: TokenDataset, collator: CausalCollator) -> float:
    import torch
    from torch.utils.data import DataLoader

    was_training = model.training
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, collate_fn=collator):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_predictions(path: Path, predictions: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )


def _business_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return evaluation.metric_key(metrics)[:4]


def _save_best_adapter(model: Any, output_dir: Path) -> Path:
    adapter_dir = SCRIPT_DIR.parents[1] / "models/lora_adapter/assertion_6_2" / output_dir.name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    destination = adapter_dir / "best_adapter"
    staging = adapter_dir / f".best_adapter.tmp.{uuid.uuid4().hex}"
    try:
        model.save_pretrained(staging, safe_serialization=True, save_embedding_layers=False)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _make_epoch_callback(
    *,
    tokenizer: Any,
    validation_records: Sequence[dict[str, Any]],
    output_dir: Path,
    max_seq_length: int,
    max_new_tokens: int,
    patience: int,
    baseline_metrics: dict[str, Any],
) -> Any:
    from transformers import TrainerCallback

    class EpochBusinessEvaluation(TrainerCallback):
        def __init__(self) -> None:
            self.best_metrics = baseline_metrics
            self.best_business = _business_key(baseline_metrics)
            self.no_business_improvement = 0
            self.seen_epochs: set[int] = set()
            self.history: list[dict[str, Any]] = []

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            model: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            epoch = int(round(state.epoch or 0))
            if epoch <= 0 or epoch in self.seen_epochs:
                return control
            self.seen_epochs.add(epoch)
            if model is None or metrics is None or "eval_loss" not in metrics:
                raise RuntimeError("epoch evaluation did not provide model and eval_loss")

            scored, predictions = evaluation.evaluate_loaded_model(
                model,
                tokenizer,
                validation_records,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                val_loss=float(metrics["eval_loss"]),
            )
            epoch_dir = output_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(exist_ok=False)
            _write_json(epoch_dir / "metrics.json", scored)
            _write_predictions(epoch_dir / "predictions.jsonl", predictions)

            if evaluation.is_better(scored, self.best_metrics):
                adapter_path = _save_best_adapter(model, output_dir)
                self.best_metrics = scored
                _write_json(
                    output_dir / "best.json",
                    {
                        "source": "adapter",
                        "epoch": epoch,
                        "adapter_path": str(adapter_path),
                        "metrics": scored,
                    },
                )

            business = _business_key(scored)
            if business > self.best_business:
                self.best_business = business
                self.no_business_improvement = 0
            else:
                self.no_business_improvement += 1
            self.history.append(
                {
                    "epoch": epoch,
                    "metrics": scored,
                    "no_business_improvement": self.no_business_improvement,
                }
            )
            if self.no_business_improvement >= patience:
                control.should_training_stop = True
            return control

    return EpochBusinessEvaluation()


def _run_smoke(model: Any, dataset: TokenDataset, collator: CausalCollator) -> None:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = collator([dataset[0]])
    batch = {key: value.to(model.device) for key, value in batch.items()}
    model.eval()
    with torch.inference_mode():
        loss = float(model(**batch).loss.detach().cpu())
    print(
        json.dumps(
            {
                "smoke": "PASS",
                "sample_id": dataset[0]["sample_id"],
                "tokens": len(dataset[0]["input_ids"]),
                "loss": loss,
                "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            },
            indent=2,
        )
    )


def _run_training(args: argparse.Namespace) -> None:
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    _check_single_gpu(torch)
    set_seed(args.seed)
    records = evaluation.load_records(args.data_path)
    if args.split_manifest:
        training, validation, manifest = load_split(records, args.split_manifest)
    else:
        training, validation, manifest = split_records(records, args.validation_ratio, args.seed)

    tokenizer = _load_tokenizer(args.model_path)
    training_data = TokenDataset(
        [encode_record(tokenizer, item, args.max_seq_length) for item in training]
    )
    validation_data = TokenDataset(
        [encode_record(tokenizer, item, args.max_seq_length) for item in validation]
    )
    collator = CausalCollator(tokenizer.pad_token_id)
    model = _load_base_model(args.model_path)
    model = _attach_lora(model, args) if args.smoke else model
    if args.smoke:
        _run_smoke(model, training_data, collator)
        return

    output_dir = args.output_dir or Path("outputs/lora_6_2") / (
        args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "split_manifest.json", manifest)

    baseline_loss = _validation_loss(model, validation_data, collator)
    baseline_metrics, baseline_predictions = evaluation.evaluate_loaded_model(
        model,
        tokenizer,
        validation,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        val_loss=baseline_loss,
    )
    baseline_dir = output_dir / "epoch_0"
    baseline_dir.mkdir()
    _write_json(baseline_dir / "metrics.json", baseline_metrics)
    _write_predictions(baseline_dir / "predictions.jsonl", baseline_predictions)
    _write_json(
        output_dir / "best.json",
        {"source": "base_model", "epoch": 0, "adapter_path": None, "metrics": baseline_metrics},
    )

    model = _attach_lora(model, args)
    callback = _make_epoch_callback(
        tokenizer=tokenizer,
        validation_records=validation,
        output_dir=output_dir,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        patience=args.early_stopping_patience,
        baseline_metrics=baseline_metrics,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        do_train=True,
        do_eval=True,
        num_train_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        gradient_checkpointing=True,
        use_cache=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    try:
        result = trainer.train()
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA OOM; retry explicitly with --lora-r 16 and, if needed, a reduced "
            "--target-modules list. The script did not change the experiment automatically."
        ) from error
    _write_json(
        output_dir / "training_summary.json",
        {
            "train_metrics": result.metrics,
            "epochs_evaluated": callback.history,
            "stopped_early": len(callback.history) < args.max_epochs,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "best": json.loads((output_dir / "best.json").read_text(encoding="utf-8"))}, ensure_ascii=False, indent=2))


def _selftest() -> None:
    raw = []
    for index in range(25):
        role = evaluation.ROLES[index % len(evaluation.ROLES)]
        claim = f"Sample {index} received drug {index}."
        raw.append(
            {
                "question": "test",
                "concepts": [
                    {"concept_id": f"DRUG_{index}", "names": [f"drug {index}"], "category": "drug"}
                ],
                "evidence_text": claim,
                "assertions": [
                    {
                        "concept_id": f"DRUG_{index}",
                        "claim_quote": claim,
                        "relation_quote": "ignored",
                        "role": role,
                    }
                ],
            }
        )
    records = evaluation.prepare_records(raw)
    training, validation, manifest = split_records(records, 0.2, 42)
    assert len(training) == 20 and len(validation) == 5
    assert _roles_in(validation) == set(evaluation.ROLES)
    assert not set(manifest["training_sample_ids"]) & set(manifest["validation_sample_ids"])
    labels = _assistant_only_labels([1, 2], [1, 2, 3, 4])
    assert labels == [-100, -100, 3, 4]
    assert all(a["relation_quote"] == "" for a in records[0]["gold_assertions"])
    print("selftest: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run split and masking tests")
    parser.add_argument("--smoke", action="store_true", help="load the model and run one forward batch")
    parser.add_argument("--data-path", type=Path, default=Path("datasets/LoRA/6-2_LoRA_datas.json"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/llm/qwen3-8b"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=list(TARGET_MODULES))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.selftest:
        _selftest()
        return
    if args.max_seq_length != 2048:
        raise ValueError("max_seq_length is fixed at 2048 for this experiment")
    if args.max_epochs <= 0 or args.early_stopping_patience <= 0:
        raise ValueError("epochs and early-stopping patience must be positive")
    _run_training(args)


if __name__ == "__main__":
    main()
