"""Train and validate document-content LoRA; no retrieval or real-task inference.

Validate the base first, then every epoch on the same held-out samples. Best is
ordered by macro F1, passage accuracy, protocol validity, then lower loss.
Protocol failures count every passage as wrong. No early stopping is used.
One rolling checkpoint per seed is committed after validation (including the baseline).
Use --resume with the same seed to restart the next epoch. Successful runs remove
their checkpoint after final exports; interrupted runs retain it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from tqdm import tqdm

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag_medical.common.expansion.document_completion import selection
from rag_medical.common.expansion.document_completion.models import (
    ContentType, DocumentChunk, SelectedPassage,
)

TYPES = tuple(item.value for item in ContentType)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BEST_RULE = ["macro_f1", "accuracy", "protocol_valid_rate", "-validation_loss"]
CHECKPOINT_DIR = ROOT / "models/lora_adapter/document_content_checkpoint"


def checkpoint_path(seed):
    return CHECKPOINT_DIR / f"checkpoint_seed_{seed}.pt"


@contextmanager
def checkpoint_lock(path):
    # OS locks are released on process exit, including an unexpected termination.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError(f"cannot lock checkpoint {path}; another training with this seed may be running") from exc
        try:
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
            yield
        finally:
            stream.seek(0)
            unlock()


def save_checkpoint(path, state):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if state.get("completed"):
        raise ValueError("training already completed; start a new run without --resume")
    return state


def rng_state():
    import numpy as np
    import torch

    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (algorithm, keys.tolist(), position, has_gauss, cached),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch

    algorithm, keys, position, has_gauss, cached = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_resume(state, manifest):
    saved = state["manifest"]
    ignored = {"run_id", "resume"}
    args = lambda value: {k: v for k, v in value["args"].items() if k not in ignored}
    if args(saved) != args(manifest):
        raise ValueError("resume arguments changed; keep seed, epochs and training settings unchanged")
    for key in ("hashes", "effective_epochs", "prompt_version", "prompt_sha256", "versions",
                "best_rule", "quantization", "enable_thinking", "do_sample", "full_split_zero_based",
                "used_training_ids", "used_validation_ids", "total_samples", "max_total_tokens"):
        if saved[key] != manifest[key]:
            raise ValueError(f"resume conditions changed: {key}")
    if not 0 <= state["epoch"] <= manifest["effective_epochs"]:
        raise ValueError("checkpoint epoch outside training schedule")


@contextmanager
def stage_progress(total, description):
    # Reuse one terminal line; redirected logs get plain updates every 30 seconds.
    started = last_log = time.monotonic()
    with tqdm(total=total, desc=description, unit="sample", file=sys.stdout,
              dynamic_ncols=True, ascii=True, leave=False, mininterval=0.5,
              disable=not sys.stdout.isatty()) as bar:
        if bar.disable:
            print(f"{description}: 0/{total}", flush=True)

        def advance(completed, **metrics):
            nonlocal last_log
            bar.set_postfix(metrics, refresh=False)
            bar.update(completed - bar.n)
            now = time.monotonic()
            if bar.disable and (completed == total or now - last_log >= 30):
                elapsed = now - started
                eta = elapsed * (total - completed) / completed if completed else 0
                details = " ".join(f"{key}={value}" for key, value in metrics.items())
                print(f"{description}: {completed}/{total} elapsed={elapsed:.0f}s ETA={eta:.0f}s {details}", flush=True)
                last_log = now

        yield advance


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_records(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("dataset must contain at least two samples")
    for index, record in enumerate(records, 1):
        if (not isinstance(record, dict) or set(record) != {"section_hint", "passages"}
                or not isinstance(record["section_hint"], str)
                or not isinstance(record["passages"], list) or not record["passages"]):
            raise ValueError(f"invalid sample structure at {index}")
        for passage in record["passages"]:
            if (not isinstance(passage, dict) or set(passage) != {"text", "type"}
                    or not isinstance(passage["text"], str) or not passage["text"].strip()
                    or passage["type"] not in TYPES):
                raise ValueError(f"invalid passage at sample {index}")
    return records


def split_records(records, ratio, seed):
    # Keep samples connected by identical passage text on the same side.
    parent, owners = list(range(len(records))), {}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, record in enumerate(records):
        for passage in record["passages"]:
            key = passage["text"].strip()
            if key in owners:
                parent[find(index)] = find(owners[key])
            owners[key] = index
    groups = {}
    for index in range(len(records)):
        groups.setdefault(find(index), []).append(index)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    target, validation = max(1, round(len(records) * ratio)), []
    for group in groups:
        if len(validation) < target and len(validation) + len(group) < len(records):
            validation.extend(group)
    training = sorted(set(range(len(records))) - set(validation))
    validation.sort()
    for name, indices in (("training", training), ("validation", validation)):
        present = {p["type"] for i in indices for p in records[i]["passages"]}
        if present != set(TYPES):
            raise ValueError(f"{name} lacks classes: {sorted(set(TYPES) - present)}")
    return training, validation


def prepare_record(record, index, seed):
    sample_id = f"S{index + 1:05d}"
    text = "\n".join(p["text"] for p in record["passages"])
    chunk = DocumentChunk(sample_id, sample_id, record["section_hint"], text, {}, None, [])
    # IDs are local protocol labels, not source-document identities.
    start_id = 1 + int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:8], 16) % 9000
    offset, passages, gold = 0, [], {}
    for number, item in enumerate(record["passages"], start_id):
        pid = f"P{number:04d}"
        passages.append(SelectedPassage(pid, sample_id, offset, offset + len(item["text"]),
                                        item["text"], False, False))
        offset += len(item["text"]) + 1
        gold[pid] = item["type"]
    return {"sample_id": sample_id, "messages": selection.build_selection_messages(chunk, passages),
            "gold": gold}


def token_ids(value):
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value[0] if value and isinstance(value[0], list) else value


def encode_record(record, tokenizer, max_length, max_new_tokens):
    answer = json.dumps({"labels": [dict(id=pid, type=kind) for pid, kind in record["gold"].items()]},
                        ensure_ascii=False, separators=(",", ":"))
    prompt = token_ids(tokenizer.apply_chat_template(record["messages"], tokenize=True,
                                                   add_generation_prompt=True, enable_thinking=False))
    full = token_ids(tokenizer.apply_chat_template(
        record["messages"] + [{"role": "assistant", "content": answer}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False))
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError(f"chat-template prefix mismatch: {record['sample_id']}")
    if len(full) > max_length or len(full) - len(prompt) > max_new_tokens:
        raise ValueError(f"token budget exceeded (no truncation): {record['sample_id']}")
    return dict(record, prompt_ids=prompt, input_ids=full,
                labels=[-100] * len(prompt) + full[len(prompt):])


def score_predictions(rows):
    confusion = {kind: dict.fromkeys((*TYPES, "REJECTED"), 0) for kind in TYPES}
    errors = Counter()
    valid = exact = 0
    for row in rows:
        if row["error"]:
            errors[row["error"].split(":", 1)[0]] += 1
        else:
            valid += 1
            exact += row["prediction"] == row["gold"]
        for pid, actual in row["gold"].items():
            predicted = "REJECTED" if row["error"] else row["prediction"][pid]
            confusion[actual][predicted] += 1
    per_class = {}
    for kind in TYPES:
        tp = confusion[kind][kind]
        support = sum(confusion[kind].values())
        predicted = sum(confusion[actual][kind] for actual in TYPES)
        per_class[kind] = {"support": support, "correct": tp, "predicted": predicted,
                           "precision": tp / predicted if predicted else 0.0,
                           "recall": tp / support if support else 0.0,
                           "f1": 2 * tp / (support + predicted) if support + predicted else 0.0}
    total = sum(item["support"] for item in per_class.values())
    correct = sum(item["correct"] for item in per_class.values())
    return {"macro_f1": sum(item["f1"] for item in per_class.values()) / len(TYPES),
            "accuracy": correct / total, "correct": correct, "passage_count": total,
            "protocol_valid_rate": valid / len(rows), "valid_samples": valid,
            "sample_count": len(rows), "exact_sample_accuracy": exact / len(rows),
            "protocol_errors": dict(errors), "per_class": per_class, "confusion": confusion}


def metric_key(metrics):
    return (metrics["macro_f1"], metrics["accuracy"], metrics["protocol_valid_rate"],
            -metrics["validation_loss"])


def batch_for(item, device):
    import torch
    ids = torch.tensor([item["input_ids"]], device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([item["labels"]], device=device)}


def evaluate(model, tokenizer, records, max_new_tokens, *, description="Validation"):
    import torch
    was_training, old_cache = model.training, model.config.use_cache
    model.eval()
    model.config.use_cache = True
    rows, weighted_loss, answer_tokens, rejected = [], 0.0, 0, 0
    try:
        with (stage_progress(len(records), description) as advance,
              torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)):
            for completed, item in enumerate(records, 1):
                batch = batch_for(item, model.device)
                count = int((batch["labels"][:, 1:] != -100).sum())
                loss = float(model(**batch).loss)
                if not math.isfinite(loss):
                    raise RuntimeError("nonfinite validation loss")
                weighted_loss += loss * count
                answer_tokens += count
                ids = torch.tensor([item["prompt_ids"]], device=model.device)
                raw, generated_count = selection._generate_selection_text(
                    selection.SelectionRuntime(tokenizer, model), ids, torch.ones_like(ids), max_new_tokens)
                prediction, error = None, None
                try:
                    prediction = {pid: kind.value for pid, kind in
                                  selection._parse_labels(raw, set(item["gold"])).items()}
                except ValueError as exc:
                    error = f"invalid_json:{exc}" if isinstance(exc, json.JSONDecodeError) else str(exc)
                rows.append({"sample_id": item["sample_id"], "gold": item["gold"],
                             "raw_output": raw, "prediction": prediction, "error": error,
                             "input_tokens": len(item["prompt_ids"]), "output_tokens": generated_count})
                rejected += error is not None
                advance(completed, loss=f"{weighted_loss / answer_tokens:.4f}", rejected=rejected)
    finally:
        model.config.use_cache = old_cache
        model.train(was_training)
    metrics = score_predictions(rows)
    metrics["validation_loss"] = weighted_loss / answer_tokens
    return metrics, rows


def smoke_subset(indices, records, lengths, limit):
    # Exercise all labels and the longest available sample; never cross the split.
    chosen, covered = [max(indices, key=lambda i: lengths[i])], set()
    for index in chosen:
        covered.update(p["type"] for p in records[index]["passages"])
    for index in indices:
        kinds = {p["type"] for p in records[index]["passages"]}
        if index not in chosen and (kinds - covered or len(chosen) < limit):
            chosen.append(index)
            covered.update(kinds)
        if len(chosen) >= limit and covered == set(TYPES):
            break
    return sorted(chosen)


def run(args):
    import torch
    from peft import (LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict,
                      prepare_model_for_kbit_training, set_peft_model_state_dict)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler, set_seed

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support required")
    checkpoint_file = checkpoint_path(args.seed)
    checkpoint = load_checkpoint(checkpoint_file) if args.resume else None
    set_seed(args.seed)
    records = read_records(args.data_path)
    train_ids, val_ids = split_records(records, args.validation_ratio, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_record(prepare_record(r, i, args.seed), tokenizer, args.max_seq_length,
                             args.max_new_tokens) for i, r in enumerate(records)]
    lengths = [len(item["input_ids"]) for item in encoded]
    full_split = {"train": train_ids, "validation": val_ids}
    if args.smoke:
        train_ids = smoke_subset(train_ids, records, lengths, 8)
        val_ids = smoke_subset(val_ids, records, lengths, 5)
    requested_id = ("smoke_" if args.smoke else "") + args.run_id if args.run_id else None
    if checkpoint:
        run_id = checkpoint["run_id"]
        if requested_id and requested_id != run_id:
            raise ValueError("--run-id does not match the checkpoint")
    else:
        run_id = requested_id or (("smoke_" if args.smoke else "")
                                  + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name")
    output = ROOT / "outputs/lora_document_content" / run_id
    adapter_dir = ROOT / "models/lora_adapter/document_content" / str(len(records)) / run_id
    if checkpoint:
        if not output.is_dir() or checkpoint["adapter_dir"] != str(adapter_dir.resolve()):
            raise ValueError("original run paths do not match the checkpoint")
    else:
        if output.exists() or adapter_dir.exists():
            raise FileExistsError("run directory already exists; choose another run-id")
        output.mkdir(parents=True)
    tracked = [Path(__file__), Path(selection.__file__), args.data_path]
    tracked += [args.model_path / name for name in ("config.json", "tokenizer_config.json", "tokenizer.json",
                 "chat_template.jinja", "generation_config.json") if (args.model_path / name).is_file()]
    tracked += sorted(path for path in args.model_path.iterdir() if path.is_file()
                      and (path.suffix in {".safetensors", ".bin"} or path.name.endswith(".index.json")))
    frozen = {str(path.resolve()): digest(path) for path in tracked}
    epochs = 2 if args.smoke else args.epochs
    manifest = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "effective_epochs": epochs, "early_stopping": False, "hashes": frozen,
                "prompt_version": selection.PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(selection.SYSTEM_PROMPT.encode()).hexdigest(),
                "best_rule": BEST_RULE, "quantization": "8bit", "enable_thinking": False,
                "versions": {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")},
                "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "baseline_preparation": "same kbit preparation as training, before attaching LoRA",
                "do_sample": False, "full_split_zero_based": full_split,
                "used_training_ids": [encoded[i]["sample_id"] for i in train_ids],
                "used_validation_ids": [encoded[i]["sample_id"] for i in val_ids],
                "total_samples": len(records), "max_total_tokens": max(lengths),
                "scope": "synthetic held-out validation, not real-task evaluation"}
    if checkpoint:
        validate_resume(checkpoint, manifest)
        if json.loads((output / "manifest.json").read_text(encoding="utf-8")) != checkpoint["manifest"]:
            raise ValueError("original run manifest changed")
        manifest = checkpoint["manifest"]
    else:
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"run": str(output), "train": len(train_ids), "validation": len(val_ids),
                      "epochs": epochs, "max_tokens": max(lengths), "resume": args.resume,
                      "checkpoint": str(checkpoint_file), "checkpoint_mode": "resume" if checkpoint else "replace"}),
          flush=True)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    load_seconds = time.monotonic() - started
    # Keep baseline and epoch validation in the same compute precision.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": True})
    model.config.use_cache = False
    validation = [encoded[i] for i in val_ids]
    if checkpoint:
        best, history = checkpoint["best"], checkpoint["history"]
    else:
        best, rows = evaluate(model, tokenizer, validation, args.max_new_tokens, description="Baseline validation")
        history = [{"epoch": 0, **best}]
        write_json(output / "epoch_000.json", {"metrics": best, "predictions": rows})
        print(f"Baseline | F1={best['macro_f1']:.4f} accuracy={best['accuracy']:.4f} "
              f"val_loss={best['validation_loss']:.4f}", flush=True)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, bias="none", target_modules=TARGET_MODULES))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_ids) / args.gradient_accumulation)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=math.ceil(epochs * steps_per_epoch * 0.03),
                              num_training_steps=epochs * steps_per_epoch)
    best_state, best_epoch, optimizer_steps = None, 0, 0
    initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()} if args.smoke else None
    start_epoch = 1
    if checkpoint:
        set_peft_model_state_dict(model, checkpoint["adapter_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_state, best_epoch = checkpoint["best_state"], checkpoint["best_epoch"]
        optimizer_steps, initial = checkpoint["optimizer_steps"], checkpoint["initial"]
        load_seconds = checkpoint["load_seconds"]
        start_epoch = checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng_state"])
        print(f"Resuming after epoch {checkpoint['epoch']}; target total epochs={epochs}", flush=True)
    else:
        checkpoint = {"format_version": 1, "run_id": run_id, "manifest": manifest,
                      "adapter_dir": str(adapter_dir.resolve()), "completed": False}

    def persist(epoch):
        # Save adapter-only weights, not another copy of the 8B base model.
        checkpoint.update(epoch=epoch, adapter_state={k: v.detach().cpu().clone()
                          for k, v in get_peft_model_state_dict(model).items()},
                          optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                          best=best, best_state=best_state, best_epoch=best_epoch,
                          optimizer_steps=optimizer_steps, history=history, initial=initial,
                          load_seconds=load_seconds, rng_state=rng_state())
        save_checkpoint(checkpoint_file, checkpoint)

    if not args.resume:
        persist(0)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = list(train_ids)
        random.Random(args.seed + epoch).shuffle(order)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        with stage_progress(len(order), f"Train {epoch}/{epochs}") as advance:
            for position, index in enumerate(order):
                remaining_group = min(args.gradient_accumulation,
                                      len(order) - position // args.gradient_accumulation * args.gradient_accumulation)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch_for(encoded[index], model.device)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite training loss at epoch {epoch}")
                total_loss += float(loss.detach())
                try:
                    (loss / remaining_group).backward()
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM: epoch={epoch}, sample={encoded[index]['sample_id']}, tokens={lengths[index]}", flush=True)
                    raise
                if (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order):
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                advance(position + 1, loss=f"{total_loss / (position + 1):.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        metrics, rows = evaluate(model, tokenizer, validation, args.max_new_tokens,
                                 description=f"Valid {epoch}/{epochs}")
        improved = metric_key(metrics) > metric_key(best)
        if improved:
            best, best_epoch = metrics, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            model.save_pretrained(adapter_dir / "best_adapter", safe_serialization=True, save_embedding_layers=False)
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "is_new_best": improved, **metrics})
        write_json(output / f"epoch_{epoch:03d}.json", {"metrics": history[-1], "predictions": rows})
        persist(epoch)
        print(f"Epoch {epoch}/{epochs} | train_loss={total_loss / len(order):.4f} "
              f"val_loss={metrics['validation_loss']:.4f} F1={metrics['macro_f1']:.4f} "
              f"accuracy={metrics['accuracy']:.4f} valid={metrics['valid_samples']}/{len(validation)} "
              f"best_epoch={best_epoch}" + (" NEW BEST" if improved else ""), flush=True)
    if initial is not None and not any(not torch.equal(initial[k], v.detach().cpu())
                                       for k, v in get_peft_model_state_dict(model).items()):
        raise RuntimeError("smoke failed: adapter weights did not change")
    best_path = None
    if best_state is not None:
        set_peft_model_state_dict(model, best_state)
        best_path = adapter_dir / "best_adapter"
        model.save_pretrained(best_path, safe_serialization=True, save_embedding_layers=False)
    roundtrip = None
    if args.smoke:
        # Verify serialization even when the base remains best; do not label it best.
        test_path = best_path or adapter_dir / "smoke_roundtrip_adapter"
        if best_path is None:
            model.save_pretrained(test_path, safe_serialization=True, save_embedding_layers=False)
        expected = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        base = model.unload()
        model = PeftModel.from_pretrained(base, test_path, is_trainable=False)
        actual = get_peft_model_state_dict(model)
        if set(expected) != set(actual) or any(not torch.equal(expected[k], v.detach().cpu()) for k, v in actual.items()):
            raise RuntimeError("smoke failed: saved/reloaded adapter differs")
        reloaded_metrics, reloaded_rows = evaluate(model, tokenizer, validation, args.max_new_tokens,
                                                   description="Reload validation")
        reference = json.loads((output / f"epoch_{best_epoch if best_path else epochs:03d}.json").read_text())
        if [r["raw_output"] for r in reference["predictions"]] != [r["raw_output"] for r in reloaded_rows]:
            raise RuntimeError("smoke failed: reloaded generation differs")
        roundtrip = {"weights_equal": True, "raw_outputs_equal": True, "adapter_path": str(test_path)}
        write_json(output / "roundtrip.json", {"check": roundtrip, "metrics": reloaded_metrics})
    if any(digest(path) != sha for path, sha in frozen.items()):
        raise RuntimeError("input/source hashes changed during run")
    write_json(output / "best.json", {"source": "adapter" if best_path else "base_model", "epoch": best_epoch,
                                     "adapter_path": str(best_path) if best_path else None, "metrics": best})
    summary = {"completed_epochs": epochs, "optimizer_steps": optimizer_steps, "load_seconds": load_seconds,
               "best_epoch": best_epoch, "best_adapter": str(best_path) if best_path else None,
               "smoke": args.smoke, "roundtrip": roundtrip, "hashes_unchanged": True, "history": history,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    write_json(output / "summary.json", summary)
    checkpoint_file.unlink()
    print(json.dumps({"status": "PASS", "output": str(output), "best_epoch": best_epoch,
                      "best_adapter": summary["best_adapter"], "roundtrip": roundtrip}), flush=True)


def selftest():
    from contextlib import redirect_stdout
    from io import StringIO

    captured = StringIO()
    with redirect_stdout(captured):
        with stage_progress(2, "Test progress") as advance:
            advance(1, loss="0.5")
            advance(2, loss="0.4")
    assert "0/2" in captured.getvalue() and "2/2" in captured.getvalue()
    assert "loss=0.4" in captured.getvalue() and "\r" not in captured.getvalue()
    records = [{"section_hint": "test", "passages": [{"text": f"{i}-{kind}.", "type": kind} for kind in TYPES]}
               for i in range(20)]
    records[1]["passages"][0] = dict(records[0]["passages"][0])
    train, val = split_records(records, 0.2, 42)
    assert set(train).isdisjoint(val) and set(train + val) == set(range(20))
    assert (0 in train) == (1 in train)
    assert (train, val) == split_records(records, 0.2, 42)
    prepared = prepare_record(records[0], 0, 42)
    payload = json.loads(prepared["messages"][1]["content"].split("\n", 1)[1])
    assert all(set(p) == {"id", "text"} for p in payload["passages"])
    assert set(prepared["gold"]) == {p["id"] for p in payload["passages"]}
    good = {"gold": prepared["gold"], "prediction": prepared["gold"], "error": None}
    bad = dict(good, prediction=None, error="missing_passage_ids:test")
    assert score_predictions([good])["macro_f1"] == 1
    assert score_predictions([bad])["accuracy"] == 0
    assert score_predictions([good, bad])["accuracy"] == 0.5
    assert abs(score_predictions([good, bad])["macro_f1"] - 2 / 3) < 1e-9
    assert token_ids({"input_ids": [[1, 2]]}) == [1, 2]
    base = dict(score_predictions([good]), validation_loss=1.0)
    assert metric_key(dict(base, validation_loss=0.9)) > metric_key(base)
    assert metric_key(dict(base, macro_f1=0.9, validation_loss=0.1)) < metric_key(base)
    for raw in ['{"labels":[]}', '{"labels":[],"labels":[]}', '```json\n{}\n```',
                '{"labels":[{"id":"missing","type":"METHOD"}]}']:
        try:
            selection._parse_labels(raw, set(prepared["gold"]))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid protocol accepted")
    print("selftest: PASS (progress, split, leakage, IDs, metrics, protocol, token normalization, best rule)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "datasets/LoRA/文献段落类别判别LoRA.json")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/llm/qwen3-8b")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true", help="resume this seed's checkpoint at the next epoch")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="8 train / 5 validation samples, 2 real epochs + reload")
    parser.add_argument("--selftest", action="store_true", help="CPU-only checks; no model weights")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if (not 0 < args.validation_ratio < 1 or args.epochs < 1 or args.gradient_accumulation < 1
            or args.learning_rate <= 0 or min(args.lora_r, args.lora_alpha, args.max_seq_length, args.max_new_tokens) < 1):
        parser.error("invalid training settings")
    with checkpoint_lock(checkpoint_path(args.seed)):
        run(args)


if __name__ == "__main__":
    main()
