"""Replay the frozen real-paper Step 4 workload. Never retrieve, pack or retry."""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag_medical.common.expansion.chunk_layer import load_chunk_layer_index
from rag_medical.common.expansion.coverage_input import load_coverage_input_package
from rag_medical.common.expansion.document_completion import selection as sel
from rag_medical.common.expansion.document_completion.completion import group_evidence_by_document
from rag_medical.common.expansion.document_completion.models import SelectionParameters
from rag_medical.common.expansion.document_completion.sources import load_document_materials
from rag_medical.common.step09_clean_evidence import clean_text
from rag_medical.common.step11_generate_answer import extract_model_inputs

STEP3 = Path("cache_output_test/document_completion_step3_20261021_170325")
ARCHIVE = Path("cache_output_test/document_completion_step4_classification_20261022_234452")
BASE = Path("models/llm/qwen3-8b")
OUT = Path("results/rag/lora/document_content")
RUNS = {"baseline": None, "42": "20261028_125724", "10": "20261029_120807",
        "15": "20261030_093834", "6": "20261030_105202"}
# Developer reference cases, fixed before inference; not independent human gold.
CASES = {"CNKI-05626E68AF2A": {"P0006": "CONCLUSION"},
         "CNKI-26B45C2D723A": {"P0038": "METHOD", "P0047": "RESULT"},
         "CNKI-E56D82AC3E64": {"P0002": "DESIGN", "P0003": "METHOD", "P0005": "RESULT",
                                "P0006": "CONCLUSION", "P0037": "OTHER", "P0038": "OTHER"},
         "CNKI-F014306DBDFA": {"P0028": "OTHER"},
         "CNKI-E17D9698DCBF": {"P0089": "CONCLUSION"}}  # Mixed-in conclusion is still a conclusion.


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def folder(branch):
    run = RUNS[branch]
    return OUT / ("基线(baseline)" if run is None else f"{run[4:8]}-{run[9:13]}-s{branch}")


def identity(branch):
    run = RUNS[branch]
    if run is None:
        return {"adapter": None}
    path = Path("models/lora_adapter/document_content") / run / "best_adapter"
    training = Path("outputs/lora_document_content") / run
    manifest, best = read(training / "manifest.json"), read(training / "best.json")
    config = read(path / "adapter_config.json")
    assert manifest["args"]["seed"] == int(branch) and not manifest["args"]["smoke"]
    assert best["source"] == "adapter" and Path(best["adapter_path"]).resolve() == path.resolve()
    assert Path(config["base_model_name_or_path"]).resolve() == BASE.resolve()
    assert manifest["prompt_sha256"] == hashlib.sha256(sel.SYSTEM_PROMPT.encode()).hexdigest()
    return {"adapter": str(path.resolve()), "seed": int(branch), "run": run, "best_epoch": best["epoch"],
            "weights_sha256": sha(path / "adapter_model.safetensors"), "config_sha256": sha(path / "adapter_config.json")}


def inputs(tokenizer):
    old, frozen = read(ARCHIVE / "inputs.json"), read(STEP3 / "inputs.json")
    for path, expected in frozen["input_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"changed input: {path}")
    layers = load_chunk_layer_index(**{k + "_paths": list(map(Path, v)) for k, v in frozen["layer_paths"].items()})
    q = frozen["question"]
    question = load_coverage_input_package(*(Path(q[k]) for k in ("source_evidence_path",
                                           "source_query_plan_path", "source_context_path")), layers,
                                           text_cleaner=clean_text)
    paths = list(map(Path, frozen["metadata_paths"]))
    metadata = {row["chunk_id"]: row for path in paths for line in path.read_text().splitlines()
                if line.strip() for row in [json.loads(line)]}
    tasks = group_evidence_by_document(question, metadata_by_chunk_id=metadata)
    materials = load_document_materials(tasks, metadata_paths=paths, layer_index=layers)
    assert fingerprint([asdict(m) for m in materials]) == fingerprint(old["materials"])
    params = SelectionParameters(**read(ARCHIVE / "preflight.json")["parameters"])
    groups, batches = {}, []
    for item, previous in zip(materials, old["prepared"], strict=True):
        doc = item.task.document_id
        passages = sel.prepare_selection_passages(item)
        assert previous["document_id"] == doc and [asdict(p) for p in passages] == previous["passages"]
        groups[doc] = passages
        by_chunk = {c.chunk_id: c for c in item.chunks}
        for call in previous["calls"]:
            chunk = by_chunk[call["chunk_id"]]
            group = [p for p in passages if p.chunk_id == chunk.chunk_id]
            messages = sel.build_selection_messages(chunk, group)
            assert messages[1] == call["messages"][1] and [p.passage_id for p in group] == call["passage_ids"]
            ids, _ = extract_model_inputs(tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_tensors="pt"))
            assert ids.shape[-1] <= params.max_input_tokens
            batches.append([doc, chunk.chunk_id, fingerprint(ids.tolist()), ids.shape[-1]])
    assert (len(materials), len(batches), sum(map(len, groups.values()))) == (7, 67, 518)
    info = {"question": question.question_text, "documents": list(groups), "input_token_ids_hash": fingerprint(batches),
            "archived_inputs_sha256": sha(ARCHIVE / "inputs.json"), "prompt_version": sel.PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(sel.SYSTEM_PROMPT.encode()).hexdigest(), "parameters": asdict(params),
            "calls": batches, "source_sha256": sha(Path(sel.__file__)), "driver_sha256": sha(Path(__file__)),
            "model": str(BASE.resolve()), "base_index_sha256": sha(BASE / "model.safetensors.index.json"),
            "quantization": "8bit", "enable_thinking": False, "do_sample": False}
    return materials, groups, params, info


def replay(item, passages, runtime, params):
    started = time.monotonic()
    row = {"document_id": item.task.document_id, "candidates": [asdict(p) for p in passages]}
    try:
        result, = sel.select_document_supplements([item], runtime=runtime, parameters=params)
    except sel.SelectionOutputError as exc:
        row.update(status="rejected", task=asdict(item.task), reason=str(exc), error_type=type(exc).__name__,
                   failed_chunk=exc.chunk_id, raw_output=exc.raw_output, input_token_count=exc.input_token_count,
                   output_token_count=exc.output_token_count, completed_calls=[asdict(c) for c in exc.completed_calls])
    else:
        assert result.task == item.task and set(result.passage_types) == {p.passage_id for p in passages}
        row.update(status="accepted", selection=asdict(result))
    row["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return row


def run(branch):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    model_info = identity(branch)
    target = folder(branch)
    if target.exists():
        raise FileExistsError(target)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True)
    materials, groups, params, info = inputs(tokenizer)
    assert torch.cuda.device_count() == 1
    runtime = sel.load_selection_runtime(BASE, quantization="8bit")
    assert not getattr(runtime.model, "peft_config", None)
    if model_info["adapter"]:
        model = PeftModel.from_pretrained(runtime.model, model_info["adapter"], adapter_name="content", is_trainable=False)
        assert model.active_adapters == ["content"]
        runtime = sel.SelectionRuntime(runtime.tokenizer, model.eval())
    target.mkdir(parents=True, exist_ok=False)
    write(target / "conditions.json", dict(info, **model_info, gpu=torch.cuda.get_device_name(0),
                                          visible_device=os.environ.get("CUDA_VISIBLE_DEVICES")))
    summary = []
    for item in materials:
        doc = item.task.document_id
        try:
            row = replay(item, groups[doc], runtime, params)
        except Exception as exc:
            write(target / f"{doc}.json", {"document_id": doc, "task": asdict(item.task),
                  "status": "technical_failure", "error_type": type(exc).__name__, "reason": str(exc)})
            raise
        write(target / f"{doc}.json", row)
        result = row.get("selection", {})
        summary.append({"document_id": doc, "status": row["status"], "reason": row.get("reason"),
                        "elapsed_seconds": row["elapsed_seconds"], "candidate_count": len(groups[doc]),
                        "supplements": len(result["supplements"]) if result else None,
                        "reserve": len(result["reserve"]) if result else None,
                        "type_counts": Counter(result.get("passage_types", {}).values()) if result else None})
        print(branch, doc, row["status"], row["elapsed_seconds"], row.get("reason", ""), flush=True)
    write(target / "summary.json", {"branch": branch, "model": model_info, "documents": summary})


def compare():
    dst = OUT / "逐句对照.json"
    if dst.exists():
        raise FileExistsError(dst)
    conditions = {b: read(folder(b) / "conditions.json") for b in RUNS}
    for key in ("input_token_ids_hash", "archived_inputs_sha256", "prompt_sha256", "parameters",
                "source_sha256", "driver_sha256", "model", "quantization", "do_sample", "enable_thinking"):
        assert len({fingerprint(c[key]) for c in conditions.values()}) == 1, key
    rows = []
    for doc in conditions["baseline"]["documents"]:
        documents = {b: read(folder(b) / f"{doc}.json") for b in RUNS}
        candidates = documents["baseline"]["candidates"]
        assert all(d["candidates"] == candidates for d in documents.values())
        for p in candidates:
            pid = p["passage_id"]
            values = {}
            for branch, document in documents.items():
                kind = document.get("selection", {}).get("passage_types", {}).get(pid)
                disposition = ("initial" if p["is_initial"] else "unavailable" if kind is None else
                               "reserve" if kind == "OTHER" else "supplement")
                values[branch] = {"document_status": document["status"], "type": kind, "disposition": disposition}
            rows.append(dict(document_id=doc, **p, branches=values, reference=CASES.get(doc, {}).get(pid)))
    assert len(rows) == 518
    write(dst, {"scope": "one real question; references are developer spot-checks, not independent gold",
                "branches": {b: folder(b).name for b in RUNS},
                "summaries": {b: read(folder(b) / "summary.json") for b in RUNS}, "passages": rows})


def check():
    from types import SimpleNamespace
    from unittest.mock import patch
    from transformers import AutoTokenizer

    for b in RUNS:
        identity(b)
        assert not folder(b).exists(), folder(b)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True)
    materials, groups, params, info = inputs(tokenizer)
    item = materials[0]
    passages = groups[item.task.document_id]
    runtime = sel.SelectionRuntime(tokenizer, SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960)))
    replies = [(json.dumps({"labels": [{"id": p.passage_id, "type": "OTHER"} for p in passages
                      if p.chunk_id == c.chunk_id]}), 1) for c in item.chunks if any(p.chunk_id == c.chunk_id for p in passages)]
    with patch.object(sel, "_generate_selection_text", side_effect=replies):
        assert replay(item, passages, runtime, params)["selection"]["supplements"] == []
    with patch.object(sel, "_generate_selection_text", return_value=('{"labels":[]}', 1)):
        rejected = replay(item, passages, runtime, params)
        assert rejected["status"] == "rejected" and "selection" not in rejected and rejected["raw_output"]
    print("CHECK PASS: four adapters, production acceptance/rejection, 7 documents, 67 calls, 518 passages;",
          "input hash:", info["input_token_ids_hash"], flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "compare", *RUNS])
    action = parser.parse_args().action
    if action == "check":
        check()
    elif action == "compare":
        compare()
    else:
        run(action)
