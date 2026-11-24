"""Compare a fresh baseline and selected LoRA adapters on the same blind snapshot."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


TYPES = ("DESIGN", "METHOD", "RESULT", "CONCLUSION", "OTHER")
ROOT = Path.cwd()


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save(p, obj):
    with p.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def adapter_roster(paths, allow_unfinished=False):
    branches = {"baseline": {"adapter": None}}
    counts = set()
    for path in paths:
        path = path.resolve(strict=True)
        run = path.parent.name
        training = ROOT / "outputs/lora_document_content" / run
        manifest = read(training / "manifest.json")
        count, seed = manifest["total_samples"], manifest["args"]["seed"]
        if path.name != "best_adapter" or str(count) != path.parent.parent.name or manifest["args"]["smoke"]:
            raise ValueError(f"adapter directory disagrees with training manifest: {path}")
        if str(seed) in branches:
            raise ValueError(f"duplicate seed: {seed}")
        complete = (training/"summary.json").is_file() and (training/"best.json").is_file()
        if not complete and not allow_unfinished:
            raise ValueError(f"unfinished training: {run}; explicitly allow its saved adapter")
        best = read(training/"best.json") if complete else {}
        if complete and (best["source"] != "adapter" or Path(best["adapter_path"]).parent.name != run):
            raise ValueError(f"best record disagrees with adapter: {run}")
        config = read(path/"adapter_config.json")
        if Path(config["base_model_name_or_path"]).resolve() != (ROOT/"models/llm/qwen3-8b").resolve():
            raise ValueError(f"wrong base model: {path}")
        counts.add(count)
        branches[str(seed)] = dict(adapter=str(path), seed=seed, run=run, best_epoch=best.get("epoch"),
            training_complete=complete, training_prompt_sha256=manifest["prompt_sha256"],
            weights_sha256=sha(path/"adapter_model.safetensors"), config_sha256=sha(path/"adapter_config.json"))
    if len(counts) != 1:
        raise ValueError("supply adapters from one training-set size per evaluation")
    return dict(total_samples=counts.pop(), branches=branches)


def export(xlsx, snapshot, out, confirmations=()):
    from openpyxl import load_workbook
    docs = read(snapshot)
    w = load_workbook(xlsx, read_only=True, data_only=False)
    gold, seen = [], set()
    try:
        assert len(w.worksheets) == len(docs) + 1
        for sheet, doc in zip(w.worksheets[1:], docs, strict=True):
            rows = sheet.values
            first = next(rows)
            assert first[0] == doc["title"] and first[5] == doc["doc"]
            next(rows)
            next(rows)
            for i, (r, src) in enumerate(zip(rows, doc["rows"], strict=True), 4):
                for col in (0, 1, 2, 5, 6):
                    assert r[col] == src[col] or (r[col] is None and src[col] == ""), (sheet.title, i, col)
                assert r[3] is None or r[3] in TYPES, (sheet.title, i, r[3])
                assert src[3:5] == [None, None], "prediction input contains annotations"
                key = (doc["doc"], r[6], r[0])
                assert key not in seen
                seen.add(key)
                gold.append(dict(document=key[0], chunk=key[1], id=key[2], label=r[3],
                                 note=r[4], cell=f"{sheet.title}!D{i}"))
    finally:
        w.close()
    for cell,label in confirmations:
        matches=[r for r in gold if r["cell"]==cell]
        assert len(matches)==1 and label in TYPES and matches[0]["label"] in (None,label)
        matches[0].update(label=label,provenance="user_confirmation")
    out.mkdir(parents=True, exist_ok=False)
    (out/"inputs.json").write_bytes(snapshot.read_bytes())
    save(out/"gold.json", dict(workbook=xlsx.name, workbook_sha256=sha(xlsx), inputs_sha256=sha(snapshot), rows=gold))
    print("export", len(docs), len(seen), Counter(r["label"] for r in gold))


def infer(out, branch):
    # This function reads only the unlabelled snapshot, never the workbook or gold.
    import torch
    from peft import PeftModel
    sys.path.insert(0,str(ROOT/"scripts/lora_document_content"))
    from retest import BASE, sel
    from rag_medical.common.expansion.document_completion.models import DocumentChunk, SelectedPassage
    from rag_medical.common.step11_generate_answer import extract_model_inputs
    from itertools import groupby

    docs = read(out/"inputs.json")
    records = {d["doc"] for d in docs}
    chunks = {}
    with Path("data/articles/processed/chinese/article_chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["document_id"] in records:
                assert r["chunk_id"] not in chunks
                chunks[r["chunk_id"]] = r
    batches = {}
    for d in docs:
        batches[d["doc"]] = []
        for cid, group in groupby(d["rows"], key=lambda row: row[6]):
            rows = list(group)
            r = chunks[cid]
            assert d["doc"] == r["document_id"] and all(v[5]==r["section"] for v in rows)
            c = DocumentChunk(d["doc"], cid, r["section"], r["text"], r, None, [])
            spans = sel._sentence_spans(c.text)
            passages = []
            for (start,end), row in zip(spans, rows, strict=True):
                assert c.text[start:end] == row[2]
                passages.append(SelectedPassage(row[0], cid, start, end, row[2], False, False))
            batches[d["doc"]].append((cid, [p.passage_id for p in passages], sel.build_selection_messages(c, passages)))

    dest = out/branch
    info = read(out/"evaluation.json")["branches"][branch]
    if info["adapter"]:
        path = Path(info["adapter"])
        assert sha(path/"adapter_model.safetensors") == info["weights_sha256"]
        assert sha(path/"adapter_config.json") == info["config_sha256"]
    if dest.exists() and any(dest.iterdir()): raise FileExistsError(dest)
    dest.mkdir(exist_ok=True)
    save(dest/"conditions.json", dict(**info, input_sha256=sha(out/"inputs.json"),
         prompt_sha256=hashlib.sha256(sel.SYSTEM_PROMPT.encode()).hexdigest(), base=str(BASE),
         code_sha256=sha(Path(sel.__file__)), evaluator_sha256=sha(Path(__file__)),
         gpu=torch.cuda.get_device_name(0), quantization="8bit", enable_thinking=False,
         do_sample=False, max_input_tokens=16384, max_new_tokens=2048))
    assert torch.cuda.device_count()==1, "expose one GPU with CUDA_VISIBLE_DEVICES"
    rt = sel.load_selection_runtime(BASE, quantization="8bit")
    assert not getattr(rt.model, "peft_config", None)
    if info["adapter"]:
        m = PeftModel.from_pretrained(rt.model, info["adapter"], adapter_name="content", is_trainable=False)
        assert m.active_adapters == ["content"]
        rt = sel.SelectionRuntime(rt.tokenizer, m.eval())
    for doc, calls in batches.items():
        log, labels = [], {}
        status, error = "accepted", None
        try:
            for cid, ids, messages in calls:
                tok, mask = extract_model_inputs(rt.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_tensors="pt"))
                n = tok.shape[-1]
                limit=getattr(rt.model.config,"max_position_embeddings",None)
                if n>16384 or (isinstance(limit,int) and n+2048>limit):
                    raise ValueError(f"input_budget:{cid}:{n}")
                raw, cnt = sel._generate_selection_text(rt, tok, mask, 2048)
                log.append(dict(chunk=cid, ids=ids, raw=raw, input_tokens=n, output_tokens=cnt))
                try:
                    labels.update({k:v.value for k,v in sel._parse_labels(raw,set(ids)).items()})
                except ValueError as e:
                    status, error, labels = "rejected", str(e), {}
                    break
        except Exception as exc:
            save(dest/(doc+".json"), dict(status="technical_failure", reason=repr(exc), labels={}, calls=log))
            raise
        save(dest/(doc+".json"), dict(status=status, reason=error, labels=labels, calls=log))
        print(branch, doc, status, len(labels), error, flush=True)


def metrics(rows):
    rows = [(g,p) for g,p in rows if g is not None]
    conf = Counter(rows)
    def ratio(a,b): return a/b if b else None
    def stats(tp, actual, predicted):
        return dict(tp=tp, support=actual, predicted=predicted, precision=ratio(tp,predicted),
                    recall=ratio(tp,actual), f1=ratio(2*tp,actual+predicted))
    pc = {t:stats(conf[t,t],sum(g==t for g,p in rows),sum(p==t for g,p in rows)) for t in TYPES}
    useful = set(TYPES)-{"OTHER"}
    keep = stats(sum(g in useful and p in useful for g,p in rows),
                 sum(g in useful for g,p in rows),sum(p in useful for g,p in rows))
    f1 = [v["f1"] for v in pc.values() if v["support"]]
    correct = sum(g==p for g,p in rows)
    return dict(n=len(rows), correct=correct, accuracy=ratio(correct,len(rows)), per_class=pc,
                macro_f1=ratio(sum(f1),len(f1)), four_class_selection=keep,
                confusion={t:{p:conf[t,p] for p in (*TYPES,None)} for t in TYPES})


def score(out):
    inputs=read(out/"inputs.json")
    gold=read(out/"gold.json")
    assert gold["inputs_sha256"]==sha(out/"inputs.json")
    expected=[(d["doc"],r[6],r[0]) for d in inputs for r in d["rows"]]
    assert expected==[(r["document"],r["chunk"],r["id"]) for r in gold["rows"]]
    result, compare = {}, [dict(**r, text=item[2], predictions={}) for r,item in
        zip(gold["rows"],[x for d in inputs for x in d["rows"]],strict=True)]
    conditions=[]
    for b in read(out/"evaluation.json")["branches"]:
        cond=read(out/b/"conditions.json"); assert cond["input_sha256"]==gold["inputs_sha256"]
        assert cond["evaluator_sha256"]==sha(Path(__file__))
        conditions.append(tuple(cond[k] for k in ("prompt_sha256","code_sha256","base","max_input_tokens","max_new_tokens","quantization","enable_thinking","do_sample")))
        docs={d["doc"]:read(out/b/(d["doc"]+".json")) for d in inputs}
        for d in inputs:
            v=docs[d["doc"]]; assert v["status"] in ("accepted","rejected","technical_failure")
            assert set(v["labels"])==({r[0] for r in d["rows"]} if v["status"]=="accepted" else set())
            assert set(v["labels"].values())<=set(TYPES)
        for row in compare:
            row["predictions"][b]=docs[row["document"]]["labels"].get(row["id"])
        def pairs(rows):return [(r["label"],r["predictions"][b]) for r in rows]
        result[b]=dict(protocol={k:dict(status=v["status"],reason=v["reason"]) for k,v in docs.items()},
                       all=metrics(pairs(compare)), delivered=metrics(pairs([r for r in compare if r["predictions"][b] is not None])),
                       by_document={k:metrics(pairs([r for r in compare if r["document"]==k])) for k in docs})
    assert len(set(conditions))==1
    save(out/"comparison.json",compare); save(out/"metrics.json",result)
    lines=["# 第四步：人工盲标与分类器实测","",
           "同一份当前部署提示词；拒收文献的所有已标句段均计入端到端分母。四类保留指标以全部已标句段计算，不表示临床相关性。","",
           "| 分支 | 接受篇数 | 五类正确 / 总数 | Macro-F1 | 四类保留精确率 | 四类保留召回率 |",
           "|---|---:|---:|---:|---:|---:|"]
    for b,v in result.items():
        a=v["all"]; k=a["four_class_selection"]
        fmt=lambda x:"NA" if x is None else f"{x:.1%}"
        lines.append(f"| {b} | {sum(d['status']=='accepted' for d in v['protocol'].values())}/{len(inputs)} | {a['correct']}/{a['n']} | {fmt(a['macro_f1'])} | {fmt(k['precision'])} | {fmt(k['recall'])} |")
    lines += ["","## 逐类召回（正确 / 金标）","","| 分支 | "+" | ".join(TYPES)+" |","|---|"+"---:|"*len(TYPES)]
    for b,v in result.items():
        p=v["all"]["per_class"]
        lines.append("| "+b+" | "+" | ".join(f"{p[t]['tp']}/{p[t]['support']}" for t in TYPES)+" |")
    with (out/"报告.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
    for b,v in result.items():print(b, v["all"]["correct"], v["all"]["n"], v["all"]["accuracy"],v["all"]["macro_f1"])


def run_all(package, adapters, devices, allow_unfinished):
    roster = adapter_roster(adapters, allow_unfinished)
    if not devices:
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[:1]
    if len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        raise ValueError("specify distinct GPU indices")
    dest = ROOT/"results/rag/lora/document_content"/str(roster["total_samples"])/"LoRA_versus_test"/datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    dest.mkdir(parents=True,exist_ok=False)
    for name in ("inputs.json","gold.json"):
        (dest/name).write_bytes((package/name).read_bytes())
    assert read(dest/"gold.json")["inputs_sha256"]==sha(dest/"inputs.json")
    save(dest/"evaluation.json", dict(**roster, package=str(package.resolve()), devices=devices))
    print("results:",dest,flush=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1")
    def run_device(index):
        for b in list(roster["branches"])[index::len(devices)]:
            with (dest/f"{b}.log").open("x", encoding="utf-8") as log:
                subprocess.run([sys.executable,"-u",__file__,"infer",str(dest),b],
                    env=dict(env, CUDA_VISIBLE_DEVICES=devices[index]), check=True,
                    stdout=log, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(run_device, range(len(devices))))
    score(dest)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="action",required=True)
    all_cmd=sub.add_parser("all")
    all_cmd.add_argument("package",type=Path)
    all_cmd.add_argument("--adapter",type=Path,action="append",required=True)
    all_cmd.add_argument("--devices",nargs="+")
    all_cmd.add_argument("--allow-unfinished",action="store_true")
    a=sub.add_parser("export");a.add_argument("xlsx",type=Path);a.add_argument("snapshot",type=Path)
    a.add_argument("--confirm",nargs=2,action="append",default=[],metavar=("SHEET!CELL","LABEL"))
    for name in ("infer","score","export"):
        cmd=a if name=="export" else sub.add_parser(name)
        cmd.add_argument("out",type=Path)
        if name=="infer":cmd.add_argument("branch")
    args=ap.parse_args()
    if args.action=="all": run_all(args.package,args.adapter,args.devices,args.allow_unfinished)
    elif args.action=="export": export(args.xlsx,args.snapshot,args.out,args.confirm)
    elif args.action=="infer": infer(args.out,args.branch)
    else: score(args.out)
