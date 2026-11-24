"""Step 5: assemble citable excerpts, without inference or reclassification."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

from .models import ContentType, DocumentMaterials, DocumentSelection
from .selection import prepare_selection_passages


DEFAULT_ROOT = Path("results/rag/lora/document_concat")


def _seed_blocks(task, chunks, by_chunk):
    """Use verified chunk slices; preserve an unlocatable R1 excerpt as evidence."""
    anchors = defaultdict(list)
    opaque = {}
    for e in task.evidence_packages:
        if e.matched_text != chunks[e.chunk_id].text:
            raise ValueError(f"initial text disagrees with source: {e.evidence_id}")
        ids = list(dict.fromkeys(e.source_chunk_ids or [e.chunk_id]))
        expanded = e.expanded_text or e.matched_text
        # Do not assume that an R1 source ID means that its whole chunk was used.
        located = []
        for cid in ids:
            if cid in chunks and chunks[cid].text and chunks[cid].text in expanded:
                start = expanded.index(chunks[cid].text)
                located.append((start, start + len(chunks[cid].text), cid))
        pos = 0
        for start, end, _ in sorted(located):
            if start < pos or expanded[pos:start].strip():
                break
            pos = end
        else:
            if not expanded[pos:].strip():
                for cid in dict.fromkeys([e.chunk_id, *(c for _, _, c in located)]):
                    anchors[cid].append(e.evidence_id)
                continue
        key = (tuple(ids), expanded)
        block = opaque.setdefault(key, dict(kind="initial_r1", chunk_id=None, section=None,
            source_chunk_ids=ids, start=None, end=None, passage_ids=[], evidence_ids=[], text=expanded))
        block["evidence_ids"].append(e.evidence_id)
        if e.matched_text not in expanded:
            anchors[e.chunk_id].append(e.evidence_id)

    blocks = {cid: dict(kind="initial", chunk_id=cid, section=chunks[cid].section,
                        start=0, end=len(chunks[cid].text), evidence_ids=eids,
                        passage_ids=[p.passage_id for p in by_chunk[cid]], text=chunks[cid].text)
              for cid, eids in anchors.items()}
    return blocks, list(opaque.values())


def _render(blocks):
    def header(b):
        ids = b["passage_ids"]
        ref = b["chunk_id"] or ",".join(b["evidence_ids"])
        span = (f" | {ids[0]}" + (f"..{ids[-1]}" if len(ids) > 1 else "")) if ids else ""
        return f"[{ref}{span} | {b['section'] or '章节未定位'}]"

    return "\n\n".join(f"{header(b)}\n{b['text']}" for b in blocks)


def _overlapping_supplements(chunks, by_chunk, candidates):
    """Only collapse matching boundary sentences in proven overlapping source windows."""
    eligible = {p.passage_id for p in candidates}
    aliases = {}
    for left, right in zip(chunks, chunks[1:]):
        a, b = left.source_metadata, right.source_metadata
        bounds = [m.get(k) for m, k in ((a, "sentence_start"), (b, "sentence_start"),
                                         (a, "sentence_end"), (b, "sentence_end"))]
        if (not left.section or left.section != right.section
                or not all(type(n) is int for n in bounds)
                or not (bounds[0] <= bounds[1] <= bounds[2] <= bounds[3])):
            continue
        before, after = by_chunk[left.chunk_id], by_chunk[right.chunk_id]
        for n in range(min(len(before), len(after)), 0, -1):
            pairs = list(zip(before[-n:], after[:n]))
            if all(x.text == y.text for x, y in pairs):
                for x, y in pairs:
                    if x.passage_id in eligible and y.passage_id in eligible:
                        aliases[y.passage_id] = aliases.get(x.passage_id, x.passage_id)
                break
    return aliases


def build_document_packet(materials: DocumentMaterials, selection: DocumentSelection | None,
                          *, tokenizer, material_token_budget: int,
                          rejection_reason: str | None = None) -> dict:
    """Budget is per document, net of downstream instructions, question and reply.

    Pass a complete step-4 selection, or None with a rejection reason. Never
    interpret partial predictions as a usable document.
    """
    if type(material_token_budget) is not int or material_token_budget <= 0:
        raise ValueError("material_token_budget must be a positive integer")
    if (selection is None) != bool(rejection_reason):
        raise ValueError("supply either a selection or a rejection reason")
    task = materials.task
    chunks = {c.chunk_id: c for c in materials.chunks}
    passages = prepare_selection_passages(materials)
    by_chunk = defaultdict(list)
    for p in passages:
        by_chunk[p.chunk_id].append(p)
    labels = {}
    if selection is not None:
        if selection.task != task or set(selection.passage_types) != {p.passage_id for p in passages}:
            raise ValueError("step-4 selection belongs to another task or passage set")
        labels = {k: ContentType(v) for k, v in selection.passage_types.items()}
        candidates = [p for p in passages if not p.is_initial and labels[p.passage_id] != ContentType.OTHER]
        if candidates != selection.supplements:
            raise ValueError("step-4 supplements disagree with original source or labels")
    else:
        candidates = []
    initial, opaque = _seed_blocks(task, chunks, by_chunk)
    candidates = [p for p in candidates if p.chunk_id not in initial]
    aliases = _overlapping_supplements(materials.chunks, by_chunk, candidates)
    candidates = [p for p in candidates if p.passage_id not in aliases]

    def assemble(keep):
        blocks = list(opaque)
        for cid, chunk in chunks.items():  # step 3 order, not global passage rank
            if cid in initial:
                blocks.append(initial[cid])
                continue
            current = None
            for p in by_chunk[cid]:
                if p.passage_id not in keep:
                    current = None  # never bridge an omitted passage
                    continue
                if current is None:
                    current = dict(kind="supplement", chunk_id=cid, section=chunk.section,
                                   start=p.start, evidence_ids=[], passage_ids=[])
                    blocks.append(current)
                current["end"] = p.end
                current["passage_ids"].append(p.passage_id)
                current["text"] = chunk.text[current["start"]:p.end]
        text = _render(blocks)
        return blocks, text, len(tokenizer.encode(text, add_special_tokens=False))

    keep = {p.passage_id for p in candidates}
    blocks, text, count = assemble(keep)
    omitted = []
    if count > material_token_budget:
        keep.clear()
        blocks, text, count = assemble(keep)
        if count > material_token_budget:
            raise ValueError(f"initial_evidence_over_budget:{count}>{material_token_budget}")
        for p in candidates:
            keep.add(p.passage_id)
            trial = assemble(keep)
            if trial[2] <= material_token_budget:
                blocks, text, count = trial
            else:
                keep.remove(p.passage_id)
                omitted.append(dict(passage_id=p.passage_id, chunk_id=p.chunk_id,
                                    start=p.start, end=p.end))
    warnings = list(dict.fromkeys(materials.warnings + (selection.warnings if selection else [])))
    if opaque:
        warnings.append("r1_excerpt_preserved_without_per_chunk_offsets")
    return dict(query_id=task.query_id, question_text=task.question_text, document_id=task.document_id,
                title=task.evidence_packages[0].title, evidence_ids=[e.evidence_id for e in task.evidence_packages],
                selection_status="accepted" if selection else "rejected", rejection_reason=rejection_reason,
                prompt_version=selection.prompt_version if selection else None, ordering_note=materials.ordering_note,
                text=text, blocks=blocks, passage_types={pid: labels[pid].value for b in blocks
                    for pid in b["passage_ids"] if pid in labels}, warnings=warnings,
                token_count=count, material_token_budget=material_token_budget, budget_omissions=omitted,
                overlap_aliases=aliases, initial_sources={e.evidence_id: dict(
                    chunk_id=e.chunk_id, source_chunk_ids=e.source_chunk_ids) for e in task.evidence_packages},
                auxiliary_assertions={k: asdict(v) for k, v in task.assertion_results.items()})


def _component(value):
    if not value or value in {".", ".."}:
        raise ValueError(f"invalid path component: {value!r}")
    return quote(value, safe="-_.")


def save_document_packets(packets, *, classifier_id: str, selection_source: str,
                          root: Path = DEFAULT_ROOT) -> Path:
    """Publish one complete question/run; never overwrite an existing run."""
    packets = list(packets)
    if not packets or len({(p["query_id"], p["question_text"]) for p in packets}) != 1:
        raise ValueError("expected packets for exactly one question")
    if not selection_source:
        raise ValueError("step-4 selection source is required")
    parent = root / _component(packets[0]["query_id"])
    name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f") + "_" + _component(classifier_id)
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / name
    with TemporaryDirectory(dir=parent) as tmp:
        stage = Path(tmp) / "run"
        (stage / "documents").mkdir(parents=True)
        documents = []
        for packet in packets:
            key = packet["document_id"] or ("unresolved_" + packet["evidence_ids"][0])
            filename = "documents/" + _component(key) + ".json"
            with (stage / filename).open("x", encoding="utf-8") as f:
                json.dump(packet, f, ensure_ascii=False, indent=2)
            documents.append(dict(document_id=packet["document_id"], file=filename,
                                  material_token_budget=packet["material_token_budget"],
                                  token_count=packet["token_count"], status=packet["selection_status"]))
        manifest = dict(query_id=packets[0]["query_id"], question_text=packets[0]["question_text"],
                        classifier_id=classifier_id, selection_source=selection_source, documents=documents)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(dest)
    return dest
