"""Read explicit strict metadata sources and collect same-document chunks.

No retrieval, PDF access, model calls, semantic selection, or text rewriting.
Only verified section-local positions are used for ordering. A shared source
identifier does not establish that all text belongs to the same study.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..chunk_layer import ChunkLayer, ChunkLayerIndex
from .models import DocumentChunk, DocumentCompletionTask, DocumentMaterials

SourceRecords = dict[str, tuple[Path, dict[str, Any]]]


def _document_id(record: Mapping[str, Any]) -> str:
    for key in ("document_id", "pmcid", "pmid", "source_path"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def _require_strict(chunk_id: str, layers: ChunkLayerIndex) -> None:
    if layers.by_chunk_id.get(chunk_id) is not ChunkLayer.STRICT:
        raise ValueError(f"chunk is not verified strict: {chunk_id}")


def _read_sources(paths: Sequence[Path], layers: ChunkLayerIndex) -> SourceRecords:
    records: SourceRecords = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: metadata must be an object")
                for key in ("chunk_id", "text"):
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError(f"{path}:{line_number}: missing {key}")
                chunk_id = record["chunk_id"]
                if not _document_id(record):
                    raise ValueError(f"{path}:{line_number}: missing document identity")
                if not isinstance(record.get("section", ""), str):
                    raise ValueError(f"{path}:{line_number}: section must be text")
                _require_strict(chunk_id, layers)
                if chunk_id in records:
                    if records[chunk_id][1] != record:
                        raise ValueError(f"conflicting duplicate chunk_id: {chunk_id}")
                    continue
                records[chunk_id] = (path, record)
    return records


def _source_chunk(path: Path, record: dict[str, Any]) -> DocumentChunk:
    return DocumentChunk(
        document_id=_document_id(record), chunk_id=record["chunk_id"],
        section=record.get("section", ""), text=record["text"],
        source_metadata=deepcopy(record), metadata_path=path, seed_evidence_ids=[],
    )


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _order_chunks(chunks: list[DocumentChunk], warnings: list[str]) -> list[DocumentChunk]:
    sections: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        sections.setdefault(chunk.section, []).append(chunk)
    ordered: list[DocumentChunk] = []
    for section, group in sections.items():
        indices = [c.source_metadata.get("chunk_index") for c in group]
        starts = [c.source_metadata.get("sentence_start") for c in group]
        valid = (
            bool(section) and all(_positive_integer(i) for i in indices + starts)
            and len(set(indices)) == len(indices)
        )
        if valid:
            candidate = sorted(group, key=lambda c: c.source_metadata["chunk_index"])
            sentence_starts = [c.source_metadata["sentence_start"] for c in candidate]
            valid = sentence_starts == sorted(sentence_starts)
        if valid:
            ordered.extend(candidate)
        else:
            ordered.extend(group)
            warnings.append(f"section_local_order_unknown:{section or '<missing>'}")
    return ordered


def _validate_tasks(tasks: Sequence[DocumentCompletionTask]) -> None:
    question = (tasks[0].query_id, tasks[0].question_text)
    documents: set[str] = set()
    evidence_ids: set[str] = set()
    for task in tasks:
        if (task.query_id, task.question_text) != question or not all(question):
            raise ValueError("tasks must belong to one non-empty question")
        if not task.evidence_packages:
            raise ValueError("document task must contain initial evidence")
        if task.document_id:
            if task.document_id in documents:
                raise ValueError(f"duplicate document task: {task.document_id}")
            documents.add(task.document_id)
        for evidence in task.evidence_packages:
            if not evidence.evidence_id or evidence.evidence_id in evidence_ids:
                raise ValueError(f"missing or duplicate evidence_id: {evidence.evidence_id}")
            evidence_ids.add(evidence.evidence_id)
            target = evidence.target_requirement
            if (target.query_id, target.question_text) != question:
                raise ValueError(f"evidence belongs to another question: {evidence.evidence_id}")
            if evidence.document_id and evidence.document_id != task.document_id:
                raise ValueError(f"evidence document_id mismatch: {evidence.evidence_id}")


def _collect_task(
    task: DocumentCompletionTask,
    records: SourceRecords,
    same_document: list[tuple[Path, dict[str, Any]]],
    layers: ChunkLayerIndex,
) -> DocumentMaterials:
    warnings = [*task.input_warnings, "full_document_order_unverified",
                "study_attribution_not_verified"]
    chunks = {record["chunk_id"]: _source_chunk(path, record)
              for path, record in same_document}
    if task.document_id is None:
        warnings.append("document_id_missing")
    elif not chunks:
        warnings.append(f"document_not_in_metadata:{task.document_id}")

    for evidence in task.evidence_packages:
        if not evidence.chunk_id or not evidence.matched_text.strip():
            raise ValueError(f"initial evidence missing chunk_id or text: {evidence.evidence_id}")
        source_ids = dict.fromkeys([evidence.chunk_id, *evidence.source_chunk_ids])
        for chunk_id in source_ids:
            _require_strict(chunk_id, layers)
            source = records.get(chunk_id) if task.document_id is not None else None
            if source and _document_id(source[1]) != task.document_id:
                raise ValueError(f"initial/R1 chunk belongs to another document: {chunk_id}")
            if chunk_id == evidence.chunk_id:
                existing = chunks.get(chunk_id)
                if existing and existing.text != evidence.matched_text:
                    raise ValueError(f"initial text conflicts with source: {chunk_id}")
                if existing is None:
                    chunks[chunk_id] = DocumentChunk(
                        document_id=task.document_id, chunk_id=chunk_id,
                        section=evidence.section, text=evidence.matched_text,
                        source_metadata={}, metadata_path=None, seed_evidence_ids=[],
                    )
                    warnings.append(f"initial_chunk_metadata_missing:{chunk_id}")
            elif chunk_id not in chunks:
                # Keep the original R1 text in task; do not guess per-chunk slices.
                warnings.append(f"r1_source_chunk_metadata_missing:{chunk_id}")
            if chunk_id in chunks and evidence.evidence_id not in chunks[chunk_id].seed_evidence_ids:
                chunks[chunk_id].seed_evidence_ids.append(evidence.evidence_id)

    if chunks and all(c.source_metadata.get("source_type") == "pubmed_abstract"
                      for c in chunks.values()):
        warnings.append("abstract_only_source")
    ordered = _order_chunks(list(chunks.values()), warnings)
    return DocumentMaterials(
        task=deepcopy(task), chunks=ordered,
        ordering_note=("Sections retain first-source encounter order; verified chunk_index "
                       "and sentence_start order chunks within sections only. "
                       "Full-article reading order and study attribution are not verified."),
        warnings=warnings,
    )


def load_document_materials(
    tasks: Sequence[DocumentCompletionTask],
    *,
    metadata_paths: Sequence[Path],
    layer_index: ChunkLayerIndex,
) -> list[DocumentMaterials]:
    """Read each explicit metadata path once, then preserve every task and seed.

    Only source-document identity is checked. Raw text is never concatenated,
    cleaned, selected by medical meaning, or truncated. Missing source metadata
    keeps the original hit and full task, with a warning rather than invented text.
    """
    if not tasks:
        return []
    _validate_tasks(tasks)
    records = _read_sources(metadata_paths, layer_index)
    by_document: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, record in records.values():
        by_document.setdefault(_document_id(record), []).append((path, record))
    return [_collect_task(task, records, by_document.get(task.document_id, []), layer_index)
            for task in tasks]
