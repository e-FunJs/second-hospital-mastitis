"""Group one question's existing evidence without retrieval or model calls.

Evidence and assertion results are copied, never filtered by role or readiness.
Document identity follows the existing R1 metadata fallback; no identity is
inferred from titles or chunk-ID prefixes. This step does not load source files,
select extra passages, restore article order, or establish study attribution.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from ..coverage_models import CoverageInputPackage, QuestionInputPackage
from .models import DocumentCompletionTask

if TYPE_CHECKING:
    from ..assertion_extract import AssertionExtractionResult


def _document_identity(
    evidence: CoverageInputPackage,
    metadata_by_chunk_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, list[str]]:
    document_id = evidence.document_id.strip()
    metadata_id = ""
    record = metadata_by_chunk_id.get(evidence.chunk_id, {})
    if record.get("chunk_id", evidence.chunk_id) != evidence.chunk_id:
        raise ValueError(f"metadata chunk_id mismatch: {evidence.evidence_id}")
    for key in ("document_id", "pmcid", "pmid", "source_path"):
        value = str(record.get(key) or "").strip()
        if value:
            metadata_id = value
            break
    if document_id and metadata_id and document_id != metadata_id:
        raise ValueError(f"document_id disagrees with metadata: {evidence.evidence_id}")
    if document_id:
        return document_id, []
    if metadata_id:
        return metadata_id, [f"document_id_recovered:{evidence.evidence_id}"]
    return None, [f"document_id_missing:{evidence.evidence_id}"]


def group_evidence_by_document(
    question: QuestionInputPackage,
    *,
    assertion_results: Mapping[str, AssertionExtractionResult] | None = None,
    metadata_by_chunk_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[DocumentCompletionTask]:
    """Keep first-seen document order and each document's evidence order.

    Optional assertion results must belong to this question. Missing results and
    valid empty assertions are normal. Unresolved documents remain one task per
    evidence, with document_id=None rather than a fabricated document identifier.
    """
    if not question.query_id.strip() or not question.question_text.strip():
        raise ValueError("query_id and question_text must be non-empty")
    evidence_ids: set[str] = set()
    for evidence in question.evidence_packages:
        if not evidence.evidence_id.strip() or not evidence.chunk_id.strip():
            raise ValueError("evidence_id and chunk_id must be non-empty")
        if evidence.evidence_id in evidence_ids:
            raise ValueError(f"duplicate evidence_id: {evidence.evidence_id}")
        evidence_ids.add(evidence.evidence_id)
        target = evidence.target_requirement
        if target.query_id != question.query_id or target.question_text != question.question_text:
            raise ValueError(f"evidence belongs to another question: {evidence.evidence_id}")

    assertions = assertion_results if assertion_results is not None else {}
    for evidence_id, result in assertions.items():
        if evidence_id not in evidence_ids:
            raise ValueError(f"unknown assertion evidence_id: {evidence_id}")
        if result.evidence_id != evidence_id:
            raise ValueError(f"assertion evidence_id mismatch: {evidence_id}")

    metadata = metadata_by_chunk_id if metadata_by_chunk_id is not None else {}
    tasks: dict[tuple[str, str], DocumentCompletionTask] = {}
    documents_by_chunk: dict[str, str] = {}
    for evidence in question.evidence_packages:
        document_id, warnings = _document_identity(evidence, metadata)
        if document_id:
            previous = documents_by_chunk.setdefault(evidence.chunk_id, document_id)
            if previous != document_id:
                raise ValueError(f"chunk belongs to different documents: {evidence.chunk_id}")
        key = ("document", document_id) if document_id else ("unresolved", evidence.evidence_id)
        if key not in tasks:
            tasks[key] = DocumentCompletionTask(
                query_id=question.query_id,
                question_text=question.question_text,
                document_id=document_id,
                evidence_packages=[],
                assertion_results={},
                input_warnings=list(question.input_warnings),
            )
        task = tasks[key]
        task.evidence_packages.append(deepcopy(evidence))
        task.input_warnings.extend(warnings)
        if evidence.evidence_id in assertions:
            task.assertion_results[evidence.evidence_id] = deepcopy(assertions[evidence.evidence_id])
    return list(tasks.values())
