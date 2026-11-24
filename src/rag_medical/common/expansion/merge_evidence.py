"""合并 R0 evidence 与机械执行得到的 R2 命中。

用途：按 chunk_id 去重、保留全部查询来源，并分配可复现的 evidence_id。
输入：内存中的 R0 evidence、R2 批次结果和完整查询计划。
输出：全新 evidence 列表及合并统计；不修改任何输入对象。
不做什么：不读写文件、不执行检索、不扩展上下文，也不判断证据覆盖状态。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from .coverage_models import QueryHitProvenance
from .models import RetrievalQuery, RetrievalStage
from .query_execute import ExecutionStatus, QueryExecutionBatchResult, QueryHit, QueryTaskResult


@dataclass(frozen=True)
class MergeStatistics:
    r0_evidence_count: int
    r2_hit_count: int
    r2_unique_chunk_count: int
    r0_r2_overlap_count: int
    r2_internal_duplicate_count: int
    added_evidence_count: int
    merged_evidence_count: int
    zero_hit_task_count: int
    technical_failure_task_count: int


@dataclass(frozen=True)
class EvidenceMergeResult:
    evidence: list[dict[str, Any]]
    statistics: MergeStatistics


@dataclass
class _ChunkGroup:
    record: dict[str, Any]
    query_hits: list[QueryHitProvenance]
    r0_id_number: int | None


_SOURCE_FIELDS = ("document_id", "language", "source_type", "pmcid", "pmid", "doi", "title",
                  "journal", "year", "section", "source_url")
_STAGE_ORDER = {stage: index for index, stage in enumerate(RetrievalStage)}


def _required_text(record: Mapping[str, Any], key: str, location: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {key} must be a non-empty string")
    return value


def _required_int(record: Mapping[str, Any], key: str, location: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}: {key} must be an integer")
    return value


def _optional_float(record: Mapping[str, Any], key: str, location: str) -> float | None:
    value = record.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f"{location}: {key} must be numeric or null")
    return None if value is None else float(value)


def _parse_evidence_id(value: Any, location: str) -> int:
    match = re.fullmatch(r"E([1-9]\d*)", value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"{location}: evidence_id must use E<number> format")
    return int(match.group(1))


def _make_r0_provenance(
    evidence: Mapping[str, Any], queries: Mapping[int, RetrievalQuery], location: str
) -> QueryHitProvenance:
    query_index = _required_int(evidence, "query_index", location)
    if query_index not in queries:
        raise ValueError(f"{location}: query_index {query_index} is absent from query plan")
    planned_query = queries[query_index]
    try:
        retrieval_stage = RetrievalStage(evidence.get("retrieval_stage"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: invalid retrieval_stage") from exc
    if retrieval_stage is not planned_query.retrieval_stage:
        raise ValueError(f"{location}: retrieval_stage disagrees with query plan")
    return QueryHitProvenance(
        retrieval_stage, query_index, planned_query.generation_kind, planned_query.query_text,
        _required_text(evidence, "retrieval_query", location), planned_query.query_language,
        str(evidence.get("retrieval_language") or ""), _required_int(evidence, "rank", location),
        _optional_float(evidence, "score", location),
        _optional_float(evidence, "dense_score", location),
        _optional_float(evidence, "rerank_score", location),
    )


def _validate_task_result(
    result: QueryTaskResult, queries: Mapping[int, RetrievalQuery]
) -> RetrievalQuery:
    task = result.task
    if task.query_index not in queries:
        raise ValueError(f"R2 query_index {task.query_index} is absent from query plan")
    query = queries[task.query_index]
    comparisons = (
        (task.retrieval_stage, query.retrieval_stage, "retrieval_stage"),
        (task.generation_kind, query.generation_kind, "generation_kind"),
        (task.planned_query_text, query.query_text, "planned_query_text"),
        (task.query_language, query.query_language, "query_language"),
    )
    for actual, expected, name in comparisons:
        if actual != expected:
            raise ValueError(f"R2 task {name} disagrees with query plan")
    if (result.status is ExecutionStatus.HITS_FOUND) != bool(result.hits):
        raise ValueError("execution task status disagrees with hits")
    return query


def _make_r2_provenance(
    hit: QueryHit, result: QueryTaskResult, query: RetrievalQuery
) -> QueryHitProvenance:
    task = result.task
    names = (
        "query_index", "retrieval_stage", "generation_kind", "planned_query_text",
        "executed_query_text", "query_language", "retrieval_language",
    )
    for name in names:
        if getattr(hit, name) != getattr(task, name):
            raise ValueError(f"R2 hit {name} disagrees with execution task")
    return QueryHitProvenance(
        hit.retrieval_stage, hit.query_index, hit.generation_kind, query.query_text,
        hit.executed_query_text, query.query_language, hit.retrieval_language, hit.rank,
        hit.score, hit.dense_score, hit.rerank_score,
    )


def _merge_chunk_metadata(
    target: dict[str, Any], candidate: Mapping[str, Any], chunk_id: str
) -> None:
    if _required_text(target, "matched_text", chunk_id) != _required_text(
        candidate, "matched_text", chunk_id
    ):
        raise ValueError(f"chunk {chunk_id}: matched_text conflict")
    for name in _SOURCE_FIELDS:
        existing, incoming = str(target.get(name) or ""), str(candidate.get(name) or "")
        if existing and incoming and existing != incoming:
            raise ValueError(f"chunk {chunk_id}: {name} conflict")
        if not existing and incoming:
            target[name] = incoming


def _ordered_hits(hits: Sequence[QueryHitProvenance]) -> list[QueryHitProvenance]:
    if not hits:
        raise ValueError("merged evidence must contain at least one query hit")
    primary = min(hits, key=lambda hit: (
        _STAGE_ORDER[hit.retrieval_stage], hit.rank, hit.query_index,
        hit.retrieval_language, hit.executed_query_text,
    ))
    remaining = list(hits)
    remaining.remove(primary)
    return [primary, *sorted(remaining, key=lambda hit: (
        _STAGE_ORDER[hit.retrieval_stage], hit.query_index, hit.retrieval_language,
        hit.rank, hit.executed_query_text,
    ))]


def _query_hit_dict(hit: QueryHitProvenance) -> dict[str, Any]:
    record = asdict(hit)
    for name in ("retrieval_stage", "generation_kind"):
        record[name] = record[name].value
    return record


def _apply_primary_fields(record: dict[str, Any], primary: QueryHitProvenance) -> None:
    record.update({
        "rank": primary.rank, "score": primary.score, "dense_score": primary.dense_score,
        "rerank_score": primary.rerank_score, "retrieval_stage": primary.retrieval_stage.value,
        "retrieval_query": primary.executed_query_text, "query_index": primary.query_index,
        "retrieval_language": primary.retrieval_language,
    })


def _validate_primary_fields(evidence: Mapping[str, Any], primary: QueryHitProvenance) -> None:
    expected = {
        "query_index": primary.query_index, "retrieval_stage": primary.retrieval_stage.value,
        "rank": primary.rank, "score": primary.score, "dense_score": primary.dense_score,
        "rerank_score": primary.rerank_score,
        "retrieval_language": primary.retrieval_language,
        "retrieval_query": primary.executed_query_text,
    }
    for name, value in expected.items():
        if evidence.get(name) != value:
            raise ValueError(f"primary query hit disagrees with evidence.{name}")


def _finalize_group(group: _ChunkGroup, evidence_id: str) -> dict[str, Any]:
    record = deepcopy(group.record)
    record.update({"evidence_id": evidence_id, "document_id": record.get("document_id", "")})
    ordered = _ordered_hits(group.query_hits)
    if group.r0_id_number is None:
        chunk_id = _required_text(record, "chunk_id", evidence_id)
        matched_text = _required_text(record, "matched_text", evidence_id)
        parts = (record.get("pmcid"), record.get("year"), record.get("title"),
                 record.get("section"))
        record.update({
            "citation": " | ".join(str(value) for value in parts if value),
            "context_mode": "hit_only", "context_chunk_ids": [chunk_id], "text": matched_text,
        })
    _apply_primary_fields(record, ordered[0])
    record["query_hits"] = [_query_hit_dict(hit) for hit in ordered]
    _validate_primary_fields(record, ordered[0])
    return record


def _collect_r0(
    evidence: Sequence[dict[str, Any]], queries: Mapping[int, RetrievalQuery]
) -> tuple[dict[str, _ChunkGroup], int]:
    rows, seen, maximum = [], set(), 0
    for row_number, source in enumerate(evidence, start=1):
        location, record = f"R0 evidence row {row_number}", deepcopy(source)
        evidence_id = _required_text(record, "evidence_id", location)
        if evidence_id in seen:
            raise ValueError(f"duplicate R0 evidence_id {evidence_id}")
        seen.add(evidence_id)
        number = _parse_evidence_id(evidence_id, location)
        maximum = max(maximum, number)
        record.setdefault("document_id", "")
        rows.append((number, record, _make_r0_provenance(record, queries, location)))
    groups: dict[str, _ChunkGroup] = {}
    for number, record, provenance in sorted(rows, key=lambda row: row[0]):
        chunk_id = _required_text(record, "chunk_id", f"evidence E{number}")
        if chunk_id not in groups:
            groups[chunk_id] = _ChunkGroup(record, [provenance], number)
        else:
            _merge_chunk_metadata(groups[chunk_id].record, record, chunk_id)
            groups[chunk_id].query_hits.append(provenance)
    return groups, maximum


def _collect_r2(batch: QueryExecutionBatchResult, queries: Mapping[int, RetrievalQuery],
                groups: dict[str, _ChunkGroup]) -> tuple[int, set[str], int, int]:
    hit_count, chunk_ids, zero_count, failure_count = 0, set(), 0, 0
    for result in batch.task_results:
        query = _validate_task_result(result, queries)
        zero_count += result.status is ExecutionStatus.ZERO_HITS
        failure_count += result.status is ExecutionStatus.TECHNICAL_FAILURE
        for hit in result.hits:
            hit_count += 1
            provenance, record = _make_r2_provenance(hit, result, query), asdict(hit.chunk)
            record["language"] = record.pop("chunk_language")
            chunk_id = _required_text(record, "chunk_id", "R2 hit")
            chunk_ids.add(chunk_id)
            if chunk_id not in groups:
                groups[chunk_id] = _ChunkGroup(record, [provenance], None)
            else:
                _merge_chunk_metadata(groups[chunk_id].record, record, chunk_id)
                groups[chunk_id].query_hits.append(provenance)
    return hit_count, chunk_ids, zero_count, failure_count


def merge_evidence(
    r0_evidence: list[dict[str, Any]],
    r2_batch_result: QueryExecutionBatchResult,
    retrieval_queries: Sequence[RetrievalQuery],
) -> EvidenceMergeResult:
    """返回按 chunk 去重的新 evidence 集合，不修改任何输入记录。"""

    queries = {query.query_index: query for query in retrieval_queries}
    if len(queries) != len(retrieval_queries):
        raise ValueError("duplicate query_index in query plan")
    groups, max_r0_id = _collect_r0(r0_evidence, queries)
    r0_chunk_ids = set(groups)
    r2_hits, r2_chunk_ids, zero_count, failure_count = _collect_r2(
        r2_batch_result, queries, groups
    )
    new_groups = sorted(
        (group for group in groups.values() if group.r0_id_number is None),
        key=lambda group: (str(group.record.get("chunk_id") or ""),
                           str(group.record.get("document_id") or "")),
    )
    for number, group in enumerate(new_groups, start=max_r0_id + 1):
        group.record["evidence_id"] = f"E{number}"
    merged = [_finalize_group(group, _required_text(group.record, "evidence_id", "merged"))
              for group in groups.values()]
    merged.sort(key=lambda row: _parse_evidence_id(row["evidence_id"], "merged evidence"))
    overlap = len(r0_chunk_ids & r2_chunk_ids)
    statistics = MergeStatistics(
        len(r0_evidence), r2_hits, len(r2_chunk_ids), overlap,
        r2_hits - len(r2_chunk_ids), len(r2_chunk_ids - r0_chunk_ids), len(merged),
        zero_count, failure_count,
    )
    return EvidenceMergeResult(merged, statistics)
