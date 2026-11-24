"""加载并组装证据覆盖阶段的输入包。

用途：读取 evidence JSON、查询计划 JSON 和 R1 JSONL，按稳定键连接并清洗文本。
输入：三个现有产物路径、调用方预加载的层索引和调用方配置好的文本清洗函数。
输出：一个 QuestionInputPackage；不产生任何磁盘中间文件。
不做什么：不加载模型或配置、不调用 Qwen、不识别断言，也不判断证据覆盖状态。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping
from rag_medical.common.step09_clean_evidence import TextCleanResult

from .chunk_layer import ChunkLayer, ChunkLayerIndex
from .context_expand import ExpansionMode
from .coverage_models import (
    CoverageInputPackage,
    QueryHitProvenance,
    QuestionInputPackage,
    TargetRequirement,
)
from .models import QueryGenerationKind, RetrievalQuery, RetrievalStage
JsonObject = dict[str, Any]
TextCleaner = Callable[[str], TextCleanResult]
def _read_json_object(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value
def _read_jsonl_objects(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records
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
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}: {key} must be numeric or null")
    return float(value)
def _parse_queries(payload: Mapping[str, Any], path: Path) -> list[RetrievalQuery]:
    values = payload.get("retrieval_queries")
    if not isinstance(values, list):
        raise ValueError(f"{path}: retrieval_queries must be a list")
    queries: list[RetrievalQuery] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: retrieval_queries[{index}] must be an object")
        try:
            queries.append(RetrievalQuery.from_dict(value))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid retrieval_queries[{index}]") from exc
    return queries
def _index_queries(queries: list[RetrievalQuery], path: Path) -> dict[int, RetrievalQuery]:
    indexed: dict[int, RetrievalQuery] = {}
    for query in queries:
        if not isinstance(query.query_index, int) or isinstance(query.query_index, bool) or query.query_index < 0:
            raise ValueError(f"{path}: query_index must be a non-negative integer")
        if query.query_index in indexed:
            raise ValueError(f"{path}: duplicate query_index {query.query_index}")
        indexed[query.query_index] = query
    return indexed


def _make_primary_query_hit(
    evidence: Mapping[str, Any],
    planned_query: RetrievalQuery,
    evidence_stage: RetrievalStage,
    location: str,
) -> QueryHitProvenance:
    """把现有单值字段整理成主命中，供旧数据兼容和一致性校验。"""

    return QueryHitProvenance(
        retrieval_stage=evidence_stage,
        query_index=planned_query.query_index,
        generation_kind=planned_query.generation_kind,
        planned_query_text=planned_query.query_text,
        executed_query_text=_required_text(evidence, "retrieval_query", location),
        query_language=planned_query.query_language,
        retrieval_language=str(evidence.get("retrieval_language") or ""),
        rank=_required_int(evidence, "rank", location),
        score=_optional_float(evidence, "score", location),
        dense_score=_optional_float(evidence, "dense_score", location),
        rerank_score=_optional_float(evidence, "rerank_score", location),
    )


def _required_string(
    record: Mapping[str, Any], key: str, location: str, *, allow_empty: bool = False
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{location}: {key} must be {qualifier}")
    return value


def _parse_query_hit(
    record: Mapping[str, Any],
    queries_by_index: Mapping[int, RetrievalQuery],
    location: str,
) -> QueryHitProvenance:
    query_index = _required_int(record, "query_index", location)
    if query_index not in queries_by_index:
        raise ValueError(f"{location}: query_index {query_index} is absent from query plan")
    planned_query = queries_by_index[query_index]
    try:
        retrieval_stage = RetrievalStage(record.get("retrieval_stage"))
        generation_kind = QueryGenerationKind(record.get("generation_kind"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: invalid enum value") from exc

    hit = QueryHitProvenance(
        retrieval_stage=retrieval_stage,
        query_index=query_index,
        generation_kind=generation_kind,
        planned_query_text=_required_text(record, "planned_query_text", location),
        executed_query_text=_required_text(record, "executed_query_text", location),
        query_language=_required_text(record, "query_language", location),
        retrieval_language=_required_string(
            record, "retrieval_language", location, allow_empty=True
        ),
        rank=_required_int(record, "rank", location),
        score=_optional_float(record, "score", location),
        dense_score=_optional_float(record, "dense_score", location),
        rerank_score=_optional_float(record, "rerank_score", location),
    )
    if hit.retrieval_stage is not planned_query.retrieval_stage:
        raise ValueError(f"{location}: retrieval_stage disagrees with query plan")
    if hit.generation_kind is not planned_query.generation_kind:
        raise ValueError(f"{location}: generation_kind disagrees with query plan")
    if hit.planned_query_text != planned_query.query_text:
        raise ValueError(f"{location}: planned_query_text disagrees with query plan")
    if hit.query_language != planned_query.query_language:
        raise ValueError(f"{location}: query_language disagrees with query plan")
    return hit


def _load_query_hits(
    evidence: Mapping[str, Any],
    queries_by_index: Mapping[int, RetrievalQuery],
    primary_hit: QueryHitProvenance,
    location: str,
) -> list[QueryHitProvenance]:
    values = evidence.get("query_hits")
    if values is None:
        return [primary_hit]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{location}: query_hits must be a non-empty list")

    query_hits: list[QueryHitProvenance] = []
    for index, value in enumerate(values):
        hit_location = f"{location}: query_hits[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{hit_location} must be an object")
        query_hits.append(_parse_query_hit(value, queries_by_index, hit_location))
    if query_hits[0] != primary_hit:
        raise ValueError(f"{location}: query_hits[0] disagrees with primary fields")
    return query_hits
def _make_target(
    query_id: str, question_text: str, queries: list[RetrievalQuery], path: Path
) -> TargetRequirement:
    originals = [
        query
        for query in queries
        if query.query_index == 0
        and query.generation_kind is QueryGenerationKind.ORIGINAL
    ]
    if len(originals) != 1:
        raise ValueError(f"{path}: exactly one ORIGINAL query with query_index 0 is required")
    unresolved = ["relation", "conditions"]
    if not originals[0].concept_ids:
        unresolved.append("target_entities")
    return TargetRequirement(
        query_id=query_id,
        question_text=question_text,
        required_concept_ids=list(originals[0].concept_ids),
        relation_text=None,
        condition_texts=[],
        unresolved_items=unresolved,
    )
def _index_records(
    records: list[JsonObject], *, id_key: str, source: Path
) -> dict[str, JsonObject]:
    indexed: dict[str, JsonObject] = {}
    for row_number, record in enumerate(records, start=1):
        record_id = _required_text(record, id_key, f"{source}:record {row_number}")
        if record_id in indexed:
            raise ValueError(f"{source}: duplicate {id_key} {record_id!r}")
        indexed[record_id] = record
    return indexed
def _clean_text(
    cleaner: TextCleaner, text: str, evidence_id: str, field_name: str
) -> TextCleanResult:
    try:
        result = cleaner(text)
    except Exception as exc:
        raise ValueError(
            f"evidence {evidence_id}: text_cleaner failed for {field_name}"
        ) from exc
    if not isinstance(result, TextCleanResult):
        raise ValueError(
            f"evidence {evidence_id}: text_cleaner returned invalid result"
        )
    return result
def _validate_join(
    evidence_id: str, evidence: Mapping[str, Any], context: Mapping[str, Any],
    layer_index: ChunkLayerIndex,
) -> tuple[str, ChunkLayer, ExpansionMode]:
    chunk_id = _required_text(evidence, "chunk_id", f"evidence {evidence_id}")
    original_chunk_id = _required_text(
        context, "original_chunk_id", f"context {evidence_id}"
    )
    if original_chunk_id != chunk_id:
        raise ValueError(f"evidence {evidence_id}: context chunk_id mismatch")
    if chunk_id not in layer_index.by_chunk_id:
        raise ValueError(f"evidence {evidence_id}: chunk_id missing from layer index")
    try:
        hit_layer = ChunkLayer(context.get("hit_layer"))
        expansion_mode = ExpansionMode(context.get("expansion_mode"))
    except ValueError as exc:
        raise ValueError(f"context {evidence_id}: invalid enum value") from exc
    if layer_index.by_chunk_id[chunk_id] is not hit_layer:
        raise ValueError(f"evidence {evidence_id}: hit_layer disagrees with layer index")
    return chunk_id, hit_layer, expansion_mode
def _string_list(context: Mapping[str, Any], key: str, evidence_id: str) -> list[str]:
    values = context.get(key)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"context {evidence_id}: {key} must be a list of strings")
    return list(values)
def _build_package(
    evidence_id: str, evidence: Mapping[str, Any], context: Mapping[str, Any],
    target: TargetRequirement, queries_by_index: Mapping[int, RetrievalQuery],
    layer_index: ChunkLayerIndex, text_cleaner: TextCleaner,
) -> CoverageInputPackage:
    location = f"evidence {evidence_id}"
    query_index = _required_int(evidence, "query_index", location)
    if query_index not in queries_by_index:
        raise ValueError(f"{location}: query_index {query_index} is absent from query plan")
    planned_query = queries_by_index[query_index]
    try:
        evidence_stage = RetrievalStage(evidence.get("retrieval_stage"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: invalid retrieval_stage") from exc
    if evidence_stage is not planned_query.retrieval_stage:
        raise ValueError(f"{location}: retrieval_stage disagrees with query plan")
    primary_hit = _make_primary_query_hit(
        evidence, planned_query, evidence_stage, location
    )
    query_hits = _load_query_hits(
        evidence, queries_by_index, primary_hit, location
    )

    chunk_id, hit_layer, expansion_mode = _validate_join(
        evidence_id, evidence, context, layer_index
    )
    matched_text = str(evidence.get("matched_text") or "")
    expanded_text = str(context.get("expanded_text") or "")
    matched_result = _clean_text(
        text_cleaner, matched_text, evidence_id, "matched_text"
    )
    expanded_result = _clean_text(
        text_cleaner, expanded_text, evidence_id, "expanded_text"
    )

    warnings: list[str] = []
    if not target.required_concept_ids:
        warnings.append("no_required_concepts")
    if not str(context.get("parent_id") or "").strip():
        warnings.append("missing_parent_id")
    if matched_result.reject_reason:
        warnings.append("matched_text_technical_reject")
    if expanded_result.reject_reason:
        warnings.append("expanded_text_technical_reject")
    if not expanded_result.reject_reason and not expanded_result.text.strip():
        warnings.append("empty_cleaned_expanded_text")
    analysis_ready = bool(
        target.required_concept_ids
        and not expanded_result.reject_reason
        and expanded_result.text.strip()
    )

    return CoverageInputPackage(
        evidence_id=evidence_id,
        target_requirement=target,
        planned_query=planned_query,
        executed_query_text=primary_hit.executed_query_text,
        retrieval_language=primary_hit.retrieval_language,
        rank=primary_hit.rank,
        score=primary_hit.score,
        dense_score=primary_hit.dense_score,
        rerank_score=primary_hit.rerank_score,
        query_hits=query_hits,
        chunk_id=chunk_id,
        document_id=str(context.get("document_id") or ""),
        language=str(evidence.get("language") or ""),
        source_type=str(evidence.get("source_type") or ""),
        title=str(evidence.get("title") or ""),
        section=str(evidence.get("section") or ""),
        source_url=str(evidence.get("source_url") or ""),
        citation=str(evidence.get("citation") or ""),
        matched_text=matched_text,
        evidence_text=str(evidence.get("text") or matched_text),
        parent_id=str(context.get("parent_id") or ""),
        hit_layer=hit_layer,
        expansion_mode=expansion_mode,
        source_chunk_ids=_string_list(context, "source_chunk_ids", evidence_id),
        expanded_text=expanded_text,
        context_warnings=_string_list(context, "warnings", evidence_id),
        cleaned_matched_text=("" if matched_result.reject_reason else matched_result.text),
        cleaned_expanded_text=("" if expanded_result.reject_reason else expanded_result.text),
        matched_cleaning_actions=list(matched_result.actions),
        expanded_cleaning_actions=list(expanded_result.actions),
        matched_reject_reason=matched_result.reject_reason or None,
        expanded_reject_reason=expanded_result.reject_reason or None,
        analysis_ready=analysis_ready,
        input_warnings=warnings,
    )
def load_coverage_input_package(
    evidence_path: Path, query_plan_path: Path, context_expanded_path: Path,
    layer_index: ChunkLayerIndex, *, text_cleaner: TextCleaner,
) -> QuestionInputPackage:
    """加载同一问题的三类产物，严格连接后返回问题级输入包。"""
    evidence_payload = _read_json_object(evidence_path)
    plan_payload = _read_json_object(query_plan_path)
    context_records = _read_jsonl_objects(context_expanded_path)

    question_text = _required_text(evidence_payload, "question", str(evidence_path))
    query_record = plan_payload.get("query_record")
    if not isinstance(query_record, dict):
        raise ValueError(f"{query_plan_path}: query_record must be an object")
    query_id = _required_text(query_record, "query_id", str(query_plan_path))
    planned_question = _required_text(query_record, "query_text", str(query_plan_path))
    if planned_question != question_text:
        raise ValueError("evidence question disagrees with query plan")

    queries = _parse_queries(plan_payload, query_plan_path)
    queries_by_index = _index_queries(queries, query_plan_path)
    target = _make_target(query_id, question_text, queries, query_plan_path)
    evidence_values = evidence_payload.get("evidence")
    if not isinstance(evidence_values, list) or not all(
        isinstance(value, dict) for value in evidence_values
    ):
        raise ValueError(f"{evidence_path}: evidence must be a list of objects")
    evidence_by_id = _index_records(
        evidence_values, id_key="evidence_id", source=evidence_path
    )
    context_by_id = _index_records(
        context_records, id_key="evidence_id", source=context_expanded_path
    )
    if evidence_by_id.keys() != context_by_id.keys():
        missing = sorted(evidence_by_id.keys() - context_by_id.keys())
        extra = sorted(context_by_id.keys() - evidence_by_id.keys())
        raise ValueError(f"context evidence_id mismatch: missing={missing}, extra={extra}")

    packages = [
        _build_package(
            str(evidence["evidence_id"]),
            evidence,
            context_by_id[str(evidence["evidence_id"])],
            target,
            queries_by_index,
            layer_index,
            text_cleaner,
        )
        for evidence in evidence_values
    ]
    question_warnings = [] if target.required_concept_ids else ["no_required_concepts"]
    return QuestionInputPackage(
        query_plan_schema_version=_required_text(
            plan_payload, "schema_version", str(query_plan_path)
        ),
        query_id=query_id,
        question_text=question_text,
        target_requirement=target,
        retrieval_queries=queries,
        evidence_packages=packages,
        source_evidence_path=evidence_path,
        source_query_plan_path=query_plan_path,
        source_context_path=context_expanded_path,
        input_warnings=question_warnings,
    )
