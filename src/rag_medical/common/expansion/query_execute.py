"""
用途：机械执行编排器已经选定的 R2 查询，并保留逐查询、逐语言的检索来源。
输入：查询执行请求、检索资源路径，以及可选的单任务检索函数。
输出：每个执行任务的命中、零命中或技术失败结果；模型在一个批次中只加载一次。
不做什么：不读取 activation、不决定下一轮、不合并去重、不扩展上下文、不写文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import faiss
from rag_medical.common.expansion.models import QueryGenerationKind, RetrievalStage
from rag_medical.common.step04_build_faiss_index import read_metadata_jsonl
from rag_medical.common.step06_search_chunks import (
    encode_query,
    load_sentence_transformer,
    rerank_hits,
    search_index,
)

class ExecutionStatus(str, Enum):
    HITS_FOUND = "HITS_FOUND"
    ZERO_HITS = "ZERO_HITS"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
class ExecutionFailureKind(str, Enum):
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    RERANKER_LOAD_FAILED = "RERANKER_LOAD_FAILED"
    INDEX_LOAD_FAILED = "INDEX_LOAD_FAILED"
    METADATA_LOAD_FAILED = "METADATA_LOAD_FAILED"
    ENCODING_FAILED = "ENCODING_FAILED"
    SEARCH_FAILED = "SEARCH_FAILED"
    RERANK_FAILED = "RERANK_FAILED"
    TIMEOUT = "TIMEOUT"
    NETWORK_FAILED = "NETWORK_FAILED"

@dataclass(frozen=True)
class RetrievalParameters:
    top_k: int  # 每个查询、每个语言索引最终保留的命中数。
    candidate_k: int  # 每个查询送入重排器的稠密召回候选数。
@dataclass(frozen=True)
class QueryExecutionTask:
    query_index: int  # 查询计划中的稳定序号。
    retrieval_stage: RetrievalStage  # 本执行器只接受 R2。
    generation_kind: QueryGenerationKind  # 查询的生成方式。
    source_query_index: int | None  # 派生查询的父查询序号。
    planned_query_text: str  # 查询计划中记录的原始查询文本。
    executed_query_text: str  # 实际送入本语言索引的查询文本。
    query_language: str  # 计划查询的语言形态：zh、en、mixed 或 unknown。
    retrieval_language: str  # 本次检索的目标索引：zh 或 en。
    concept_ids: tuple[str, ...]  # 查询中保留的概念。
    omitted_concept_ids: tuple[str, ...]  # 派生查询主动省略的概念。

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str  # 命中子块的稳定标识。
    matched_text: str  # 稠密检索实际命中的子块正文。
    chunk_language: str  # metadata 语言；缺失时取目标索引语言。
    document_id: str  # 来源文档标识。
    source_type: str  # 来源类型。
    pmcid: str  # PMC 标识。
    pmid: str  # PubMed 标识。
    doi: str  # DOI。
    title: str  # 文献标题。
    journal: str  # 期刊名称。
    year: str  # 发表年份。
    section: str  # 命中章节。
    source_url: str  # 来源链接。

@dataclass(frozen=True)
class QueryHit:
    chunk: RetrievedChunk  # 命中的子块和来源字段。
    query_index: int  # 产生本命中的查询序号。
    retrieval_stage: RetrievalStage  # 固定为 R2。
    generation_kind: QueryGenerationKind  # 产生本命中的查询类型。
    planned_query_text: str  # 查询计划文本。
    executed_query_text: str  # 实际检索文本。
    query_language: str  # 查询文本语言形态。
    retrieval_language: str  # 命中的索引语言。
    rank: int  # 在 query_index + retrieval_language 内的重排排名。
    score: float  # 当前采用的最终分数。
    dense_score: float  # 稠密召回分数。
    rerank_score: float  # 重排分数。

@dataclass(frozen=True)
class ExecutionFailure:
    kind: ExecutionFailureKind  # 失败类别。
    message: str  # 不用于覆盖判断的诊断文本。
@dataclass(frozen=True)
class QueryTaskResult:
    task: QueryExecutionTask  # 原始执行任务。
    status: ExecutionStatus  # 命中、零命中或技术失败。
    hits: tuple[QueryHit, ...]  # 保持查询内排名的命中。
    failure: ExecutionFailure | None  # 仅技术失败时存在。
@dataclass(frozen=True)
class QueryExecutionRequest:
    query_id: str  # 原始问题的稳定标识。
    tasks: tuple[QueryExecutionTask, ...]  # 已由编排器选择和路由的任务。
    parameters: RetrievalParameters  # 本批次统一检索参数。
@dataclass(frozen=True)
class QueryExecutionBatchResult:
    query_id: str  # 原始问题的稳定标识。
    task_results: tuple[QueryTaskResult, ...]  # 每个任务恰好对应一条结果。

@dataclass(frozen=True)
class RetrievalResourcePaths:
    model_path: Path  # 句向量模型目录。
    reranker_path: Path  # 重排模型目录。
    chinese_index_path: Path  # 中文 FAISS 索引。
    chinese_metadata_path: Path  # 中文 metadata。
    english_index_path: Path  # 英文 FAISS 索引。
    english_metadata_path: Path  # 英文 metadata。
    device: str  # 已解析的 cpu 或 cuda 设备。
@dataclass(frozen=True)
class LanguageRuntime:
    index: Any  # 已加载的单语言 FAISS 索引。
    metadata_records: tuple[dict[str, Any], ...]  # 与索引行号一一对应的记录。
@dataclass(frozen=True)
class RetrievalRuntime:
    embedding_model: Any  # 本批次共享的查询编码模型。
    reranker: Any  # 本批次共享的重排模型。
    language_resources: Mapping[str, LanguageRuntime]  # 按 zh/en 保存的索引资源。

class RetrievalTechnicalError(RuntimeError):
    def __init__(self, kind: ExecutionFailureKind, message: str) -> None:
        super().__init__(message)
        self.failure = ExecutionFailure(kind=kind, message=message)
RetrieveTask = Callable[[QueryExecutionTask, RetrievalParameters, RetrievalRuntime], Sequence[QueryHit]]
RuntimeLoader = Callable[[RetrievalResourcePaths, tuple[str, ...]], RetrievalRuntime]

def _failure(kind: ExecutionFailureKind, error: Exception) -> ExecutionFailure:
    if isinstance(error, TimeoutError):
        kind = ExecutionFailureKind.TIMEOUT
    elif isinstance(error, ConnectionError):
        kind = ExecutionFailureKind.NETWORK_FAILED
    return ExecutionFailure(kind=kind, message=f"{type(error).__name__}: {error}")
def _raise_technical(kind: ExecutionFailureKind, error: Exception) -> None:
    failure = _failure(kind, error)
    raise RetrievalTechnicalError(failure.kind, failure.message) from error

def load_retrieval_runtime(paths: RetrievalResourcePaths, retrieval_languages: tuple[str, ...]) -> RetrievalRuntime:
    """一次加载共享模型及本批次实际需要的语言索引。"""
    try:
        embedding_model = load_sentence_transformer(paths.model_path, paths.device)
    except Exception as error:
        _raise_technical(ExecutionFailureKind.MODEL_LOAD_FAILED, error)
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(str(paths.reranker_path), device=paths.device)
    except Exception as error:
        _raise_technical(ExecutionFailureKind.RERANKER_LOAD_FAILED, error)
    language_resources: dict[str, LanguageRuntime] = {}
    path_pairs = {
        "zh": (paths.chinese_index_path, paths.chinese_metadata_path),
        "en": (paths.english_index_path, paths.english_metadata_path),
    }
    for language in retrieval_languages:
        index_path, metadata_path = path_pairs[language]
        try:
            index = faiss.read_index(str(index_path))
        except Exception as error:
            _raise_technical(ExecutionFailureKind.INDEX_LOAD_FAILED, error)
        try:
            metadata = tuple(read_metadata_jsonl(metadata_path))
        except Exception as error:
            _raise_technical(ExecutionFailureKind.METADATA_LOAD_FAILED, error)
        language_resources[language] = LanguageRuntime(index, metadata)
    return RetrievalRuntime(embedding_model, reranker, language_resources)

def _retrieve_ranked_hits(task: QueryExecutionTask, parameters: RetrievalParameters, runtime: RetrievalRuntime, language_runtime: LanguageRuntime) -> list[dict[str, Any]]:
    try:
        query_embedding = encode_query(runtime.embedding_model, task.executed_query_text)
    except Exception as error:
        _raise_technical(ExecutionFailureKind.ENCODING_FAILED, error)
    try:
        raw_hits = search_index(language_runtime.index, query_embedding, parameters.candidate_k)
    except Exception as error:
        _raise_technical(ExecutionFailureKind.SEARCH_FAILED, error)
    if not raw_hits:
        return []
    if any(int(hit["row_index"]) >= len(language_runtime.metadata_records) for hit in raw_hits):
        raise RetrievalTechnicalError(
            ExecutionFailureKind.METADATA_LOAD_FAILED,
            "search result row exceeds metadata length",
        )
    try:
        return rerank_hits(
            task.executed_query_text,
            raw_hits,
            language_runtime.metadata_records,
            runtime.reranker,
            parameters.top_k,
        )
    except Exception as error:
        _raise_technical(ExecutionFailureKind.RERANK_FAILED, error)

def _build_query_hit(
    task: QueryExecutionTask,
    metadata: Mapping[str, Any],
    hit: Mapping[str, Any],
    rank: int,
) -> QueryHit:
    chunk = RetrievedChunk(
        chunk_id=str(metadata.get("chunk_id") or ""),
        matched_text=str(metadata.get("text") or ""),
        chunk_language=str(metadata.get("language") or task.retrieval_language),
        document_id=str(metadata.get("document_id") or ""),
        source_type=str(metadata.get("source_type") or ""),
        pmcid=str(metadata.get("pmcid") or ""),
        pmid=str(metadata.get("pmid") or ""),
        doi=str(metadata.get("doi") or ""),
        title=str(metadata.get("title") or ""),
        journal=str(metadata.get("journal") or ""),
        year=str(metadata.get("year") or ""),
        section=str(metadata.get("section") or ""),
        source_url=str(metadata.get("source_url") or ""),
    )
    return QueryHit(
        chunk=chunk,
        query_index=task.query_index,
        retrieval_stage=task.retrieval_stage,
        generation_kind=task.generation_kind,
        planned_query_text=task.planned_query_text,
        executed_query_text=task.executed_query_text,
        query_language=task.query_language,
        retrieval_language=task.retrieval_language,
        rank=rank,
        score=round(float(hit["score"]), 6),
        dense_score=round(float(hit["dense_score"]), 6),
        rerank_score=round(float(hit["rerank_score"]), 6),
    )

def retrieve_query_task(task: QueryExecutionTask, parameters: RetrievalParameters, runtime: RetrievalRuntime) -> tuple[QueryHit, ...]:
    """执行单个查询和单个语言索引，不合并其他查询的重复 chunk。"""
    language_runtime = runtime.language_resources[task.retrieval_language]
    ranked_hits = _retrieve_ranked_hits(task, parameters, runtime, language_runtime)
    return tuple(
        _build_query_hit(
            task,
            language_runtime.metadata_records[int(hit["row_index"])],
            hit,
            rank,
        )
        for rank, hit in enumerate(ranked_hits, start=1)
    )
def _validate_request(request: QueryExecutionRequest) -> None:
    if not request.query_id:
        raise ValueError("query_id must not be empty")
    if request.parameters.top_k <= 0 or request.parameters.candidate_k < request.parameters.top_k:
        raise ValueError("candidate_k must be greater than or equal to positive top_k")
    task_keys: set[tuple[int, str]] = set()
    for task in request.tasks:
        if task.retrieval_stage is not RetrievalStage.R2:
            raise ValueError(f"R2 executor received {task.retrieval_stage.value}")
        if not task.executed_query_text.strip() or task.retrieval_language not in {"zh", "en"}:
            raise ValueError("task must have query text and a zh/en retrieval language")
        task_key = (task.query_index, task.retrieval_language)
        if task_key in task_keys:
            raise ValueError(f"duplicate execution task: {task_key}")
        task_keys.add(task_key)
def execute_query_batch(
    request: QueryExecutionRequest,
    resource_paths: RetrievalResourcePaths,
    *,
    retrieve_task: RetrieveTask = retrieve_query_task,
    runtime_loader: RuntimeLoader = load_retrieval_runtime,
) -> QueryExecutionBatchResult:
    """按输入顺序执行全部任务；只把明确技术异常转成任务失败。"""
    _validate_request(request)
    retrieval_languages = tuple(dict.fromkeys(task.retrieval_language for task in request.tasks))
    try:
        runtime = runtime_loader(resource_paths, retrieval_languages)
    except RetrievalTechnicalError as error:
        failed = tuple(
            QueryTaskResult(task, ExecutionStatus.TECHNICAL_FAILURE, (), error.failure)
            for task in request.tasks
        )
        return QueryExecutionBatchResult(request.query_id, failed)
    task_results: list[QueryTaskResult] = []
    for task in request.tasks:
        try:
            hits = tuple(retrieve_task(task, request.parameters, runtime))
        except RetrievalTechnicalError as error:
            task_results.append(
                QueryTaskResult(task, ExecutionStatus.TECHNICAL_FAILURE, (), error.failure)
            )
            continue
        status = ExecutionStatus.HITS_FOUND if hits else ExecutionStatus.ZERO_HITS
        task_results.append(QueryTaskResult(task, status, hits, None))
    return QueryExecutionBatchResult(request.query_id, tuple(task_results))
