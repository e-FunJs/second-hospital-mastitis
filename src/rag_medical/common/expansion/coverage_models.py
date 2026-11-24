"""定义证据覆盖阶段的只读输入契约。

用途：保存原始目标要求、单条证据的检索与上下文来源，以及问题级输入包。
输入：由 coverage_input 模块从现有 evidence、查询计划和 R1 产物组装的数据。
输出：供后续断言识别和覆盖判断使用的 dataclass 对象。
不做什么：不读写文件、不调用模型、不识别断言，也不计算证据覆盖状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunk_layer import ChunkLayer
from .context_expand import ExpansionMode
from .models import QueryGenerationKind, RetrievalQuery, RetrievalStage


@dataclass(frozen=True)
class TargetRequirement:
    """原问题中已确定的目标实体和仍待语义解析的要求。"""

    query_id: str  # 查询计划中的稳定问题标识。
    question_text: str  # 用户提交的原始问题文本。
    required_concept_ids: list[str]  # R0 原查询中必须覆盖的全部概念。
    relation_text: str | None  # 结构化目标关系；6.1 阶段尚未解析。
    condition_texts: list[str]  # 结构化人群、结局或时间条件；6.1 阶段为空。
    unresolved_items: list[str]  # 留待后续语义阶段处理的目标字段。


@dataclass(frozen=True)
class QueryHitProvenance:
    """一条 evidence 被某次计划查询命中的完整检索来源。"""

    retrieval_stage: RetrievalStage  # 命中所属的检索阶段。
    query_index: int  # 查询计划中的稳定序号。
    generation_kind: QueryGenerationKind  # 查询的生成方式。
    planned_query_text: str  # 查询计划中保存的查询文本。
    executed_query_text: str  # 实际送入对应语言索引的查询文本。
    query_language: str  # 计划查询的语言形态。
    retrieval_language: str  # 实际检索的语言索引。
    rank: int  # 该来源在对应查询与语言分支内的排名。
    score: float | None  # 当前流程采用的最终检索分数。
    dense_score: float | None  # 稠密召回分数。
    rerank_score: float | None  # 重排分数。


@dataclass(frozen=True)
class CoverageInputPackage:
    """一条 evidence 进入断言识别前所需的完整、可追溯输入。"""

    evidence_id: str  # 本次问答中的稳定证据编号。
    target_requirement: TargetRequirement  # 始终指向原问题的完整目标。
    planned_query: RetrievalQuery  # query_index 对应的计划查询。
    executed_query_text: str  # 实际送入语言索引的查询文本。
    retrieval_language: str  # 实际检索的语言分支。
    rank: int  # 该证据在对应检索结果中的排名。
    score: float | None  # 当前流程采用的最终检索分数。
    dense_score: float | None  # 稠密检索分数。
    rerank_score: float | None  # 重排分数。
    query_hits: list[QueryHitProvenance]  # 主命中在首位的全部检索来源。
    chunk_id: str  # 检索直接命中的子块标识。
    document_id: str  # R1 记录中的来源文档标识。
    language: str  # evidence metadata 中的原始语言字段。
    source_type: str  # 文献来源类型。
    title: str  # 来源文献标题。
    section: str  # 命中内容所在章节。
    source_url: str  # 可追溯来源链接。
    citation: str  # 面向报告的引用文本。
    matched_text: str  # 检索器直接命中的原始文本。
    evidence_text: str  # step08 已有上下文文本，仅供审计。
    parent_id: str  # R1 记录中的父块标识；无父块时为空。
    hit_layer: ChunkLayer  # 命中子块的权威筛选层级。
    expansion_mode: ExpansionMode  # R1 实际采用的扩展模式。
    source_chunk_ids: list[str]  # expanded_text 所来自的 strict 子块。
    expanded_text: str  # R1 生成的原始扩展文本。
    context_warnings: list[str]  # R1 扩展过程产生的 warning。
    cleaned_matched_text: str  # 清洗后的直接命中文本。
    cleaned_expanded_text: str  # 清洗后的 R1 文本，供后续语义阶段使用。
    matched_cleaning_actions: list[str]  # 命中文本执行的清洗动作。
    expanded_cleaning_actions: list[str]  # 扩展文本执行的清洗动作。
    matched_reject_reason: str | None  # 命中文本的技术拒绝原因。
    expanded_reject_reason: str | None  # 扩展文本的技术拒绝原因。
    analysis_ready: bool  # 文本和目标概念是否具备进入 6.2 的最低条件。
    input_warnings: list[str]  # 组装阶段发现的可恢复问题。


@dataclass(frozen=True)
class QuestionInputPackage:
    """同一问题下的目标、查询计划和全部证据输入。"""

    query_plan_schema_version: str  # 查询计划文件声明的 schema 版本。
    query_id: str  # 查询计划中的稳定问题标识。
    question_text: str  # 经 evidence 与查询计划交叉校验的原问题。
    target_requirement: TargetRequirement  # 原问题的完整目标要求。
    retrieval_queries: list[RetrievalQuery]  # 全部计划查询，包括未执行项。
    evidence_packages: list[CoverageInputPackage]  # 按 evidence 原顺序组装。
    source_evidence_path: Path  # evidence.json 来源路径。
    source_query_plan_path: Path  # query_plan.json 来源路径。
    source_context_path: Path  # context_expanded.jsonl 来源路径。
    input_warnings: list[str]  # 问题级可恢复输入 warning。
