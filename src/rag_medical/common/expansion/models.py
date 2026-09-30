"""
用途：定义证据检索外扩流程共享的基础枚举和核心数据结构。
输入：原始问题、检索查询及检索所得证据的结构化字段。
输出：可转换为 JSON 兼容字典、也可从字典恢复的 dataclass 对象。
不做什么：不执行检索、模型推理、覆盖判断、报告生成或文件读写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class RetrievalStage(str, Enum):
    """检索外扩所处阶段。"""

    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"


class QueryGenerationKind(str, Enum):
    """检索查询的生成方式。"""

    ORIGINAL = "ORIGINAL"
    ALIAS_SUBSTITUTION = "ALIAS_SUBSTITUTION"
    PARTIAL_COMBINATION = "PARTIAL_COMBINATION"
    SINGLE_DRUG = "SINGLE_DRUG"


class QueryActivation(str, Enum):
    """检索查询在逐级外扩流程中的启用条件。"""

    ALWAYS = "always"
    AFTER_FULL_SCHEME_GAP = "after_full_scheme_gap"
    AFTER_PARTIAL_SCHEME_GAP = "after_partial_scheme_gap"


class EvidenceState(str, Enum):
    """针对用户问题的证据支持状态。"""

    DIRECT_CANDIDATE = "DIRECT_CANDIDATE"
    PARTIAL_ONLY = "PARTIAL_ONLY"
    CONFLICTING = "CONFLICTING"
    NOT_FOUND_LOCAL = "NOT_FOUND_LOCAL"
    NOT_FOUND_SCOPE = "NOT_FOUND_SCOPE"
    CANNOT_ASSESS = "CANNOT_ASSESS"


class AssertionRole(str, Enum):
    """医学实体在证据原文中的陈述角色。"""

    STUDY_INTERVENTION = "STUDY_INTERVENTION"
    COMPARATOR = "COMPARATOR"
    PRIOR_TREATMENT = "PRIOR_TREATMENT"
    NEGATED_OR_EXCLUDED = "NEGATED_OR_EXCLUDED"
    HYPOTHETICAL = "HYPOTHETICAL"
    BACKGROUND = "BACKGROUND"
    UNCLEAR = "UNCLEAR"


class StopReason(str, Enum):
    """停止证据外扩或无法继续判断的原因。"""

    STOP_LOCAL_GAP = "STOP_LOCAL_GAP"
    STOP_PARTIAL_ONLY = "STOP_PARTIAL_ONLY"
    STOP_CONFLICT = "STOP_CONFLICT"
    STOP_METHOD_UNAVAILABLE = "STOP_METHOD_UNAVAILABLE"
    STOP_AMBIGUOUS = "STOP_AMBIGUOUS"
    STOP_SEARCH_SCOPE = "STOP_SEARCH_SCOPE"


@dataclass(frozen=True)
class QueryRecord:
    """用户提交的原始问题记录。"""

    query_id: str  # 原始问题的稳定标识。
    query_text: str  # 用户提交的原始问题文本。

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接进行 JSON 序列化的字典。"""
        return {"query_id": self.query_id, "query_text": self.query_text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QueryRecord":
        """从字段字典恢复原始问题记录。"""
        return cls(query_id=data["query_id"], query_text=data["query_text"])


@dataclass(frozen=True)
class RetrievalQuery:
    """某一检索阶段实际送入检索器的查询。"""

    retrieval_stage: RetrievalStage  # 该查询属于 R0、R1、R2 或 R3。
    query_index: int  # 同一原始问题下检索查询的稳定序号。
    query_text: str  # 实际用于检索的查询文本。
    query_language: str  # 查询文本的语言形态：zh、en、mixed 或 unknown。
    generation_kind: QueryGenerationKind  # 原查询或具体外扩生成方式。
    source_query_index: Optional[int]  # 派生查询的父查询序号；原查询为 None。
    concept_ids: list[str]  # 查询中保留的确定概念，按原文首次出现排序。
    omitted_concept_ids: list[str]  # 生成该查询时主动省略的概念。
    activation: QueryActivation  # 查询在逐级检索中的启用条件。

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接进行 JSON 序列化的字典。"""
        return {
            "retrieval_stage": self.retrieval_stage.value,
            "query_index": self.query_index,
            "query_text": self.query_text,
            "query_language": self.query_language,
            "generation_kind": self.generation_kind.value,
            "source_query_index": self.source_query_index,
            "concept_ids": list(self.concept_ids),
            "omitted_concept_ids": list(self.omitted_concept_ids),
            "activation": self.activation.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalQuery":
        """从字段字典恢复检索查询，并解析阶段枚举。"""
        return cls(
            retrieval_stage=RetrievalStage(data["retrieval_stage"]),
            query_index=data["query_index"],
            query_text=data["query_text"],
            query_language=data["query_language"],
            generation_kind=QueryGenerationKind(data["generation_kind"]),
            source_query_index=data["source_query_index"],
            concept_ids=list(data["concept_ids"]),
            omitted_concept_ids=list(data["omitted_concept_ids"]),
            activation=QueryActivation(data["activation"]),
        )


@dataclass(frozen=True)
class EvidenceItem:
    """一条带检索来源和原文上下文的证据记录。"""

    evidence_id: str  # 在本次问答中的稳定证据编号，如 E1。
    retrieval_stage: RetrievalStage  # 发现该证据的检索阶段。
    retrieval_query: str  # 实际送入对应语言索引的查询文本。
    query_index: int  # 命中该证据的检索查询序号。
    rank: int  # 该证据在对应检索查询中的排名。
    chunk_id: str  # 命中子块的稳定标识。
    matched_text: str  # 检索器直接命中的子块原文。
    text: str  # 提供给后续流程的完整局部上下文。
    score: Optional[float] = None  # 当前流程采用的最终检索分数。
    dense_score: Optional[float] = None  # BGE 稠密检索分数。
    rerank_score: Optional[float] = None  # 重排模型分数。
    language: str = ""  # 证据文本语言，如 zh 或 en。
    source_type: str = ""  # 证据来源类型，如 cnki_pdf 或 pmc_xml。
    document_id: str = ""  # 来源文档在语料库中的稳定标识。
    pmcid: str = ""  # PubMed Central 文献标识。
    pmid: str = ""  # PubMed 文献标识。
    doi: str = ""  # 数字对象标识符。
    title: str = ""  # 来源文献标题。
    journal: str = ""  # 来源期刊名称。
    year: str = ""  # 来源文献发表年份。
    section: str = ""  # 命中内容所在章节。
    source_url: str = ""  # 可追溯的来源链接。
    citation: str = ""  # 面向报告展示的引用文本。
    context_mode: str = "hit_only"  # 上下文扩展方式。
    context_chunk_ids: list[str] = field(default_factory=list)  # 上下文包含的子块编号。

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接进行 JSON 序列化的字典。"""
        data = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
        data["retrieval_stage"] = self.retrieval_stage.value
        data["context_chunk_ids"] = list(self.context_chunk_ids)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceItem":
        """从字段字典恢复证据，并解析检索阶段枚举。"""
        values = dict(data)
        values["retrieval_stage"] = RetrievalStage(values["retrieval_stage"])
        values["context_chunk_ids"] = list(values.get("context_chunk_ids", []))
        return cls(**values)
