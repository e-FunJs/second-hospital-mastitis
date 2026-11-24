"""将合并后的全部证据统一送入 R1，并原子写出上下文。

输入：C 步骤合并结果、调用方指定的层文件/metadata/tokenizer 路径及输出位置。
输出：与输入顺序一致的内存上下文和 JSONL 路径。
不做什么：不检索、不读取旧上下文、不判断证据覆盖，也不修改 evidence。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rag_medical.common.step04_build_faiss_index import read_metadata_jsonl

from .chunk_layer import ChunkLayerIndex, load_chunk_layer_index
from .context_expand import ExpandedContext, expand_evidence_context
from .context_expand_io import write_context_expanded
from .merge_evidence import EvidenceMergeResult


@dataclass(frozen=True)
class ContextExpansionRunResult:
    """保留本次完整扩展结果及对应的原子写入路径。"""

    contexts: list[ExpandedContext]  # 与合并证据逐项对应的上下文。
    output_path: Path  # 原子写入成功后的最终路径。


def _load_metadata_index(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in read_metadata_jsonl(path):
            chunk_id = record.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"empty chunk_id in {path}")
            if chunk_id in metadata_by_id:
                raise ValueError(f"duplicate chunk_id in metadata: {chunk_id}")
            metadata_by_id[chunk_id] = record
    return metadata_by_id


def _load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )


def _validated_evidence_ids(merge_result: EvidenceMergeResult) -> list[str]:
    evidence = merge_result.evidence
    if merge_result.statistics.merged_evidence_count != len(evidence):
        raise ValueError("merged_evidence_count disagrees with evidence length")
    ids: list[str] = []
    for row in evidence:
        evidence_id = row.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string")
        if not isinstance(row.get("query_hits"), list) or not row["query_hits"]:
            raise ValueError(f"evidence {evidence_id}: missing C-step query_hits")
        ids.append(evidence_id)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evidence_id in merged evidence")
    return ids


def _validate_chunk_sources(
    evidence: Sequence[Mapping[str, Any]],
    layer_index: ChunkLayerIndex,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    for row in evidence:
        evidence_id = row["evidence_id"]
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"evidence {evidence_id}: chunk_id must be non-empty")
        if chunk_id not in layer_index.by_chunk_id:
            raise ValueError(f"evidence {evidence_id}: chunk_id missing from layer index")
        if chunk_id not in metadata_by_id:
            raise ValueError(f"evidence {evidence_id}: chunk_id missing from metadata")


def _validate_contexts(contexts: Sequence[ExpandedContext], expected_ids: list[str]) -> None:
    if len(contexts) != len(expected_ids):
        raise ValueError("expanded context count disagrees with evidence length")
    if [context.evidence_id for context in contexts] != expected_ids:
        raise ValueError("expanded context evidence_id order disagrees with input")


def expand_merged_evidence_contexts(
    merge_result: EvidenceMergeResult,
    *,
    strict_paths: Sequence[Path],
    review_paths: Sequence[Path],
    excluded_paths: Sequence[Path],
    metadata_paths: Sequence[Path],
    tokenizer_path: Path,
    max_tokens: int,
    output_dir: Path,
    query_slug: str,
) -> ContextExpansionRunResult:
    """统一重算全部合并证据的 R1 上下文，校验后原子写入。"""

    expected_ids = _validated_evidence_ids(merge_result)
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    layer_index = load_chunk_layer_index(
        strict_paths=strict_paths,
        review_paths=review_paths,
        excluded_paths=excluded_paths,
    )
    metadata_by_id = _load_metadata_index(metadata_paths)
    _validate_chunk_sources(merge_result.evidence, layer_index, metadata_by_id)
    tokenizer = _load_tokenizer(tokenizer_path)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # R0/R2 采用同一批资源重算，避免复用旧上下文引入版本差异。
    contexts = [
        expand_evidence_context(
            row,
            layer_index,
            metadata_by_id,
            max_tokens=max_tokens,
            token_counter=count_tokens,
        )
        for row in merge_result.evidence
    ]
    _validate_contexts(contexts, expected_ids)
    output_path = write_context_expanded(contexts, output_dir, query_slug)
    return ContextExpansionRunResult(contexts=contexts, output_path=output_path)
