"""在 strict 层边界内计算单条证据的安全上下文。

输入由调用方提供：一条 evidence、按 chunk_id 建立的 metadata 映射、层级
索引、token 上限和 token 计数函数。输出为内存中的 ``ExpandedContext``。
本模块不读写文件、不加载 tokenizer，也不接入检索或生成。多个子块固定用
单个 ``"\n"`` 拼接；该分隔符属于最终文本，必须计入 token 数。
review/excluded 子块既不会进入结果，也不能被跨越。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from .chunk_layer import ChunkLayer, ChunkLayerIndex

class ExpansionMode(str, Enum):
    """R1 上下文扩展实际采用的模式。"""

    PARENT_BLOCK = "parent_block"
    STRICT_SIBLINGS_ONLY = "strict_siblings_only"
    HIT_ONLY = "hit_only"
    NONE = "none"

@dataclass(frozen=True)
class ExpandedContext:
    """一条 evidence 的内存扩展结果，供下一步序列化使用。"""

    evidence_id: str
    document_id: str
    original_chunk_id: str
    parent_id: str
    hit_layer: ChunkLayer
    expansion_mode: ExpansionMode
    source_chunk_ids: list[str]
    expanded_text: str
    token_count: int
    warnings: list[str]

@dataclass(frozen=True)
class _Hit:
    evidence_id: str
    chunk_id: str
    document_id: str
    parent_id: str
    text: str
    metadata: Mapping[str, Any] | None

TokenCounter = Callable[[str], int]
Metadata = Mapping[str, Any]

def _text(record: Mapping[str, Any], key: str) -> str:
    return str(record.get(key) or "").strip()

def _content(record: Mapping[str, Any], key: str) -> str:
    return str(record.get(key) or "")

def _warn(warnings: list[str], value: str) -> None:
    if value not in warnings:
        warnings.append(value)

def _count(text: str, token_counter: TokenCounter) -> int:
    try:
        value = token_counter(text)
    except Exception as exc:
        raise ValueError("token_counter failed") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("token_counter must return an integer")
    if value < 0:
        raise ValueError("token_counter must not return a negative value")
    return value

def _document_id(*records: Mapping[str, Any]) -> str:
    for record in records:
        for key in ("document_id", "pmcid", "pmid", "source_path"):
            value = _text(record, key)
            if value:
                return value
    return ""

def _result(
    hit: _Hit, layer: ChunkLayer, mode: ExpansionMode, source_ids: list[str],
    text: str, token_count: int, warnings: list[str],
) -> ExpandedContext:
    return ExpandedContext(
        hit.evidence_id, hit.document_id, hit.chunk_id, hit.parent_id, layer,
        mode, source_ids, text, token_count, warnings,
    )

def _validate(
    evidence: Mapping[str, Any], index: ChunkLayerIndex, max_tokens: int
) -> tuple[str, str, ChunkLayer]:
    evidence_id = _text(evidence, "evidence_id")
    chunk_id = _text(evidence, "chunk_id")
    if not evidence_id:
        raise ValueError("evidence_id must be non-empty")
    if not chunk_id:
        raise ValueError("chunk_id must be non-empty")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if chunk_id not in index.by_chunk_id:
        raise ValueError(f"chunk_id {chunk_id!r} is missing from layer index")
    return evidence_id, chunk_id, index.by_chunk_id[chunk_id]

def _make_hit(
    evidence: Mapping[str, Any], evidence_id: str, chunk_id: str,
    metadata: Metadata | None,
) -> tuple[_Hit, list[str]]:
    warnings: list[str] = []
    hit_text = _content(metadata or {}, "text")
    if not hit_text.strip():
        hit_text = _content(evidence, "matched_text")
        if hit_text.strip() and metadata is not None:
            warnings.append("hit_text_from_evidence")
    return _Hit(
        evidence_id, chunk_id, _document_id(metadata or {}, evidence),
        _text(metadata or {}, "parent_id"), hit_text, metadata,
    ), warnings

def _same_boundary(hit: Metadata, candidate: Metadata, parent_id: str) -> bool:
    if _text(candidate, "parent_id") != parent_id:
        return False
    hit_document = _document_id(hit)
    candidate_document = _document_id(candidate)
    return bool(
        hit_document
        and hit_document == candidate_document
        and _text(hit, "section") == _text(candidate, "section")
    )

def _position(record: Metadata) -> tuple[int, str]:
    for key in ("sentence_start", "chunk_index"):
        try:
            return int(record.get(key)), _text(record, "chunk_id")
        except (TypeError, ValueError):
            pass
    return 2**31 - 1, _text(record, "chunk_id")

def _try_parent(
    hit: Metadata, layers: Mapping[str, ChunkLayer],
    metadata_by_id: Mapping[str, Metadata], max_tokens: int,
    token_counter: TokenCounter, warnings: list[str],
) -> tuple[str, list[str], int] | None:
    if any(layer is not ChunkLayer.STRICT for layer in layers.values()):
        if ChunkLayer.REVIEW in layers.values():
            _warn(warnings, "parent_contains_review")
        if ChunkLayer.EXCLUDED in layers.values():
            _warn(warnings, "parent_contains_excluded")
        return None
    records = [metadata_by_id.get(chunk_id) for chunk_id in layers]
    if any(record is None for record in records):
        _warn(warnings, "parent_metadata_incomplete")
        return None
    siblings = [record for record in records if record is not None]
    parent_id = _text(hit, "parent_id")
    if any(not _same_boundary(hit, record, parent_id) for record in siblings):
        _warn(warnings, "parent_boundary_mismatch")
        return None
    parent_text = _content(hit, "parent_text")
    if not parent_text.strip():
        _warn(warnings, "parent_text_missing")
        return None
    token_count = _count(parent_text, token_counter)
    if token_count > max_tokens:
        _warn(warnings, "parent_block_exceeds_token_limit")
        return None
    siblings.sort(key=_position)
    return parent_text, [_text(record, "chunk_id") for record in siblings], token_count

def _walk(
    hit: Metadata, direction: str, layers: Mapping[str, ChunkLayer],
    metadata_by_id: Mapping[str, Metadata], warnings: list[str],
) -> list[Metadata]:
    parent_id = _text(hit, "parent_id")
    cursor = _text(hit, direction)
    visited = {_text(hit, "chunk_id")}
    found: list[Metadata] = []
    while cursor:
        if cursor in visited:
            _warn(warnings, "neighbor_cycle_detected")
            break
        visited.add(cursor)
        layer = layers.get(cursor)
        if layer is None:
            _warn(warnings, "neighbor_layer_missing")
            break
        if layer is not ChunkLayer.STRICT:
            _warn(warnings, f"neighbor_blocked_by_{layer.value}")
            break
        candidate = metadata_by_id.get(cursor)
        if candidate is None:
            _warn(warnings, "strict_neighbor_metadata_missing")
            break
        if not _same_boundary(hit, candidate, parent_id):
            _warn(warnings, "neighbor_boundary_mismatch")
            break
        found.append(candidate)
        cursor = _text(candidate, direction)
    return found

def _joined(records: list[Metadata]) -> str:
    return "\n".join(_content(record, "text") for record in records)

def _window(
    hit: Metadata, layers: Mapping[str, ChunkLayer],
    metadata_by_id: Mapping[str, Metadata], max_tokens: int,
    token_counter: TokenCounter, warnings: list[str],
) -> tuple[list[Metadata], str, int]:
    candidates = [
        _walk(hit, "previous_chunk_id", layers, metadata_by_id, warnings),
        _walk(hit, "next_chunk_id", layers, metadata_by_id, warnings),
    ]
    selected: list[list[Metadata]] = [[], []]
    active = [True, True]
    for distance in range(max(map(len, candidates), default=0)):
        for side in (0, 1):
            if not active[side] or distance >= len(candidates[side]):
                continue
            proposed = selected[side] + [candidates[side][distance]]
            before = proposed if side == 0 else selected[0]
            after = selected[1] if side == 0 else proposed
            records = list(reversed(before)) + [hit] + after
            if _count(_joined(records), token_counter) > max_tokens:
                active[side] = False
                _warn(warnings, "token_limit_reached")
            else:
                selected[side] = proposed
    records = list(reversed(selected[0])) + [hit] + selected[1]
    joined = _joined(records)
    return records, joined, _count(joined, token_counter)

def _expand_strict(
    hit: _Hit, index: ChunkLayerIndex,
    metadata_by_id: Mapping[str, Metadata], max_tokens: int,
    token_counter: TokenCounter, warnings: list[str],
) -> ExpandedContext:
    if not hit.text.strip():
        return _result(hit, ChunkLayer.STRICT, ExpansionMode.NONE, [], "", 0,
                       warnings + ["empty_hit_text"])
    hit_count = _count(hit.text, token_counter)
    if hit.metadata is None:
        if hit_count > max_tokens:
            raise ValueError("oversized hit block cannot be verified without metadata")
        return _result(hit, ChunkLayer.STRICT, ExpansionMode.HIT_ONLY,
                       [hit.chunk_id], hit.text, hit_count,
                       warnings + ["hit_metadata_missing"])
    if hit_count > max_tokens:
        _warn(warnings, "hit_block_exceeds_token_limit")
    if not hit.parent_id:
        _warn(warnings, "no_parent_id")
    if hit_count > max_tokens or not hit.parent_id:
        return _result(hit, ChunkLayer.STRICT, ExpansionMode.HIT_ONLY,
                       [hit.chunk_id], hit.text, hit_count, warnings)
    layers = index.by_parent_id.get(hit.parent_id)
    if layers is None or layers.get(hit.chunk_id) is not ChunkLayer.STRICT:
        raise ValueError(f"parent layer mapping is inconsistent for {hit.chunk_id!r}")
    parent = _try_parent(
        hit.metadata, layers, metadata_by_id, max_tokens, token_counter, warnings
    )
    if parent is not None:
        text, source_ids, token_count = parent
        return _result(hit, ChunkLayer.STRICT, ExpansionMode.PARENT_BLOCK,
                       source_ids, text, token_count, warnings)
    records, text, token_count = _window(
        hit.metadata, layers, metadata_by_id, max_tokens, token_counter, warnings
    )
    mode = (ExpansionMode.STRICT_SIBLINGS_ONLY
            if len(records) > 1 else ExpansionMode.HIT_ONLY)
    return _result(hit, ChunkLayer.STRICT, mode,
                   [_text(record, "chunk_id") for record in records],
                   text, token_count, warnings)

def expand_evidence_context(
    evidence: Mapping[str, Any],
    layer_index: ChunkLayerIndex,
    chunk_metadata_by_id: Mapping[str, Metadata],
    *,
    max_tokens: int,
    token_counter: TokenCounter,
) -> ExpandedContext:
    """返回一条 evidence 的安全上下文，不执行文件或模型 I/O。"""

    evidence_id, chunk_id, layer = _validate(evidence, layer_index, max_tokens)
    metadata = chunk_metadata_by_id.get(chunk_id)
    hit, warnings = _make_hit(evidence, evidence_id, chunk_id, metadata)
    if layer is ChunkLayer.REVIEW:
        return _result(hit, layer, ExpansionMode.NONE, [], "", 0,
                       ["unexpected_review_hit"])
    if layer is ChunkLayer.EXCLUDED:
        return _result(hit, layer, ExpansionMode.NONE, [], "", 0,
                       ["unexpected_excluded_hit"])
    return _expand_strict(
        hit, layer_index, chunk_metadata_by_id,
        max_tokens, token_counter, warnings,
    )
