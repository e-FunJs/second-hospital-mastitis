"""把已经计算完成的 R1 上下文扩展结果写成 JSONL。

输入：调用方提供的 ``ExpandedContext`` 序列、已存在的输出目录和稳定的
query_slug。输出：``<query_slug>_context_expanded.jsonl`` 及其路径。
本模块不读取 evidence、不计算上下文、不加载 tokenizer，也不创建输出目录。
写入先在内存中完成校验和序列化，再通过同目录临时文件原子替换目标文件。
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .context_expand import ExpandedContext


def _validate_query_slug(query_slug: str) -> None:
    if not query_slug.strip():
        raise ValueError("query_slug must be non-empty")
    if query_slug in {".", ".."} or "/" in query_slug or "\\" in query_slug:
        raise ValueError("query_slug must not contain path traversal components")


def _serialize_context(context: ExpandedContext) -> str:
    record = {
        "evidence_id": context.evidence_id,
        "document_id": context.document_id,
        "original_chunk_id": context.original_chunk_id,
        "parent_id": context.parent_id,
        "hit_layer": context.hit_layer.value,
        "expansion_mode": context.expansion_mode.value,
        "source_chunk_ids": context.source_chunk_ids,
        "expanded_text": context.expanded_text,
        "token_count": context.token_count,
        "warnings": context.warnings,
    }
    return json.dumps(record, ensure_ascii=False)


def _build_payload(contexts: Sequence[ExpandedContext]) -> str:
    evidence_ids: set[str] = set()
    serialized_lines: list[str] = []
    for context in contexts:
        if not context.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if context.evidence_id in evidence_ids:
            raise ValueError(f"duplicate evidence_id: {context.evidence_id!r}")
        evidence_ids.add(context.evidence_id)
        serialized_lines.append(_serialize_context(context))
    if len(serialized_lines) != len(contexts):
        raise ValueError("serialized record count does not match input count")
    return "\n".join(serialized_lines) + ("\n" if serialized_lines else "")


def _atomic_write(output_path: Path, payload: str) -> None:
    prefix = f".{output_path.name}.tmp."
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=prefix,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_context_expanded(
    contexts: Sequence[ExpandedContext],
    output_dir: Path,
    query_slug: str,
) -> Path:
    """校验并原子覆盖写入上下文扩展 JSONL，返回最终文件路径。"""

    _validate_query_slug(query_slug)
    if not output_dir.exists():
        raise FileNotFoundError(f"output directory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")

    output_path = output_dir / f"{query_slug}_context_expanded.jsonl"
    payload = _build_payload(contexts)
    _atomic_write(output_path, payload)
    return output_path
