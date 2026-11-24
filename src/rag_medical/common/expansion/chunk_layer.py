"""建立检索子块的分层归属索引。

用途：以 strict、review、excluded 三类 JSONL 文件的成员关系为唯一依据，
为每个子块记录层级，并按非空 parent_id 汇总父块内的子块层级。
输入：由调用方显式传入的三组 JSONL 路径。
输出：完全驻留内存的 ``ChunkLayerIndex``。
不做：不推断文件路径，不读取 evidence，不扩展上下文，也不写任何文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence


class ChunkLayer(str, Enum):
    """子块在筛选流程中的权威层级。"""

    STRICT = "strict"
    REVIEW = "review"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ChunkLayerIndex:
    """保存子块层级及父块内逐子块层级的内存索引。"""

    by_chunk_id: dict[str, ChunkLayer]
    by_parent_id: dict[str, dict[str, ChunkLayer]]


def _load_layer_paths(
    paths: Sequence[Path],
    layer: ChunkLayer,
    by_chunk_id: dict[str, ChunkLayer],
    by_parent_id: dict[str, dict[str, ChunkLayer]],
    source_by_chunk_id: dict[str, tuple[Path, int]],
) -> None:
    """读取一层文件，并拒绝任何重复或冲突的子块归属。"""

    for path in paths:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                if not raw_line.strip():
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"{path}:{line_number}: JSON record must be an object"
                    )

                chunk_id = str(record.get("chunk_id") or "").strip()
                if not chunk_id:
                    raise ValueError(
                        f"{path}:{line_number}: chunk_id must be non-empty"
                    )

                if chunk_id in by_chunk_id:
                    previous_path, previous_line = source_by_chunk_id[chunk_id]
                    previous_layer = by_chunk_id[chunk_id]
                    raise ValueError(
                        f"{path}:{line_number}: duplicate chunk_id {chunk_id!r}; "
                        f"already assigned to {previous_layer.value} at "
                        f"{previous_path}:{previous_line}"
                    )

                by_chunk_id[chunk_id] = layer
                source_by_chunk_id[chunk_id] = (path, line_number)

                parent_id = str(record.get("parent_id") or "").strip()
                if parent_id:
                    by_parent_id.setdefault(parent_id, {})[chunk_id] = layer


def load_chunk_layer_index(
    *,
    strict_paths: Sequence[Path],
    review_paths: Sequence[Path],
    excluded_paths: Sequence[Path],
) -> ChunkLayerIndex:
    """从调用方指定的层文件加载完整的子块归属索引。

    同一 ``chunk_id`` 无论出现在同层还是跨层文件中都视为数据错误，
    因为静默覆盖会使后续上下文扩展失去可审计的层级边界。
    """

    by_chunk_id: dict[str, ChunkLayer] = {}
    by_parent_id: dict[str, dict[str, ChunkLayer]] = {}
    source_by_chunk_id: dict[str, tuple[Path, int]] = {}

    _load_layer_paths(
        strict_paths,
        ChunkLayer.STRICT,
        by_chunk_id,
        by_parent_id,
        source_by_chunk_id,
    )
    _load_layer_paths(
        review_paths,
        ChunkLayer.REVIEW,
        by_chunk_id,
        by_parent_id,
        source_by_chunk_id,
    )
    _load_layer_paths(
        excluded_paths,
        ChunkLayer.EXCLUDED,
        by_chunk_id,
        by_parent_id,
        source_by_chunk_id,
    )

    return ChunkLayerIndex(
        by_chunk_id=by_chunk_id,
        by_parent_id=by_parent_id,
    )
