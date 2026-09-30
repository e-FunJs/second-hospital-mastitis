"""
用途：把已经生成的检索查询计划写成稳定、可审计的 JSON 文件。
输入：原始问题记录、RetrievalQuery 列表和一个已经存在的输出目录。
输出：<query_slug>_query_plan.json，并返回该文件路径。
不做什么：不生成查询、不执行检索、不加载模型，也不自动创建输出目录。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag_medical.common.expansion.models import QueryRecord, RetrievalQuery


QUERY_PLAN_SCHEMA_VERSION = "1.0"
_QUERY_SLUG_MAX_LENGTH = 60


def _make_query_slug(text: str, max_length: int = _QUERY_SLUG_MAX_LENGTH) -> str:
    """按当前 RAG 文件命名契约生成稳定的查询标识。"""
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", text).strip("_")
    return slug[:max_length] or "rag_query"


def write_query_plan(
    query_record: QueryRecord,
    queries: list[RetrievalQuery],
    output_dir: Path,
) -> Path:
    """覆盖写入查询计划；调用方必须事先准备好输出目录。"""
    if not output_dir.exists():
        raise FileNotFoundError(f"输出目录不存在：{output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"输出路径不是目录：{output_dir}")

    output_path = output_dir / (
        f"{_make_query_slug(query_record.query_text)}_query_plan.json"
    )
    payload = {
        "schema_version": QUERY_PLAN_SCHEMA_VERSION,
        "query_record": query_record.to_dict(),
        "retrieval_queries": [query.to_dict() for query in queries],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
