"""
用途：验证 embedding、metadata 与 FAISS 索引是否严格一致。

默认输入：
    data/index/chinese/strict/chunk_embeddings.npy
    data/index/chinese/strict/chunk_metadata.jsonl
    data/index/chinese/strict/faiss.index
    data/index/chinese/strict/embedding_manifest.json
    data/index/chinese/strict/faiss_manifest.json

默认输出：
    data/index/chinese/strict/index_validation_report.json

检查内容：
    - embedding 数量、维度、有限值和 L2 归一化；
    - metadata 行号、chunk_id、正文与中文页码追溯字段；
    - FAISS 的向量数、维度以及抽样重建向量；
    - 两份 manifest 记录的数量和维度。

说明：
    本模块不加载 BGE，不重新生成 embedding，也不执行医学问答。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag_medical.common.build_faiss_index import load_embeddings, read_metadata_jsonl


def add_issue(issues: list[dict[str, Any]], code: str, detail: Any) -> None:
    """使用稳定的机器可读格式记录错误或提醒。"""

    issues.append({"code": code, "detail": detail})


def sample_row_indices(row_count: int) -> list[int]:
    if row_count <= 0:
        return []
    return sorted({0, row_count // 2, row_count - 1})


def validate_metadata(
    metadata_records: list[dict[str, Any]],
    expected_language: str | None,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> tuple[Counter[str], Counter[str]]:
    chunk_ids: set[str] = set()
    language_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()

    for expected_row, record in enumerate(metadata_records):
        if record.get("row_index") != expected_row:
            add_issue(
                errors,
                "metadata_row_index_mismatch",
                {
                    "expected": expected_row,
                    "actual": record.get("row_index"),
                },
            )

        chunk_id = str(record.get("chunk_id") or "")
        if not chunk_id:
            add_issue(errors, "missing_chunk_id", {"row_index": expected_row})
        elif chunk_id in chunk_ids:
            add_issue(errors, "duplicate_chunk_id", chunk_id)
        else:
            chunk_ids.add(chunk_id)

        if not str(record.get("text") or "").strip():
            add_issue(errors, "empty_chunk_text", chunk_id or expected_row)

        language = str(record.get("language") or "")
        source_type = str(record.get("source_type") or "")
        language_counts[language] += 1
        source_type_counts[source_type] += 1

        if expected_language and language != expected_language:
            add_issue(
                errors,
                "unexpected_language",
                {
                    "chunk_id": chunk_id,
                    "expected": expected_language,
                    "actual": language,
                },
            )

        # 中文 PDF 的检索结果必须能回到原文页，不能只剩一段脱离来源的文字。
        if expected_language == "zh":
            if not str(record.get("source_path") or ""):
                add_issue(errors, "missing_source_path", chunk_id)
            if not record.get("source_pages"):
                add_issue(errors, "missing_source_pages", chunk_id)

        if not str(record.get("title") or ""):
            add_issue(warnings, "missing_title", chunk_id or expected_row)

    return language_counts, source_type_counts


def validate_manifests(
    embedding_manifest: dict[str, Any],
    faiss_manifest: dict[str, Any],
    row_count: int,
    embedding_dim: int,
    errors: list[dict[str, Any]],
) -> None:
    for name, manifest in (
        ("embedding_manifest", embedding_manifest),
        ("faiss_manifest", faiss_manifest),
    ):
        if int(manifest.get("chunk_count", -1)) != row_count:
            add_issue(
                errors,
                f"{name}_chunk_count_mismatch",
                {
                    "expected": row_count,
                    "actual": manifest.get("chunk_count"),
                },
            )
        if int(manifest.get("embedding_dim", -1)) != embedding_dim:
            add_issue(
                errors,
                f"{name}_dimension_mismatch",
                {
                    "expected": embedding_dim,
                    "actual": manifest.get("embedding_dim"),
                },
            )


def validate_index_artifacts(
    embeddings: np.ndarray,
    metadata_records: list[dict[str, Any]],
    index: faiss.Index,
    embedding_manifest: dict[str, Any],
    faiss_manifest: dict[str, Any],
    expected_language: str | None = None,
) -> dict[str, Any]:
    """验证已经加载到内存的五类索引对象。"""

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    row_count = int(embeddings.shape[0]) if embeddings.ndim == 2 else 0
    embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0

    if embeddings.ndim != 2:
        add_issue(errors, "embedding_not_2d", list(embeddings.shape))
    if row_count == 0:
        add_issue(errors, "empty_embeddings", row_count)
    if row_count != len(metadata_records):
        add_issue(
            errors,
            "embedding_metadata_count_mismatch",
            {
                "embeddings": row_count,
                "metadata": len(metadata_records),
            },
        )
    if not np.isfinite(embeddings).all():
        add_issue(errors, "non_finite_embedding_values", True)

    norms = np.linalg.norm(embeddings, axis=1) if row_count else np.array([], dtype=np.float32)
    if norms.size and not np.allclose(norms, 1.0, atol=1e-3):
        add_issue(
            errors,
            "embedding_norm_mismatch",
            {
                "minimum": float(norms.min()),
                "maximum": float(norms.max()),
            },
        )

    language_counts, source_type_counts = validate_metadata(
        metadata_records,
        expected_language,
        errors,
        warnings,
    )

    if int(index.ntotal) != row_count:
        add_issue(
            errors,
            "faiss_count_mismatch",
            {"expected": row_count, "actual": int(index.ntotal)},
        )
    if int(index.d) != embedding_dim:
        add_issue(
            errors,
            "faiss_dimension_mismatch",
            {"expected": embedding_dim, "actual": int(index.d)},
        )

    reconstruction_errors: list[float] = []
    if index.ntotal == row_count and index.d == embedding_dim:
        for row_index in sample_row_indices(row_count):
            expected = embeddings[row_index].astype(np.float32, copy=True)
            expected /= max(float(np.linalg.norm(expected)), 1e-12)
            reconstructed = np.asarray(index.reconstruct(row_index), dtype=np.float32)
            reconstruction_errors.append(float(np.max(np.abs(expected - reconstructed))))
        if reconstruction_errors and max(reconstruction_errors) > 1e-4:
            add_issue(
                errors,
                "faiss_reconstruction_mismatch",
                max(reconstruction_errors),
            )

    validate_manifests(
        embedding_manifest,
        faiss_manifest,
        row_count,
        embedding_dim,
        errors,
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": row_count,
        "embedding_dim": embedding_dim,
        "embedding_dtype": str(embeddings.dtype),
        "embedding_norm_min": float(norms.min()) if norms.size else None,
        "embedding_norm_max": float(norms.max()) if norms.size else None,
        "faiss_index_type": type(index).__name__,
        "faiss_ntotal": int(index.ntotal),
        "faiss_dimension": int(index.d),
        "sample_reconstruction_max_error": (
            max(reconstruction_errors) if reconstruction_errors else None
        ),
        "language_counts": dict(language_counts),
        "source_type_counts": dict(source_type_counts),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "ready_for_retrieval": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_index_files(
    embedding_path: Path,
    metadata_path: Path,
    index_path: Path,
    embedding_manifest_path: Path,
    faiss_manifest_path: Path,
    expected_language: str | None,
) -> dict[str, Any]:
    embeddings = load_embeddings(embedding_path)
    metadata_records = read_metadata_jsonl(metadata_path)
    index = faiss.read_index(str(index_path))
    embedding_manifest = read_json(embedding_manifest_path)
    faiss_manifest = read_json(faiss_manifest_path)
    return validate_index_artifacts(
        embeddings,
        metadata_records,
        index,
        embedding_manifest,
        faiss_manifest,
        expected_language,
    )


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate embedding, metadata and FAISS alignment."
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data/index/chinese/strict"))
    parser.add_argument("--expected-language")
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report or args.index_dir / "index_validation_report.json"
    required_paths = {
        "embeddings": args.index_dir / "chunk_embeddings.npy",
        "metadata": args.index_dir / "chunk_metadata.jsonl",
        "index": args.index_dir / "faiss.index",
        "embedding_manifest": args.index_dir / "embedding_manifest.json",
        "faiss_manifest": args.index_dir / "faiss_manifest.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        print(f"missing index files: {missing}", file=sys.stderr)
        return 2

    report = validate_index_files(
        required_paths["embeddings"],
        required_paths["metadata"],
        required_paths["index"],
        required_paths["embedding_manifest"],
        required_paths["faiss_manifest"],
        args.expected_language,
    )
    write_report(report_path, report)

    print(f"chunks={report['chunk_count']}")
    print(f"embedding_dim={report['embedding_dim']}")
    print(f"errors={report['error_count']}")
    print(f"warnings={report['warning_count']}")
    print(f"ready_for_retrieval={report['ready_for_retrieval']}")
    print(f"report={report_path}")
    return 0 if report["ready_for_retrieval"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
