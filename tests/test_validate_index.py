"""索引结构验证器测试。"""

from __future__ import annotations

import faiss
import numpy as np

from rag_medical.common.build_faiss_index import build_faiss_index
from rag_medical.common.validate_index import validate_index_artifacts


def metadata_record(row_index: int, chunk_id: str) -> dict:
    return {
        "row_index": row_index,
        "chunk_id": chunk_id,
        "language": "zh",
        "source_type": "cnki_pdf",
        "title": "非哺乳期乳腺炎治疗研究",
        "source_path": f"data/articles/raw/chinese/cnki_pdf/{chunk_id}.pdf",
        "source_pages": [row_index + 1],
        "text": "患者治疗后病灶缩小。",
    }


def manifest(count: int, dimension: int) -> dict:
    return {"chunk_count": count, "embedding_dim": dimension}


def test_valid_index_artifacts_are_ready_for_retrieval() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    metadata = [metadata_record(index, f"chunk-{index}") for index in range(3)]
    index = build_faiss_index(embeddings)

    report = validate_index_artifacts(
        embeddings,
        metadata,
        index,
        manifest(3, 3),
        manifest(3, 3),
        expected_language="zh",
    )

    assert report["ready_for_retrieval"] is True
    assert report["error_count"] == 0
    assert report["sample_reconstruction_max_error"] == 0.0


def test_misaligned_index_and_metadata_are_rejected() -> None:
    embeddings = np.eye(2, dtype=np.float32)
    metadata = [metadata_record(0, "chunk-0")]
    index = faiss.IndexFlatIP(2)
    index.add(embeddings[:1])

    report = validate_index_artifacts(
        embeddings,
        metadata,
        index,
        manifest(2, 2),
        manifest(2, 2),
        expected_language="zh",
    )

    codes = {error["code"] for error in report["errors"]}
    assert report["ready_for_retrieval"] is False
    assert "embedding_metadata_count_mismatch" in codes
    assert "faiss_count_mismatch" in codes
