"""FAISS 建索引与检索模块测试。

测试只使用小型 toy embedding，验证向量行号和 metadata 行号不会错位。
"""

from __future__ import annotations

import json

import numpy as np

from rag_medical.build_faiss_index import (
    build_faiss_index,
    validate_embeddings_and_metadata,
    write_faiss_manifest,
)
from rag_medical.search_chunks import build_search_results, search_index


def test_validate_embeddings_and_metadata_rejects_row_mismatch() -> None:
    embeddings = np.zeros((2, 3), dtype=np.float32)
    metadata = [{"row_index": 0, "chunk_id": "a"}]

    try:
        validate_embeddings_and_metadata(embeddings, metadata)
    except ValueError as exc:
        assert "metadata rows" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_faiss_index_and_search_returns_expected_metadata_order() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"row_index": 0, "chunk_id": "chunk-x", "title": "X", "section": "A", "text": "alpha"},
        {"row_index": 1, "chunk_id": "chunk-y", "title": "Y", "section": "B", "text": "beta"},
        {"row_index": 2, "chunk_id": "chunk-z", "title": "Z", "section": "C", "text": "gamma"},
    ]

    index = build_faiss_index(embeddings)
    hits = search_index(index, np.array([0.0, 1.0, 0.0], dtype=np.float32), top_k=2)
    results = build_search_results(hits, metadata)

    assert index.ntotal == 3
    assert results[0]["chunk_id"] == "chunk-y"
    assert results[0]["row_index"] == 1
    assert results[0]["score"] > results[1]["score"]


def test_write_faiss_manifest_records_index_shape(tmp_path) -> None:
    manifest_path = tmp_path / "faiss_manifest.json"

    write_faiss_manifest(
        manifest_path=manifest_path,
        embedding_path="data/index/chunk_embeddings.npy",
        metadata_path="data/index/chunk_metadata.jsonl",
        index_path="data/index/faiss.index",
        chunk_count=3,
        embedding_dim=1024,
        metric="inner_product_cosine",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["chunk_count"] == 3
    assert manifest["embedding_dim"] == 1024
    assert manifest["metric"] == "inner_product_cosine"
