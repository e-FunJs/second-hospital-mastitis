"""build_embeddings 模块测试。

这些测试只覆盖不依赖 BGE/GPU 的数据整理与落盘逻辑；真实模型 encode 由脚本运行验证。
"""

from __future__ import annotations

import json

import numpy as np

from rag_medical.build_embeddings import (
    build_embedding_text,
    build_metadata_record,
    write_embedding_outputs,
)


def test_build_embedding_text_uses_semantic_fields_only() -> None:
    chunk = {
        "chunk_id": "PMC1::Treatment::001",
        "pmcid": "PMC1",
        "doi": "10.1/example",
        "title": "Granulomatous mastitis treatment study",
        "section": "Treatment > Corticosteroids",
        "text": "Steroid therapy reduced lesion size and pain.",
    }

    text = build_embedding_text(chunk)

    assert "Title: Granulomatous mastitis treatment study" in text
    assert "Section: Treatment > Corticosteroids" in text
    assert "Text: Steroid therapy reduced lesion size and pain." in text
    assert "PMC1" not in text
    assert "10.1/example" not in text


def test_build_metadata_record_preserves_row_alignment_and_source_fields() -> None:
    chunk = {
        "chunk_id": "PMC1::Abstract::001",
        "pmcid": "PMC1",
        "pmid": "123",
        "doi": "10.1/example",
        "title": "Example",
        "journal": "Journal",
        "year": "2026",
        "section": "Abstract",
        "text": "Example chunk text.",
        "source_paragraph_indices": [1, 2],
    }

    metadata = build_metadata_record(chunk, row_index=7)

    assert metadata["row_index"] == 7
    assert metadata["chunk_id"] == "PMC1::Abstract::001"
    assert metadata["source_paragraph_indices"] == [1, 2]
    assert metadata["text"] == "Example chunk text."


def test_write_embedding_outputs_keeps_numpy_and_metadata_order(tmp_path) -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metadata = [
        {"row_index": 0, "chunk_id": "chunk-a", "text": "A"},
        {"row_index": 1, "chunk_id": "chunk-b", "text": "B"},
    ]
    embedding_path = tmp_path / "chunk_embeddings.npy"
    metadata_path = tmp_path / "chunk_metadata.jsonl"
    manifest_path = tmp_path / "embedding_manifest.json"

    write_embedding_outputs(
        embeddings=embeddings,
        metadata_records=metadata,
        embedding_path=embedding_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        model_path="models/bge/bge-m3",
        input_path="data/articles/processed/article_chunks.jsonl",
        batch_size=16,
        device="cuda",
    )

    loaded = np.load(embedding_path)
    metadata_lines = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())

    assert loaded.shape == (2, 2)
    assert loaded.dtype == np.float32
    assert metadata_lines[1]["chunk_id"] == "chunk-b"
    assert manifest["chunk_count"] == 2
    assert manifest["embedding_dim"] == 2
    assert manifest["model_path"] == "models/bge/bge-m3"
