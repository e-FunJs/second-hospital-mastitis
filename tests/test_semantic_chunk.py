"""semantic_chunk 模块测试。

这些测试只验证不依赖 GPU 的核心切分逻辑，真实 BGE-M3 加载由脚本 smoke test 覆盖。
"""

from __future__ import annotations

from rag_medical.semantic_chunk import (
    ChunkConfig,
    build_chunk_records,
    choose_semantic_boundaries,
    split_sentences,
)


def test_split_sentences_keeps_common_medical_abbreviations_together() -> None:
    text = "Patients received antibiotics, e.g. cefuroxime. Symptoms improved. Recurrence was rare."

    assert split_sentences(text) == [
        "Patients received antibiotics, e.g. cefuroxime.",
        "Symptoms improved.",
        "Recurrence was rare.",
    ]


def test_choose_semantic_boundaries_prefers_local_similarity_valleys() -> None:
    sentences = [f"Sentence {index}." for index in range(1, 7)]
    similarities = [0.91, 0.88, 0.42, 0.89, 0.86]
    config = ChunkConfig(min_sentences=2, max_sentences=4, similarity_percentile=35.0)

    assert choose_semantic_boundaries(sentences, similarities, config) == [3]


def test_build_chunk_records_preserves_article_metadata_and_source_paragraphs() -> None:
    records = [
        {
            "pmcid": "PMC1",
            "pmid": "1",
            "doi": "10.1/example",
            "title": "Example study",
            "journal": "Example Journal",
            "year": "2026",
            "source_path": "data/articles/raw/pmc_xml/PMC1.xml",
            "section": "Results",
            "paragraph_index": 1,
            "text": "Inflammation improved after treatment. Ultrasound showed smaller lesions.",
        },
        {
            "pmcid": "PMC1",
            "pmid": "1",
            "doi": "10.1/example",
            "title": "Example study",
            "journal": "Example Journal",
            "year": "2026",
            "source_path": "data/articles/raw/pmc_xml/PMC1.xml",
            "section": "Results",
            "paragraph_index": 2,
            "text": "Relapse occurred in one patient. Steroid tapering was extended.",
        },
    ]

    chunks = build_chunk_records(records, boundary_after_sentence_indices=[2])

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "PMC1::Results::001"
    assert chunks[0]["source_paragraph_indices"] == [1]
    assert chunks[0]["text"] == "Inflammation improved after treatment. Ultrasound showed smaller lesions."
    assert chunks[1]["source_paragraph_indices"] == [2]
    assert chunks[1]["title"] == "Example study"
