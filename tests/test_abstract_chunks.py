"""摘要型知识源 chunk 测试。"""

from __future__ import annotations

import csv
import json

from rag_medical.abstract_chunks import build_abstract_chunks, registry_rows_to_chunks
from rag_medical.combine_chunks import combine_chunk_files


def test_registry_rows_to_chunks_skips_full_text_and_empty_abstracts() -> None:
    rows = [
        {
            "pmid": "1001",
            "pmcid": "",
            "doi": "10.1/a",
            "title": "Rifampicin therapy for mastitis",
            "journal": "Breast Journal",
            "year": "2024",
            "abstract": "Rifampicin was used with other antibiotics. Symptoms improved after treatment.",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/1001/",
        },
        {
            "pmid": "1002",
            "pmcid": "PMC1002",
            "title": "Full text article",
            "abstract": "This record has full text and should stay in the PMC XML branch.",
        },
        {"pmid": "1003", "title": "No abstract", "abstract": ""},
    ]

    chunks = registry_rows_to_chunks(rows)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "PMID1001::Abstract::001"
    assert chunks[0]["source_type"] == "pubmed_abstract"
    assert chunks[0]["section"] == "Abstract"
    assert "Rifampicin" in chunks[0]["text"]


def test_build_abstract_chunks_reads_registry_csv(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pmid", "pmcid", "title", "abstract"])
        writer.writeheader()
        writer.writerow(
            {
                "pmid": "2001",
                "pmcid": "",
                "title": "Treatment duration",
                "abstract": "Treatment duration varied across patients and recurrence was recorded.",
            }
        )

    out = tmp_path / "abstract_chunks.jsonl"
    count = build_abstract_chunks(registry, out)

    assert count == 1
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert records[0]["pmid"] == "2001"
    assert records[0]["source_type"] == "pubmed_abstract"


def test_combine_chunk_files_preserves_order_and_source_type(tmp_path) -> None:
    full_text = tmp_path / "article_chunks.jsonl"
    abstracts = tmp_path / "abstract_chunks.jsonl"
    out = tmp_path / "rag_chunks.jsonl"
    full_text.write_text(json.dumps({"chunk_id": "PMC1::Body::001", "text": "Full text"}) + "\n", encoding="utf-8")
    abstracts.write_text(
        json.dumps({"chunk_id": "PMID2::Abstract::001", "source_type": "pubmed_abstract", "text": "Abstract"}) + "\n",
        encoding="utf-8",
    )

    count = combine_chunk_files([full_text, abstracts], out)

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert records[0]["chunk_id"] == "PMC1::Body::001"
    assert records[0]["source_type"] == "pmc_full_text"
    assert records[1]["chunk_id"] == "PMID2::Abstract::001"
