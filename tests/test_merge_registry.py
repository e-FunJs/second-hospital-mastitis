"""文献 registry 合并测试。"""

from __future__ import annotations

import csv

from rag_medical.merge_registry import discover_source_files, merge_rows


def write_csv(path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pmid", "pmcid", "title", "abstract"])
        writer.writeheader()
        writer.writerows(rows)


def test_discover_source_files_prefers_pubmed_query_outputs(tmp_path) -> None:
    write_csv(tmp_path / "pubmed_rifampin.csv", [])
    write_csv(tmp_path / "pubmed_duration.csv", [])
    write_csv(tmp_path / "core.csv", [])

    sources = discover_source_files(tmp_path)

    assert list(sources) == ["duration", "rifampin"]


def test_merge_rows_includes_new_pubmed_query_files(tmp_path) -> None:
    write_csv(
        tmp_path / "pubmed_rifampin.csv",
        [{"pmid": "1", "pmcid": "", "title": "Rifampicin mastitis", "abstract": "A"}],
    )
    write_csv(
        tmp_path / "pubmed_duration.csv",
        [{"pmid": "2", "pmcid": "PMC2", "title": "Treatment duration", "abstract": "B"}],
    )

    rows = merge_rows(tmp_path)

    assert {row["pmid"] for row in rows} == {"1", "2"}
    assert {row["source_queries"] for row in rows} == {"rifampin", "duration"}


def test_merge_rows_keeps_all_matching_sources_for_duplicates(tmp_path) -> None:
    write_csv(tmp_path / "pubmed_rifampin.csv", [{"pmid": "1", "pmcid": "", "title": "Same", "abstract": "A"}])
    write_csv(tmp_path / "pubmed_drug_therapy.csv", [{"pmid": "1", "pmcid": "", "title": "Same", "abstract": ""}])

    rows = merge_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["source_queries"] == "drug_therapy;rifampin"
    assert rows[0]["source_query_count"] == "2"
