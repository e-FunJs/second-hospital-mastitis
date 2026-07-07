"""PubMed query config loading tests."""

from __future__ import annotations

from rag_medical.search_pubmed import load_pubmed_queries


def test_load_pubmed_queries_includes_treatment_expansion_queries() -> None:
    queries = load_pubmed_queries(__import__('pathlib').Path('configs/queries.yaml'))

    assert 'rifampin' in queries
    assert 'duration' in queries
    assert 'drug_therapy' in queries
    assert 'periductal_tx' in queries
    assert 'rifampicin' in queries['rifampin']
    assert 'duration' in queries['duration']
    assert 'azathioprine' in queries['drug_therapy']
    assert 'periductal mastitis' in queries['periductal_tx']
