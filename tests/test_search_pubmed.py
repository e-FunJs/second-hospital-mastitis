"""PubMed query config loading tests."""

from __future__ import annotations

from pathlib import Path

from rag_medical.search_pubmed import load_pubmed_queries


def test_load_pubmed_queries_includes_treatment_expansion_queries() -> None:
    queries = load_pubmed_queries(Path('configs/queries.yaml'))

    assert 'rifampin' in queries
    assert 'duration' in queries
    assert 'drug_therapy' in queries
    assert 'periductal_tx' in queries
    assert 'rifampicin' in queries['rifampin']
    assert 'duration' in queries['duration']
    assert 'azathioprine' in queries['drug_therapy']
    assert 'periductal mastitis' in queries['periductal_tx']


def test_load_pubmed_queries_includes_hospital_disease_synonyms() -> None:
    queries = load_pubmed_queries(Path("configs/queries.yaml"))

    expected_terms = [
        "granulomatous lobular mastitis",
        "granulomatous mastitis",
        "periductal mastitis",
        "mammary duct ectasia",
        "non-puerperal mastitis",
        "plasma cell mastitis",
        "non-lactational mastitis",
    ]

    assert "disease_synonyms" in queries
    disease_query_keys = [
        "disease_synonyms",
        "core_english",
        "treatment_outcome",
        "ultrasound",
        "therapies",
        "rifampin",
        "duration",
        "drug_therapy",
    ]
    for key in disease_query_keys:
        for term in expected_terms:
            assert term in queries[key].lower()
