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


def test_load_pubmed_queries_includes_targeted_anti_tubercular_drug_terms() -> None:
    queries = load_pubmed_queries(Path("configs/queries.yaml"))

    assert "anti_tb_drugs" in queries
    query = queries["anti_tb_drugs"].lower()

    for drug_term in ["ethambutol", "ethylaminobutanol", "isoniazid"]:
        assert drug_term in query

    for disease_term in [
        "granulomatous mastitis",
        "non-puerperal mastitis",
        "periductal mastitis",
        "plasma cell mastitis",
        "mammary duct ectasia",
        "tuberculous mastitis",
        "tubercular mastitis",
        "breast tuberculosis",
        "mammary tuberculosis",
    ]:
        assert disease_term in query

    for treatment_term in ["treatment", "therapy"]:
        assert treatment_term in query


def test_load_pubmed_queries_includes_anti_tubercular_combination_terms() -> None:
    queries = load_pubmed_queries(Path("configs/queries.yaml"))

    assert "anti_tb_combination" in queries
    query = queries["anti_tb_combination"].lower()

    for drug_term in ["rifampin", "rifampicin", "isoniazid", "ethambutol"]:
        assert drug_term in query

    for disease_term in [
        "granulomatous mastitis",
        "non-puerperal mastitis",
        "periductal mastitis",
        "plasma cell mastitis",
        "tuberculous mastitis",
        "tubercular mastitis",
        "breast tuberculosis",
        "mammary tuberculosis",
    ]:
        assert disease_term in query

    for regimen_term in ["combination therapy", "triple therapy", "regimen"]:
        assert regimen_term in query


def test_load_pubmed_queries_includes_recurrent_and_combination_drug_expansion() -> None:
    queries = load_pubmed_queries(Path("configs/queries.yaml"))

    expected = {
        "recurrent_drug_tx": [
            "recurrent mastitis",
            "recurrent granulomatous mastitis",
            "refractory granulomatous mastitis",
            "relapsed granulomatous mastitis",
            "drug therapy",
            "medical treatment",
        ],
        "combination_drug_tx": [
            "mastitis",
            "granulomatous mastitis",
            "non-puerperal mastitis",
            "combination therapy",
            "combined drug therapy",
            "multidrug therapy",
        ],
        "steroid_mtx_combo_tx": [
            "granulomatous mastitis",
            "corticosteroid",
            "methotrexate",
            "prednisone",
            "prednisolone",
            "steroid-sparing therapy",
        ],
        "corynebacterium_antibiotic_tx": [
            "corynebacterium",
            "corynebacterium kroppenstedtii",
            "granulomatous mastitis",
            "antibiotic therapy",
            "recurrent granulomatous mastitis",
        ],
    }

    for query_name, expected_terms in expected.items():
        assert query_name in queries
        query = queries[query_name].lower()
        for term in expected_terms:
            assert term in query
