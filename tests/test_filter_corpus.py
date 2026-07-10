"""Strict medical corpus filtering tests."""

from __future__ import annotations

from rag_medical.filter_corpus import Decision, classify_record, filter_chunks


def test_classify_record_includes_human_recurrent_granulomatous_mastitis_treatment() -> None:
    record = {
        "title": "Refractory and Recurrent Idiopathic Granulomatous Mastitis Treatment",
        "abstract": "Patients with idiopathic granulomatous mastitis received methotrexate and corticosteroid therapy; recurrence and remission were evaluated.",
        "journal": "Breast Care",
    }

    decision = classify_record(record)

    assert decision.decision == "include"
    assert "granulomatous mastitis" in decision.include_matches
    assert "methotrexate" in decision.include_matches
    assert decision.include_score > decision.exclude_score


def test_classify_record_includes_breast_tuberculosis_antitubercular_treatment() -> None:
    record = {
        "title": "Tuberculous mastitis treated with rifampicin isoniazid and ethambutol",
        "abstract": "A human breast tuberculosis case received anti-tuberculosis combination therapy and follow-up.",
        "journal": "International Journal of Surgery Case Reports",
    }

    decision = classify_record(record)

    assert decision.decision == "include"
    assert "tuberculous mastitis" in decision.include_matches
    assert "rifampicin" in decision.include_matches
    assert "ethambutol" in decision.include_matches


def test_classify_record_excludes_veterinary_bovine_mastitis() -> None:
    record = {
        "title": "Combination antimicrobial treatment for bovine mastitis in dairy cows",
        "abstract": "Milk yield, udder infection, somatic cell count, and herd management were measured after intramammary treatment.",
        "journal": "Veterinary Microbiology",
    }

    decision = classify_record(record)

    assert decision.decision == "exclude"
    assert "bovine" in decision.exclude_matches
    assert "dairy cow" in decision.exclude_matches
    assert "udder" in decision.exclude_matches


def test_classify_record_excludes_lactational_only_mastitis_without_target_terms() -> None:
    record = {
        "title": "Antibiotics for acute lactational mastitis during breastfeeding",
        "abstract": "This review discusses postpartum lactation, breastfeeding support, milk drainage, and maternal fever.",
        "journal": "Family Practice",
    }

    decision = classify_record(record)

    assert decision.decision == "exclude"
    assert "lactational mastitis" in decision.exclude_matches
    assert "breastfeeding" in decision.exclude_matches


def test_classify_record_sends_breast_inflammation_without_treatment_signal_to_review() -> None:
    record = {
        "title": "Inflammatory breast disease and benign breast disorders",
        "abstract": "The article discusses breast inflammation and imaging differential diagnosis without specific therapy details.",
        "journal": "Radiology Review",
    }

    decision = classify_record(record)

    assert decision.decision == "review"
    assert decision.review_reason


def test_filter_chunks_keeps_only_included_article_and_targeted_chunk() -> None:
    article_decisions = {
        "100": Decision(
            decision="include",
            include_score=8,
            exclude_score=0,
            include_matches=["granulomatous mastitis", "methotrexate"],
            exclude_matches=[],
            review_reason="",
        ),
        "200": Decision(
            decision="exclude",
            include_score=1,
            exclude_score=10,
            include_matches=["mastitis"],
            exclude_matches=["bovine", "dairy cow"],
            review_reason="animal mastitis",
        ),
    }
    chunks = [
        {
            "pmid": "100",
            "title": "Granulomatous mastitis treatment",
            "section": "Treatment",
            "text": "Methotrexate and prednisolone reduced recurrence in granulomatous mastitis.",
        },
        {
            "pmid": "100",
            "title": "Granulomatous mastitis treatment",
            "section": "Methods",
            "text": "The ultrasound machine model was recorded.",
        },
        {
            "pmid": "200",
            "title": "Bovine mastitis treatment",
            "section": "Treatment",
            "text": "Dairy cows received intramammary antimicrobial treatment.",
        },
    ]

    included, review, excluded = filter_chunks(chunks, article_decisions)

    assert len(included) == 1
    assert included[0]["pmid"] == "100"
    assert included[0]["filter_decision"] == "include"
    assert included[0]["filter_level"] == "article_and_chunk"
    assert len(review) == 1
    assert len(excluded) == 1


def test_filter_chunks_sends_animal_noise_inside_included_article_to_review() -> None:
    article_decisions = {
        "100": Decision(
            decision="include",
            include_score=8,
            exclude_score=0,
            include_matches=["granulomatous mastitis", "methotrexate"],
            exclude_matches=[],
            review_reason="",
        )
    }
    chunks = [
        {
            "pmid": "100",
            "section": "Discussion",
            "text": "Granulomatous mastitis treatment was discussed using a bovine serum model.",
        }
    ]

    included, review, excluded = filter_chunks(chunks, article_decisions)

    assert included == []
    assert excluded == []
    assert len(review) == 1
    assert review[0]["filter_level"] == "chunk_noise_review"
