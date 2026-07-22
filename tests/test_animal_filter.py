"""动物乳腺炎 BGE 二次筛选的核心规则测试。"""

from __future__ import annotations

import numpy as np

from rag_medical.animal_rules import (
    ChunkScore,
    SemanticConfig,
    decide_article,
    extract_anchors,
    top_k_mean_similarity,
)


def animal_score(value: float = 0.70, human: float = 0.30, lexical: bool = True) -> ChunkScore:
    return ChunkScore(
        animal_score=value,
        human_score=human,
        margin=value - human,
        is_animal=True,
        animal_terms=("bovine",) if lexical else (),
    )


def human_score(value: float = 0.35, human: float = 0.72) -> ChunkScore:
    return ChunkScore(
        animal_score=value,
        human_score=human,
        margin=value - human,
        is_animal=False,
        animal_terms=(),
    )


def test_extract_animal_anchors_uses_real_corpus_sentences() -> None:
    chunks = [
        {
            "chunk_id": "PMC1::intro::1",
            "pmcid": "PMC1",
            "title": "Bovine mastitis study",
            "text": (
                "Bovine mastitis is a major disease affecting dairy cows and milk production. "
                "This unrelated short sentence should not become an anchor."
            ),
        }
    ]

    anchors = extract_anchors(chunks, "animal", max_anchors=10)

    assert len(anchors) == 1
    assert anchors[0].source_id == "pmcid:PMC1"
    assert anchors[0].text.startswith("Bovine mastitis")
    assert "bovine" in anchors[0].matched_terms


def test_top_k_mean_similarity_uses_multiple_nearest_anchors() -> None:
    texts = np.asarray([[1.0, 0.0]], dtype=np.float32)
    anchors = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)

    scores = top_k_mean_similarity(texts, anchors, top_k=2)

    assert np.isclose(scores[0], 0.9)


def test_article_over_forty_percent_animal_chunks_is_excluded() -> None:
    config = SemanticConfig(article_animal_ratio=0.40, min_animal_hits=2)
    scores = [animal_score(), animal_score(), animal_score(), human_score(), human_score()]

    decision = decide_article("strict", "Mastitis treatment", scores, config)

    assert decision.decision == "exclude_animal"
    assert decision.final_status == "excluded"
    assert decision.animal_ratio == 0.6


def test_exactly_forty_percent_is_not_directly_excluded() -> None:
    config = SemanticConfig(article_animal_ratio=0.40, min_animal_hits=2)
    scores = [animal_score(), animal_score(), human_score(), human_score(), human_score()]

    decision = decide_article("strict", "Mastitis treatment", scores, config)

    assert decision.decision == "review_animal"
    assert decision.final_status == "review"


def test_single_abstract_requires_strong_semantic_and_lexical_evidence() -> None:
    config = SemanticConfig(single_chunk_threshold=0.60)

    strong = decide_article(
        "review", "Bovine mastitis treatment", [animal_score(value=0.66)], config
    )
    no_lexical = decide_article(
        "review", "Mastitis treatment", [animal_score(value=0.66, lexical=False)], config
    )

    assert strong.final_status == "excluded"
    assert no_lexical.final_status == "review"


def test_explicit_mouse_model_title_is_excluded_after_bge_confirmation() -> None:
    config = SemanticConfig()
    scores = [animal_score(), human_score(), human_score(), human_score()]

    decision = decide_article("strict", "Mouse model of plasma cell mastitis", scores, config)

    assert decision.decision == "exclude_animal"
    assert decision.final_status == "excluded"


def test_human_title_is_protected_from_incidental_animal_references() -> None:
    config = SemanticConfig(article_animal_ratio=0.40)
    scores = [animal_score(), animal_score(), animal_score(), human_score(), human_score()]

    decision = decide_article(
        "review",
        "Staphylococcus aureus isolated from human milk of women with acute mastitis",
        scores,
        config,
    )

    assert decision.decision == "keep"
    assert decision.final_status == "review"


def test_one_incidental_animal_chunk_does_not_remove_long_human_article() -> None:
    config = SemanticConfig(article_animal_ratio=0.40, review_animal_ratio=0.20)
    scores = [animal_score()] + [human_score() for _ in range(9)]

    decision = decide_article("strict", "Human granulomatous mastitis treatment", scores, config)

    assert decision.decision == "keep"
    assert decision.final_status == "strict"
