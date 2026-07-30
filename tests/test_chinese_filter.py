"""中文医学语料筛选规则测试。"""

from __future__ import annotations

from rag_medical.chinese.filter_corpus import classify_record, filter_chunks, split_registry


def test_classify_target_combination_drug_treatment_as_include() -> None:
    record = {
        "document_id": "CNKI-1",
        "title": "异烟肼、利福平和乙胺丁醇联合治疗非哺乳期乳腺炎的疗效",
    }

    decision = classify_record(record)

    assert decision.decision == "include"
    assert "非哺乳期乳腺炎" in decision.include_matches
    assert "利福平" in decision.include_matches


def test_classify_lactational_only_article_as_excluded() -> None:
    decision = classify_record(
        {
            "document_id": "CNKI-2",
            "title": "哺乳期乳腺炎患者母乳喂养管理",
        }
    )

    assert decision.decision == "exclude"
    assert "哺乳期乳腺炎" in decision.exclude_matches


def test_classify_animal_mastitis_as_excluded() -> None:
    decision = classify_record(
        {
            "document_id": "CNKI-3",
            "title": "奶牛乳腺炎与体细胞数变化研究",
        }
    )

    assert decision.decision == "exclude"
    assert "奶牛乳腺炎" in decision.exclude_matches


def test_included_article_keeps_methods_chunk_without_repeated_disease_name() -> None:
    registry = [
        {
            "document_id": "CNKI-1",
            "title": "非哺乳期乳腺炎联合治疗的临床研究",
        }
    ]
    _, _, _, decisions = split_registry(registry)
    chunks = [
        {
            "chunk_id": "CNKI-1::SEC::001",
            "document_id": "CNKI-1",
            "section": "资料与方法",
            "text": "纳入符合标准的患者，记录年龄和实验室指标。",
        }
    ]

    strict, review, excluded = filter_chunks(chunks, decisions)

    assert len(strict) == 1
    assert review == []
    assert excluded == []
    assert strict[0]["filter_level"] == "article"
