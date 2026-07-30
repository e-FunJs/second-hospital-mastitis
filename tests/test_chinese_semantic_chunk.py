"""中文章节识别、切句和 chunk 构造测试。"""

from __future__ import annotations

from rag_medical.chinese.semantic_chunk import (
    ChunkConfig,
    build_chunk_records,
    canonical_section_heading,
    choose_semantic_boundaries,
    section_sentence_items,
    split_sentences,
)


def test_split_sentences_handles_chinese_medical_text() -> None:
    text = "患者使用利福平和异烟肼。治疗三个月后肿块缩小；随访期间未见复发！"

    assert split_sentences(text) == [
        "患者使用利福平和异烟肼。",
        "治疗三个月后肿块缩小；",
        "随访期间未见复发！",
    ]


def test_split_sentences_limits_unpunctuated_long_text() -> None:
    sentences = split_sentences("clinical treatment " * 400)

    assert len(sentences) > 1
    assert max(map(len, sentences)) <= 1800


def test_split_sentences_limits_chinese_text_units() -> None:
    sentences = split_sentences("乳腺炎治疗" * 400)

    assert len(sentences) > 1
    assert max(map(len, sentences)) <= 850


def test_section_heading_recognizes_numbered_chinese_heading() -> None:
    assert canonical_section_heading("2.1 治疗方法") == "治疗方法"
    assert canonical_section_heading("结 果") == "结果"
    assert canonical_section_heading("资料和方法") == "资料与方法"
    assert canonical_section_heading("1 仪器与方法") == "资料与方法"
    assert canonical_section_heading("1. 克氏棒状杆菌药敏分析") is None
    assert canonical_section_heading("1.3 纳入与排除标准") == "纳入与排除标准"
    assert canonical_section_heading("100 例非哺乳期乳腺炎的外科治疗") is None
    assert canonical_section_heading("30 例患者完成随访") is None
    assert canonical_section_heading("本研究结果显示两组患者复发率比较差异无统计学意义") is None
    assert canonical_section_heading("较好疗效") is None
    assert canonical_section_heading("30") is None
    assert canonical_section_heading("患者接受治疗后症状明显改善。") is None


def test_document_title_is_kept_in_body_instead_of_becoming_a_section() -> None:
    pages = [
        {
            "title": "100例非哺乳期乳腺炎的外科治疗",
            "page_number": 1,
            "extraction_method": "native_text",
            "text": "中国医药导报\n100 例非哺乳期乳腺炎的外科治疗\n摘要\n观察患者疗效。",
        }
    ]

    sections = section_sentence_items(pages)

    assert list(sections) == ["正文", "摘要"]
    assert sections["正文"][0]["text"].replace(" ", "").endswith(
        "100例非哺乳期乳腺炎的外科治疗"
    )


def test_section_sentence_items_preserves_page_numbers_and_skips_references() -> None:
    pages = [
        {
            "page_number": 1,
            "extraction_method": "native_text",
            "text": "摘要\n本研究观察非哺乳期乳腺炎患者的疗效。\n1 资料与方法\n共纳入30例患者。",
        },
        {
            "page_number": 2,
            "extraction_method": "ocr",
            "text": "结果\n联合治疗后复发率下降。\n参考文献\n[1] 某某等。",
        },
    ]

    sections = section_sentence_items(pages)

    assert sections["摘要"][0]["page_number"] == 1
    assert sections["资料与方法"][0]["page_number"] == 1
    assert sections["结果"][0]["page_number"] == 2
    assert "参考文献" not in sections


def test_choose_semantic_boundaries_uses_local_valley() -> None:
    sentences = [f"第{i}句。" for i in range(1, 7)]
    similarities = [0.91, 0.88, 0.40, 0.90, 0.87]
    config = ChunkConfig(min_sentences=2, max_sentences=5, similarity_percentile=30)

    assert choose_semantic_boundaries(sentences, similarities, config) == [3]


def test_build_chunk_records_preserves_cnki_page_metadata() -> None:
    document = {
        "document_id": "CNKI-TEST",
        "title": "非哺乳期乳腺炎治疗研究",
        "source_path": "data/articles/raw/chinese/cnki_pdf/test.pdf",
    }
    items = [
        {"text": "患者接受利福平治疗。", "page_number": 2, "extraction_method": "native_text"},
        {"text": "三个月后肿块缩小。", "page_number": 3, "extraction_method": "ocr"},
    ]
    config = ChunkConfig(min_chars=0, max_chars=1000)

    chunks = build_chunk_records(document, "结果", items, [], config)

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "CNKI-TEST"
    assert chunks[0]["language"] == "zh"
    assert chunks[0]["source_pages"] == [2, 3]
    assert chunks[0]["extraction_methods"] == ["native_text", "ocr"]


def test_build_chunk_records_enforces_raw_character_limit() -> None:
    document = {
        "document_id": "CNKI-LONG",
        "title": "Long English abstract",
        "source_path": "data/articles/raw/chinese/cnki_pdf/long.pdf",
    }
    items = [
        {
            "text": "clinical treatment outcome " * 45,
            "page_number": index,
            "extraction_method": "ocr",
        }
        for index in range(1, 7)
    ]
    config = ChunkConfig(
        min_chars=0,
        max_chars=10000,
        max_raw_chars=2000,
    )

    chunks = build_chunk_records(document, "摘要", items, [], config)

    assert len(chunks) > 1
    assert max(chunk["char_count"] for chunk in chunks) <= config.max_raw_chars


def test_build_chunk_records_merges_short_final_range_backward() -> None:
    document = {
        "document_id": "CNKI-TAIL",
        "title": "短结论合并测试",
        "source_path": "data/articles/raw/chinese/cnki_pdf/tail.pdf",
    }
    items = [
        {"text": "治疗方案与疗效观察。" * 20, "page_number": 1, "extraction_method": "ocr"},
        {"text": "复发情况与随访结果。" * 20, "page_number": 2, "extraction_method": "ocr"},
        {"text": "结论有效。", "page_number": 2, "extraction_method": "ocr"},
    ]
    config = ChunkConfig(min_chars=50, max_chars=1000)

    chunks = build_chunk_records(document, "结论", items, [1, 2], config)

    assert len(chunks) == 2
    assert chunks[-1]["sentence_count"] == 2
    assert chunks[-1]["text"].endswith("结论有效。")
