"""通用语料就绪检查器测试。"""

from __future__ import annotations

from rag_medical.common.check_corpus import validate_corpus


def valid_chinese_chunk() -> dict:
    return {
        "chunk_id": "CNKI-1::SEC::001",
        "document_id": "CNKI-1",
        "title": "非哺乳期乳腺炎治疗",
        "language": "zh",
        "source_type": "cnki_pdf",
        "source_path": "data/articles/raw/chinese/cnki_pdf/a.pdf",
        "section": "结果",
        "source_pages": [1],
        "page_start": 1,
        "page_end": 1,
        "extraction_methods": ["native_text"],
        "text": "患者接受联合治疗后肿块明显缩小，随访期间未观察到疾病复发。",
    }


def test_valid_chinese_chunk_is_ready_for_common_embedding() -> None:
    report = validate_corpus([valid_chinese_chunk()], expected_language="zh")

    assert report["ready_for_common_embedding"] is True
    assert report["error_count"] == 0


def test_duplicate_and_missing_fields_fail_readiness() -> None:
    first = valid_chinese_chunk()
    second = valid_chinese_chunk()
    second["source_pages"] = []

    report = validate_corpus([first, second], expected_language="zh")

    assert report["ready_for_common_embedding"] is False
    assert report["error_count"] == 2
