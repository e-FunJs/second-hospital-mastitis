"""中文 PDF 解析模块的纯 Python 单元测试。"""

from __future__ import annotations

from rag_medical.chinese.parse_pdf import (
    ParseConfig,
    document_id_for_path,
    native_text_is_usable,
    remove_repeated_marginal_lines,
    split_pdftotext_pages,
)


def test_split_pdftotext_pages_pads_missing_pages() -> None:
    pages = split_pdftotext_pages("第一页正文\f第二页正文\f", page_count=3)

    assert pages == ["第一页正文", "第二页正文", ""]


def test_native_text_quality_requires_chinese_content() -> None:
    config = ParseConfig(min_native_chars=10, min_native_cjk_chars=5)

    assert native_text_is_usable("非哺乳期乳腺炎治疗效果良好", config)
    assert not native_text_is_usable("1234567890abcdef", config)


def test_native_text_quality_rejects_broken_private_use_mapping() -> None:
    config = ParseConfig(min_native_chars=10, min_native_cjk_chars=5)
    broken = "非哺乳期乳腺炎治疗效果良好" + "\ue5e5" * 30

    assert not native_text_is_usable(broken, config)


def test_repeated_page_header_and_page_number_are_removed() -> None:
    records = [
        {
            "text": f"中国医药导报 2026年第1期\n第{page}页\n第{page}页正文内容不同。",
            "text_char_count": 0,
            "cjk_char_count": 0,
        }
        for page in range(1, 5)
    ]

    remove_repeated_marginal_lines(records)

    assert all("中国医药导报" not in record["text"] for record in records)
    assert all("正文内容不同" in record["text"] for record in records)


def test_document_id_is_stable_and_filename_dependent(tmp_path) -> None:
    first = tmp_path / "非哺乳期乳腺炎治疗.pdf"
    second = tmp_path / "乳腺炎诊治.pdf"

    assert document_id_for_path(first) == document_id_for_path(first)
    assert document_id_for_path(first) != document_id_for_path(second)
