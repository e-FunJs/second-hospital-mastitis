# articles/processed/chinese 目录说明

本目录保存中文 CNKI PDF 解析后的文本、chunk 和质控结果。

常见文件：

- `article_pages.jsonl`：PDF 按页解析后的中文文本。部分页面可能来自 OCR。
- `article_chunks.jsonl`：将页面文本切分后得到的中文 chunk。
- `chunk_manifest.csv`：chunk 生成清单。
- `pdf_parse_manifest.csv`：每篇 PDF 的解析状态、页数和 OCR 情况。
- `pdf_parse_report.md`：中文 PDF 解析质量报告。
- `common_readiness_report.json`：进入通用索引流程前的结构检查报告。
- `filtered/`：严格筛选后的中文 RAG chunk。
- `page_cache/`：每篇 PDF 的逐页解析缓存。

本目录是中文 RAG 索引构建的上游输入，不是最终索引。
