# articles/raw/chinese/cnki_pdf 目录说明

本目录保存中文 CNKI 文献 PDF，是中文文献解析流程的输入。

主要内容：

- `*.pdf`：人工从 CNKI 下载并转换得到的中文文献全文。
- `conversion_manifest.csv`：CAJ/PDF 转换过程的文件清单。
- `conversion_report.md`：转换结果摘要。
- `conversion.log`、`conversion_error.log`：转换过程日志，用于排查失败文件。

这些 PDF 不是最终 RAG 输出。后续流程会把 PDF 解析为页面文本、chunk，再筛选并生成向量索引。
