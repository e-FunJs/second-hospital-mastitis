# articles 目录说明

本目录保存文献内容相关数据。它回答的问题是：文献原文在哪里、原文解析成了什么、最终哪些文本片段可以进入 RAG。

- `raw/`：原始全文文件，例如英文 PMC XML、中文 CNKI PDF。
- `processed/`：从原始全文中抽取出的页面、章节、段落、chunk，以及筛选后的 RAG 语料。

注意：这里的数据还不是最终可检索索引。RAG 真正运行时优先使用 `data/index/` 下的 FAISS 索引和 metadata。
