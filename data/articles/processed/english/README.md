# articles/processed/english 目录说明

本目录保存英文文献从 XML 解析到 RAG chunk 的处理结果。

常见文件：

- `article_sections.jsonl`：按文章章节/段落解析出的文本。
- `article_chunks.jsonl`：根据章节和长度规则切分后的英文 chunk。
- `abstract_chunks.jsonl`：从摘要生成的 chunk。
- `chunk_manifest.csv`：chunk 生成清单。
- `rag_chunks*.jsonl`：经过规则筛选后的 RAG 候选语料。
- `pmc_download_manifest.csv`、`article_parse_manifest.csv`：下载与解析过程清单。
- `semantic/`：在规则筛选后，进一步用语义方法处理的英文筛选结果。

这些文件用于构建英文索引或审查英文知识库质量。
