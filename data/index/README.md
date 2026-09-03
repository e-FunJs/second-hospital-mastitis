# index 目录说明

本目录保存已经向量化、可以被检索器直接加载的 RAG 索引。它是 RAG 数据流程中最接近“最终可运行知识库”的位置。

- `english/`：英文文献索引。
- `chinese/`：中文文献索引。
- `combined/`：中英文合并索引的预留目录。

重要：索引目录通常不是只看一个文件。一个可运行索引至少需要：

1. `faiss.index`：向量检索索引本体。
2. `chunk_metadata.jsonl`：每个向量对应的原文片段、标题、来源、页码等信息。
3. `embedding_manifest.json`/`faiss_manifest.json`：记录模型、维度、输入文件等构建信息。
4. `index_validation_report.json`：确认索引数量、维度和 metadata 是否匹配。

因此，后续部署或问答脚本应按“一个索引目录”加载，而不是只复制单个文件。
