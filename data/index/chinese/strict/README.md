# index/chinese/strict 目录说明

本目录是当前中文 RAG 知识库的严格版最终输出目录。它由中文严格筛选后的 chunk 构建而来，可直接供检索脚本加载。

核心文件：

- `faiss.index`：最终向量索引文件。检索时 FAISS 主要加载它来做近邻搜索。
- `chunk_metadata.jsonl`：最终索引的文本元数据。它记录每个向量对应的 chunk 文本、来源文献、页码等信息。没有它，即使检索到了向量编号，也无法知道证据原文是什么。
- `chunk_embeddings.npy`：构建 FAISS 索引时生成的向量矩阵。它不是问答时必须加载的唯一文件，但用于重建、检查或更换索引格式很重要。
- `embedding_manifest.json`：记录 embedding 构建配置，例如模型路径、输入 chunk 文件、向量维度和数量。
- `faiss_manifest.json`：记录 FAISS 索引构建配置。
- `index_validation_report.json`：索引完整性检查结果。当前应重点看 `ready_for_retrieval` 是否为 `true`。
- `retrieval_smoke_report.json`：检索冒烟测试结果，用于确认典型问题能否检索到合理证据。

最重要结论：

如果你问“哪个文件才是最终输出结果”，严格说不是单个文件，而是本目录整体。运行 RAG 至少需要 `faiss.index` + `chunk_metadata.jsonl`，并且需要使用与 `embedding_manifest.json` 中一致的 BGE 模型来编码用户问题。
