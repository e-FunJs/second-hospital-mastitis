# articles/processed/chinese/filtered 目录说明

本目录保存中文 chunk 经过严格医学筛选后的结果。

- `rag_chunks_strict.jsonl`：可直接进入中文 RAG 索引的严格语料。
- `rag_chunks_review.jsonl`：主题相关但需要人工复核的语料。
- `rag_chunks_excluded.jsonl`：不建议进入知识库的语料。

当前中文严格索引 `data/index/chinese/strict/` 就是基于 `rag_chunks_strict.jsonl` 构建的。
