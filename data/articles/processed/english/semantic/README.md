# articles/processed/english/semantic 目录说明

本目录保存英文文献经过语义筛选后的 RAG chunk。

用途：在基础规则筛选后，进一步识别并剔除不适合进入医学知识库的内容，例如明显动物研究、主题偏离或与人类非哺乳期乳腺炎关系弱的片段。

- `rag_chunks_strict.jsonl`：严格可用语料。
- `rag_chunks_review.jsonl`：需要人工复核的语料。
- `rag_chunks_excluded.jsonl`：不建议进入知识库的语料。
