# index/english/strict 目录说明

本目录保存英文严格版 RAG 索引。它适合用于医学证据较严格的英文文献检索。

使用时通常需要同时加载 `faiss.index` 和 `chunk_metadata.jsonl`，并检查 manifest 与当前 embedding 模型是否一致。
