# articles/processed 目录说明

本目录保存文献解析、切分和筛选后的中间结果。它连接“原始文献文件”和“最终向量索引”。

- `english/`：英文文献解析、chunk、筛选结果。
- `chinese/`：中文文献解析、chunk、筛选结果。
- `combined/`：中英文合并语料的预留位置，目前不是主要结果目录。

这里的 `jsonl`/`csv` 文件通常可读、可审查，但 RAG 检索运行时应优先加载 `data/index/`。
