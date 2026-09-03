# registry/chinese/filtered 目录说明

本目录保存中文文献级别的严格筛选结果。

- `literature_registry_strict.csv`：文献级别上较适合进入知识库的中文文献。
- `literature_registry_review.csv`：需要人工复核的中文文献。
- `literature_registry_excluded.csv`：不建议进入知识库的中文文献。
- `filter_report.md`：筛选规则和数量统计报告。
- `view/`：便于人工查看的导出版本。

注意：这里是“文献级”筛选；最终进入向量索引的是 chunk 级文本，位于 `data/articles/processed/chinese/filtered/`。
