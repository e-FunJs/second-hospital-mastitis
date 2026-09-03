# registry/english/filtered 目录说明

本目录保存英文文献清单的筛选结果，通常分为严格可用、需审查、排除三类。

- `semantic/`：在规则筛选基础上，用 BGE 等语义模型进一步筛掉明显偏离主题的文献。
- `view/`：便于人工打开查看的筛选表格或导出文件。

这里是文献级筛选结果；真正用于 RAG 的文本 chunk 在 `data/articles/processed/english/`。
