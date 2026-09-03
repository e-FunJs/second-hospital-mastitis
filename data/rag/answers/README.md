# rag/answers 目录说明

本目录保存 RAG 检索和问答测试的结果。

常见文件类型：

- `*_evidence.json`：某个问题检索到的证据片段。
- `*_prompt.txt`：送入 LLM 的提示词。
- `*_answer.md`/`*_answer.json`：LLM 生成的答案。
- `*_eval.json`：规则评估或 LLM-as-judge 评估结果。

这些文件用于验证 RAG 效果、复盘引用是否可靠，不是构建索引的输入。
