# articles/processed/chinese/page_cache 目录说明

本目录保存每篇中文 PDF 的逐页解析缓存。每个 `CNKI-*.jsonl` 通常对应一篇 PDF，内部按页记录文本和解析信息。

用途：

1. 避免每次调试都重新 OCR 或重新解析 PDF。
2. 当某篇文章 chunk 异常时，可回到逐页文本定位问题。
3. 支持后续重新切分 chunk 或调整筛选规则。

这里是缓存型中间结果，不是最终知识库索引。
