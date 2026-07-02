"""本地 LLM 第二层生成脚本测试。

测试只覆盖配置、prompt 读取、输出落盘等轻量逻辑，不在单元测试里加载 Qwen 权重。
"""

from __future__ import annotations

import json

import torch
from transformers.tokenization_utils_base import BatchEncoding

from rag_medical.generate_answer import (
    GenerationConfig,
    extract_model_inputs,
    build_answer_payload,
    build_messages,
    load_llm_config,
    output_paths_for_prompt,
    write_answer_outputs,
)


def test_build_messages_contains_medical_term_glossary() -> None:
    messages = build_messages("请回答问题")

    system_content = messages[0]["content"]
    assert "IGM 指特发性肉芽肿性乳腺炎" in system_content
    assert "不要翻译成粒细胞肉芽肿性乳腺炎" in system_content


def test_extract_model_inputs_accepts_batch_encoding_style_output() -> None:
    chat_output = BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )

    input_ids, attention_mask = extract_model_inputs(chat_output)

    assert input_ids.tolist() == [[1, 2, 3]]
    assert attention_mask.tolist() == [[1, 1, 1]]


def test_load_llm_config_reads_model_path_and_generation_defaults(tmp_path) -> None:
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        """
llm:
  backend: local_hf
  local_model_path: models/llm/qwen3-8b
  quantization: 8bit
  generation:
    max_new_tokens: 256
    temperature: 0.2
""",
        encoding="utf-8",
    )

    config = load_llm_config(config_path)

    assert config.model_path == "models/llm/qwen3-8b"
    assert config.quantization == "8bit"
    assert config.generation.max_new_tokens == 256
    assert config.generation.temperature == 0.2


def test_output_paths_for_prompt_reuses_prompt_stem() -> None:
    prompt_path = "data/rag/answers/example_prompt.txt"

    paths = output_paths_for_prompt(prompt_path)

    assert paths["answer_md"].name == "example_answer.md"
    assert paths["answer_json"].name == "example_answer.json"


def test_write_answer_outputs_saves_markdown_and_json(tmp_path) -> None:
    payload = build_answer_payload(
        prompt_path="prompt.txt",
        model_path="models/llm/qwen3-8b",
        answer="结论：超声表现包括低回声病灶。[E1]",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.1, top_p=0.9),
    )
    answer_md = tmp_path / "answer.md"
    answer_json = tmp_path / "answer.json"

    write_answer_outputs(payload, answer_md, answer_json)

    assert "超声表现" in answer_md.read_text(encoding="utf-8")
    loaded = json.loads(answer_json.read_text(encoding="utf-8"))
    assert loaded["model_path"] == "models/llm/qwen3-8b"
    assert loaded["generation"]["max_new_tokens"] == 128
