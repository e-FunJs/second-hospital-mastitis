from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


# -----------------------------------------------------------------------------
# 配置对象
# -----------------------------------------------------------------------------
# 第二层 LLM 只负责读取第一层 RAG prompt 并生成答案。默认使用 8bit 量化，
# 以降低 Qwen3-8B 的显存占用；如果以后要微调，应另走 LoRA/QLoRA 流程。


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = False


@dataclass(frozen=True)
class LLMConfig:
    model_path: str
    backend: str = "local_hf"
    model_name: str = "qwen3-8b"
    quantization: str = "8bit"
    generation: GenerationConfig = GenerationConfig()


# -----------------------------------------------------------------------------
# 配置和路径处理
# -----------------------------------------------------------------------------
# prompt 文件通常来自 rag_answer.py：xxx_prompt.txt。输出文件沿用同一个前缀，
# 生成 xxx_answer.md 和 xxx_answer.json，便于同一问题的 evidence/prompt/answer 对齐。


def load_llm_config(config_path: Path) -> LLMConfig:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm = config.get("llm", {})
    generation = llm.get("generation", {}) or {}
    gen_config = GenerationConfig(
        max_new_tokens=int(generation.get("max_new_tokens", 768)),
        temperature=float(generation.get("temperature", 0.2)),
        top_p=float(generation.get("top_p", 0.9)),
        repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
        do_sample=bool(generation.get("do_sample", False)),
    )
    model_path = llm.get("local_model_path")
    if not model_path:
        raise ValueError(f"missing llm.local_model_path in {config_path}")
    return LLMConfig(
        model_path=str(model_path),
        backend=str(llm.get("backend", "local_hf")),
        model_name=str(llm.get("model_name", "qwen3-8b")),
        quantization=str(llm.get("quantization", "8bit")),
        generation=gen_config,
    )


def output_paths_for_prompt(prompt_path: str | Path) -> dict[str, Path]:
    path = Path(prompt_path)
    stem = path.stem
    if stem.endswith("_prompt"):
        stem = stem[: -len("_prompt")]
    return {
        "answer_md": path.with_name(f"{stem}_answer.md"),
        "answer_json": path.with_name(f"{stem}_answer.json"),
    }


def read_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt is empty: {path}")
    return prompt


# -----------------------------------------------------------------------------
# 模型加载与生成
# -----------------------------------------------------------------------------
# 这里是和前面 embedding 推理类似的“离线推理”，不是训练。8bit 量化需要
# bitsandbytes + accelerate；如果显存仍不足，后续再考虑 4bit 或更小模型。


def build_quantization_config(quantization: str):
    if quantization == "none":
        return None
    if quantization != "8bit":
        raise ValueError(f"unsupported quantization: {quantization}")

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_8bit=True)


def load_local_causal_lm(model_path: Path, quantization: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    quantization_config = build_quantization_config(quantization)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return tokenizer, model


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是严谨的医学文献RAG助手，只能基于用户提供的Evidence回答，并保留证据编号引用。术语约束：IGM 指特发性肉芽肿性乳腺炎，NPM 指非哺乳期乳腺炎，不要翻译成粒细胞肉芽肿性乳腺炎。",
        },
        {"role": "user", "content": prompt},
    ]


def extract_model_inputs(chat_template_output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """把 tokenizer 的不同返回格式统一成 model.generate 可用的张量。"""
    if isinstance(chat_template_output, torch.Tensor):
        input_ids = chat_template_output
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    if isinstance(chat_template_output, Mapping) and "input_ids" in chat_template_output:
        input_ids = chat_template_output["input_ids"]
        attention_mask = chat_template_output.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    raise TypeError(f"unsupported chat template output type: {type(chat_template_output)!r}")


def generate_with_local_llm(prompt: str, model_path: Path, llm_config: LLMConfig) -> str:
    tokenizer, model = load_local_causal_lm(model_path, llm_config.quantization)
    messages = build_messages(prompt)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            # Qwen3 默认可能进入 thinking 模式；RAG正式答案只需要可引用的结论文本。
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            # 兼容不支持 enable_thinking 参数的 tokenizer。
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
    else:
        joined = "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)
        input_ids = tokenizer(joined, return_tensors="pt")["input_ids"]

    input_ids, attention_mask = extract_model_inputs(input_ids)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    generation_kwargs = {
        "max_new_tokens": llm_config.generation.max_new_tokens,
        "do_sample": llm_config.generation.do_sample,
        "repetition_penalty": llm_config.generation.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if llm_config.generation.do_sample:
        generation_kwargs["temperature"] = llm_config.generation.temperature
        generation_kwargs["top_p"] = llm_config.generation.top_p

    # 关键点：只截取新生成的 token，避免把完整 prompt 也写进 answer。
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    generated_ids = output_ids[0, input_ids.shape[-1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer


# -----------------------------------------------------------------------------
# 输出文件
# -----------------------------------------------------------------------------
# Markdown 给人读，JSON 给程序和后续评估脚本读。


def build_answer_payload(
    prompt_path: str,
    model_path: str,
    answer: str,
    generation: GenerationConfig,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_path": prompt_path,
        "model_path": model_path,
        "generation": asdict(generation),
        "answer": answer,
    }


def write_answer_outputs(payload: dict[str, Any], answer_md: Path, answer_json: Path) -> None:
    answer_md.parent.mkdir(parents=True, exist_ok=True)
    answer_json.parent.mkdir(parents=True, exist_ok=True)
    answer_md.write_text(payload["answer"].strip() + "\n", encoding="utf-8")
    answer_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
# 输入一个 prompt.txt，输出 answer.md/json。prompt 的生成仍由第一层 rag_answer.py 完成。


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local Qwen answer from a first-layer RAG prompt.")
    parser.add_argument("--prompt", type=Path, required=True, help="Path to *_prompt.txt from rag_answer.py.")
    parser.add_argument("--config", type=Path, default=Path("configs/llm.yaml"))
    parser.add_argument("--model-path", type=Path, help="Override llm.local_model_path in config.")
    parser.add_argument("--answer-md", type=Path, help="Optional output Markdown path.")
    parser.add_argument("--answer-json", type=Path, help="Optional output JSON path.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation.max_new_tokens.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.prompt.exists():
        print(f"prompt not found: {args.prompt}", file=sys.stderr)
        return 2
    llm_config = load_llm_config(args.config)
    if args.max_new_tokens is not None:
        llm_config = LLMConfig(
            model_path=llm_config.model_path,
            backend=llm_config.backend,
            model_name=llm_config.model_name,
            quantization=llm_config.quantization,
            generation=GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=llm_config.generation.temperature,
                top_p=llm_config.generation.top_p,
                repetition_penalty=llm_config.generation.repetition_penalty,
                do_sample=llm_config.generation.do_sample,
            ),
        )
    model_path = args.model_path or Path(llm_config.model_path)
    if not model_path.exists():
        print(f"model path not found: {model_path}", file=sys.stderr)
        return 2

    output_paths = output_paths_for_prompt(args.prompt)
    answer_md = args.answer_md or output_paths["answer_md"]
    answer_json = args.answer_json or output_paths["answer_json"]

    prompt = read_prompt(args.prompt)
    answer = generate_with_local_llm(prompt, model_path, llm_config)
    payload = build_answer_payload(str(args.prompt), str(model_path), answer, llm_config.generation)
    write_answer_outputs(payload, answer_md, answer_json)

    print(f"answer_md={answer_md}")
    print(f"answer_json={answer_json}")
    print(f"answer_chars={len(answer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
