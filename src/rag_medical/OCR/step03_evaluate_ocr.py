"""对 Tesseract 与 DeepSeek-OCR-2 的固定页面结果进行配对评估。

输入：
    1. ``Eval_Full_Page.jsonl``：整页人工金标准，字段为
       ``task_name``、``page_number``、``text``。
    2. ``Eval_Text.jsonl``：局部段落人工金标准，字段与整页标注相同。
    3. 两个 OCR 脚本生成的逐页 JSONL 结果。

输出（写入 ``--output-dir``）：
    1. ``page_scores.csv``：逐页严格 CER、标准化 CER、WER 与噪声指标。
    2. ``snippet_scores.csv``：局部标注在整页 OCR 文本中的最佳匹配结果。
    3. ``snippet_matches.jsonl``：便于人工抽查的原文、命中片段和误差详情。
    4. ``summary.json``：覆盖率、汇总指标、配对检验与 bootstrap 区间。
    5. ``report.md``：可直接阅读的中文比较报告。

本文件还提供 ``prepare`` 子命令：从总 OCR 任务清单中选出人工标注涉及的
页面，确保两个 OCR 后端处理完全相同的输入页。评估过程不会先清洗 OCR
文本，以免把模型产生的乱码或多余内容从指标中隐藏。
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_GOLD_PAGES = Path("data/rag/OCR_Eval_Metirc/Eval_Full_Page.jsonl")
DEFAULT_GOLD_SNIPPETS = Path("data/rag/OCR_Eval_Metirc/Eval_Text.jsonl")
DEFAULT_MASTER_TASKS = Path(
    "cache_output_test/full_cn_rebuild_20260917_2011/ocr_tasks.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path("data/rag/answers/OCR")
ENGINE_NAMES = ("tesseract", "deepseek_ocr2")


def project_path(path: Path) -> Path:
    """将相对路径统一解释为项目根目录下的路径。"""

    return path if path.is_absolute() else PROJECT_DIR / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并在数据损坏时给出精确行号。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 解析失败：{path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"记录必须是对象：{path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """原子写入 JSONL，避免中断时留下半个结果文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """使用 UTF-8 BOM 写 CSV，便于 Excel 与 VS Code 直接识别中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def record_key(record: dict[str, Any], *, ocr_record: bool = False) -> tuple[str, int]:
    """以 PDF 文件名和页码对齐人工标注、任务清单及 OCR 结果。"""

    name = record.get("task_name")
    if ocr_record and not name:
        name = record.get("pdf_path")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"记录缺少 task_name/pdf_path：{record}")
    try:
        page_number = int(record["page_number"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"记录页码无效：{record}") from exc
    if page_number < 1:
        raise ValueError(f"页码必须从 1 开始：{record}")
    return Path(name).name, page_number


def validate_gold(records: list[dict[str, Any]], label: str) -> None:
    for index, record in enumerate(records, start=1):
        record_key(record)
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label} 第 {index} 条缺少有效 text")


def prepare_tasks(args: argparse.Namespace) -> int:
    """选出整页和局部标注涉及页面的并集，供两个 OCR 后端公平复用。"""

    gold_pages = read_jsonl(project_path(args.gold_pages))
    gold_snippets = read_jsonl(project_path(args.gold_snippets))
    master_tasks = read_jsonl(project_path(args.master_tasks))
    validate_gold(gold_pages, "整页标注")
    validate_gold(gold_snippets, "局部标注")

    task_index: dict[tuple[str, int], dict[str, Any]] = {}
    for task in master_tasks:
        key = record_key(task, ocr_record=True)
        if key in task_index:
            raise ValueError(f"总任务清单存在重复页面：{key}")
        task_index[key] = task

    # 先按整页标注顺序，再补充仅出现在局部标注中的页面，保证输出可复现。
    requested_keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for record in [*gold_pages, *gold_snippets]:
        key = record_key(record)
        if key not in seen:
            requested_keys.append(key)
            seen.add(key)

    missing = [key for key in requested_keys if key not in task_index]
    if missing:
        preview = ", ".join(f"{name}:p{page}" for name, page in missing[:10])
        raise ValueError(f"总 OCR 任务清单缺少 {len(missing)} 页：{preview}")

    selected = [task_index[key] for key in requested_keys]
    output_path = project_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise ValueError(f"输出已存在，请使用 --overwrite：{output_path}")
    write_jsonl(output_path, selected)
    print(f"prepared_tasks={len(selected)}")
    print(f"full_page_labels={len(gold_pages)}")
    print(f"snippet_labels={len(gold_snippets)}")
    print(f"output={output_path}")
    return 0


def normalize_strict(text: str) -> str:
    """严格模式只统一换行编码和首尾空白，不修正 OCR 内容。"""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_content_with_map(text: str) -> tuple[str, list[int]]:
    """进行 NFKC、大小写与空白标准化，并保留到原文位置的映射。

    标准化 CER 需要忽略排版换行和人工录入空格，但标点、数字及正文字符
    全部保留。位置映射用于把局部最佳匹配重新还原成可人工检查的 OCR 原文。
    """

    normalized: list[str] = []
    source_positions: list[int] = []
    for source_index, character in enumerate(text):
        expanded = unicodedata.normalize("NFKC", character).casefold()
        for normalized_character in expanded:
            if normalized_character.isspace():
                continue
            normalized.append(normalized_character)
            source_positions.append(source_index)
    return "".join(normalized), source_positions


def normalize_content(text: str) -> str:
    return normalize_content_with_map(text)[0]


TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|"
    r"[a-z]+(?:[-'][a-z]+)*|"
    r"\d+(?:\.\d+)?|"
    r"[^\s]",
    re.IGNORECASE,
)


def tokenize_mixed(text: str) -> list[str]:
    """将中文按字、英文/数字按词切分，得到可复现的混合语言 WER 单位。"""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_PATTERN.findall(normalized)


def levenshtein_distance(source: Sequence[Hashable], target: Sequence[Hashable]) -> int:
    """用 Myers 位并行算法计算精确编辑距离，适合数千字的整页文本。"""

    if len(source) > len(target):
        source, target = target, source
    source_length = len(source)
    if source_length == 0:
        return len(target)

    char_masks: dict[Hashable, int] = {}
    for index, item in enumerate(source):
        char_masks[item] = char_masks.get(item, 0) | (1 << index)

    bit_mask = (1 << source_length) - 1
    highest_bit = 1 << (source_length - 1)
    positive_vertical = bit_mask
    negative_vertical = 0
    score = source_length

    for item in target:
        equal_bits = char_masks.get(item, 0)
        vertical_or_equal = equal_bits | negative_vertical
        horizontal = (
            (((equal_bits & positive_vertical) + positive_vertical) ^ positive_vertical)
            | equal_bits
        ) & bit_mask
        positive_horizontal = (
            negative_vertical | ~(horizontal | positive_vertical)
        ) & bit_mask
        negative_horizontal = positive_vertical & horizontal
        if positive_horizontal & highest_bit:
            score += 1
        elif negative_horizontal & highest_bit:
            score -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & bit_mask
        negative_horizontal = (negative_horizontal << 1) & bit_mask
        positive_vertical = (
            negative_horizontal | ~(vertical_or_equal | positive_horizontal)
        ) & bit_mask
        negative_vertical = positive_horizontal & vertical_or_equal
    return score


def alignment_breakdown(reference: str, hypothesis: str) -> dict[str, int]:
    """给出可解释的替换、删除和插入数量。

    CER 本身使用上面的精确 Levenshtein 距离。本函数用序列对齐拆解错误类型；
    拆解用于分析“漏字”还是“额外乱码”占主导，不替代 CER。
    """

    substitutions = deletions = insertions = 0
    matcher = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        ref_size = ref_end - ref_start
        hyp_size = hyp_end - hyp_start
        if tag == "replace":
            shared = min(ref_size, hyp_size)
            substitutions += shared
            deletions += ref_size - shared
            insertions += hyp_size - shared
        elif tag == "delete":
            deletions += ref_size
        elif tag == "insert":
            insertions += hyp_size
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def best_local_match(reference: str, page_text: str) -> tuple[int, int, int]:
    """在整页 OCR 文本中寻找与局部人工标注编辑距离最小的连续片段。

    动态规划首行全部置零，表示 OCR 页开头 A 可以免费跳过；在最后一行
    选择最小值，表示页尾 C 也不计入误差。因此当 OCR 输出为 A+B+C 而
    人工标注仅为 B 时，指标只比较 B 对应的最佳片段。
    """

    if not reference or not page_text:
        return 0, 0, len(reference)

    page_length = len(page_text)
    previous_costs = [0] * (page_length + 1)
    previous_starts = list(range(page_length + 1))

    for reference_index, reference_character in enumerate(reference, start=1):
        current_costs = [reference_index] + [0] * page_length
        current_starts = [0] * (page_length + 1)
        for page_index, page_character in enumerate(page_text, start=1):
            substitution_cost = 0 if reference_character == page_character else 1
            candidates = (
                (previous_costs[page_index - 1] + substitution_cost, 0,
                 previous_starts[page_index - 1]),
                (previous_costs[page_index] + 1, 1, previous_starts[page_index]),
                (current_costs[page_index - 1] + 1, 2,
                 current_starts[page_index - 1]),
            )
            best_cost, _, best_start = min(candidates)
            current_costs[page_index] = best_cost
            current_starts[page_index] = best_start
        previous_costs = current_costs
        previous_starts = current_starts

    end = min(
        range(page_length + 1),
        key=lambda index: (
            previous_costs[index],
            abs((index - previous_starts[index]) - len(reference)),
            previous_starts[index],
        ),
    )
    return previous_starts[end], end, previous_costs[end]


def suspicious_noise_rate(text: str) -> float:
    """估计无需金标准即可发现的明显乱码比例，作为辅助指标。

    仅标记替代字符、异常控制字符、四次以上单字符重复、三次以上 2～3 字符
    短片段重复，以及连续四个以上符号。正常医学术语和普通标点不会因“不在
    词典中”而被判为噪声，因此该值偏保守，不能单独代表 OCR 正确率。
    """

    normalized = unicodedata.normalize("NFKC", text)
    non_space_positions = [i for i, char in enumerate(normalized) if not char.isspace()]
    if not non_space_positions:
        return 0.0
    suspicious: set[int] = set()

    for index, character in enumerate(normalized):
        category = unicodedata.category(character)
        if character == "\ufffd" or (category.startswith("C") and character not in "\n\t"):
            suspicious.add(index)

    patterns = (
        re.compile(r"([^\s])\1{3,}"),
        re.compile(r"([^\s]{2,3})\1{2,}"),
        re.compile(r"[^\w\s\u3400-\u4dbf\u4e00-\u9fff]{4,}"),
    )
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            suspicious.update(range(match.start(), match.end()))
    return len(suspicious) / len(non_space_positions)


def safe_rate(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else (0.0 if numerator == 0 else math.inf)


def index_ocr_results(records: list[dict[str, Any]], engine: str) -> dict[tuple[str, int], dict[str, Any]]:
    """索引 OCR 结果；失败记录仍保留，以便覆盖率报告反映真实失败。"""

    index: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = record_key(record, ocr_record=True)
        if key in index:
            raise ValueError(f"{engine} 结果存在重复页面：{key}")
        index[key] = record
    return index


def successful_text(record: dict[str, Any] | None) -> str | None:
    if not record or record.get("status") != "ok":
        return None
    text = record.get("text")
    return text if isinstance(text, str) and text.strip() else None


def page_score(
    engine: str,
    gold: dict[str, Any],
    ocr_record: dict[str, Any],
) -> dict[str, Any]:
    reference_raw = str(gold["text"])
    hypothesis_raw = str(ocr_record["text"])
    reference_strict = normalize_strict(reference_raw)
    hypothesis_strict = normalize_strict(hypothesis_raw)
    reference = normalize_content(reference_raw)
    hypothesis = normalize_content(hypothesis_raw)

    strict_distance = levenshtein_distance(reference_strict, hypothesis_strict)
    content_distance = levenshtein_distance(reference, hypothesis)
    reference_tokens = tokenize_mixed(reference_raw)
    hypothesis_tokens = tokenize_mixed(hypothesis_raw)
    word_distance = levenshtein_distance(reference_tokens, hypothesis_tokens)
    breakdown = alignment_breakdown(reference, hypothesis)
    key = record_key(gold)

    return {
        "engine": engine,
        "task_name": key[0],
        "page_number": key[1],
        "task_id": ocr_record.get("task_id", ""),
        "reference_chars": len(reference_strict),
        "ocr_chars": len(hypothesis_strict),
        "length_ratio": safe_rate(len(hypothesis), len(reference)),
        "strict_edit_distance": strict_distance,
        "strict_cer": safe_rate(strict_distance, len(reference_strict)),
        "normalized_reference_chars": len(reference),
        "normalized_ocr_chars": len(hypothesis),
        "normalized_edit_distance": content_distance,
        "normalized_cer": safe_rate(content_distance, len(reference)),
        "wer": safe_rate(word_distance, len(reference_tokens)),
        "alignment_substitutions": breakdown["substitutions"],
        "alignment_deletions": breakdown["deletions"],
        "alignment_insertions": breakdown["insertions"],
        "alignment_insertion_rate": safe_rate(
            breakdown["insertions"], len(reference)
        ),
        "suspicious_noise_rate": suspicious_noise_rate(hypothesis_raw),
        "elapsed_seconds": ocr_record.get("elapsed_seconds", ""),
    }


def snippet_score(
    engine: str,
    snippet_id: str,
    gold: dict[str, Any],
    ocr_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_raw = str(gold["text"])
    page_raw = str(ocr_record["text"])
    reference = normalize_content(reference_raw)
    page, page_to_raw = normalize_content_with_map(page_raw)
    start, end, distance = best_local_match(reference, page)

    if start < end and page_to_raw:
        raw_start = page_to_raw[start]
        raw_end = page_to_raw[end - 1] + 1
        matched_raw = page_raw[raw_start:raw_end]
    else:
        raw_start = raw_end = 0
        matched_raw = ""
    matched_normalized = page[start:end]
    reference_tokens = tokenize_mixed(reference_raw)
    matched_tokens = tokenize_mixed(matched_raw)
    word_distance = levenshtein_distance(reference_tokens, matched_tokens)
    breakdown = alignment_breakdown(reference, matched_normalized)
    key = record_key(gold)

    row = {
        "engine": engine,
        "snippet_id": snippet_id,
        "task_name": key[0],
        "page_number": key[1],
        "reference_chars": len(reference),
        "matched_chars": len(matched_normalized),
        "normalized_edit_distance": distance,
        "normalized_cer": safe_rate(distance, len(reference)),
        "wer": safe_rate(word_distance, len(reference_tokens)),
        "alignment_substitutions": breakdown["substitutions"],
        "alignment_deletions": breakdown["deletions"],
        "alignment_insertions": breakdown["insertions"],
        "normalized_start": start,
        "normalized_end": end,
        "raw_start": raw_start,
        "raw_end": raw_end,
    }
    audit = {
        **row,
        "reference_text": reference_raw,
        "matched_text": matched_raw,
    }
    return row, audit


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    if not values:
        return {"count": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
    }


def paired_test(differences: list[float]) -> dict[str, Any]:
    """优先使用 Wilcoxon；若 SciPy 不可用则退化为精确配对符号检验。"""

    nonzero = [value for value in differences if not math.isclose(value, 0.0)]
    if not nonzero:
        return {"method": "wilcoxon", "statistic": 0.0, "p_value": 1.0}
    try:
        from scipy.stats import wilcoxon  # type: ignore

        result = wilcoxon(nonzero, alternative="two-sided", method="auto")
        return {
            "method": "wilcoxon_signed_rank",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        }
    except ImportError:
        positives = sum(value > 0 for value in nonzero)
        sample_size = len(nonzero)
        tail = min(positives, sample_size - positives)
        probability = 2 * sum(
            math.comb(sample_size, count) for count in range(tail + 1)
        ) / (2**sample_size)
        return {
            "method": "paired_sign_test_fallback",
            "statistic": positives,
            "p_value": min(1.0, probability),
        }


def bootstrap_median_difference(
    differences: list[float], samples: int, seed: int
) -> dict[str, Any]:
    if not differences:
        return {"samples": samples, "estimate": None, "ci95": [None, None]}
    rng = random.Random(seed)
    estimates = [
        statistics.median(rng.choices(differences, k=len(differences)))
        for _ in range(samples)
    ]
    return {
        "samples": samples,
        "estimate": statistics.median(differences),
        "ci95": [percentile(estimates, 0.025), percentile(estimates, 0.975)],
    }


def build_summary(
    page_rows: list[dict[str, Any]],
    snippet_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    page_by_engine = {
        engine: [row for row in page_rows if row["engine"] == engine]
        for engine in ENGINE_NAMES
    }
    snippet_by_engine = {
        engine: [row for row in snippet_rows if row["engine"] == engine]
        for engine in ENGINE_NAMES
    }
    engine_summary: dict[str, Any] = {}
    for engine in ENGINE_NAMES:
        engine_summary[engine] = {
            "coverage": coverage[engine],
            "pages": {
                field: metric_summary(page_by_engine[engine], field)
                for field in (
                    "strict_cer",
                    "normalized_cer",
                    "wer",
                    "alignment_insertion_rate",
                    "suspicious_noise_rate",
                )
            },
            "snippets": {
                field: metric_summary(snippet_by_engine[engine], field)
                for field in ("normalized_cer", "wer")
            },
        }

    tesseract_index = {
        (row["task_name"], row["page_number"]): row
        for row in page_by_engine["tesseract"]
    }
    deepseek_index = {
        (row["task_name"], row["page_number"]): row
        for row in page_by_engine["deepseek_ocr2"]
    }
    paired_keys = sorted(tesseract_index.keys() & deepseek_index.keys())
    differences = [
        float(tesseract_index[key]["normalized_cer"])
        - float(deepseek_index[key]["normalized_cer"])
        for key in paired_keys
    ]
    paired = {
        "metric": "normalized_cer",
        "difference_definition": "tesseract_minus_deepseek_ocr2",
        "paired_pages": len(paired_keys),
        "deepseek_better_pages": sum(value > 0 for value in differences),
        "tesseract_better_pages": sum(value < 0 for value in differences),
        "ties": sum(math.isclose(value, 0.0) for value in differences),
        "test": paired_test(differences),
        "bootstrap_median_difference": bootstrap_median_difference(
            differences, bootstrap_samples, seed
        ),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric_notes": {
            "strict_cer": "仅统一换行编码，保留空格和排版换行。",
            "normalized_cer": "NFKC、大小写统一并忽略空白；标点、数字和正文字符保留。",
            "wer": "中文按字、英文和数字按词的混合 token 错误率。",
            "suspicious_noise_rate": "保守的无金标准乱码启发式指标，不能替代 CER。",
        },
        "engines": engine_summary,
        "paired_comparison": paired,
    }


def format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# OCR 对比评估报告",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结果概览",
        "",
        "| OCR 后端 | 成功页/目标页 | 整页标准化 CER 中位数 | 整页 WER 中位数 | 局部 CER 中位数 | 可疑噪声率中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for engine in ENGINE_NAMES:
        data = summary["engines"][engine]
        coverage = data["coverage"]
        lines.append(
            "| {engine} | {success}/{expected} | {page_cer} | {page_wer} | "
            "{snippet_cer} | {noise} |".format(
                engine=engine,
                success=coverage["successful_pages"],
                expected=coverage["expected_pages"],
                page_cer=format_metric(data["pages"]["normalized_cer"]["median"]),
                page_wer=format_metric(data["pages"]["wer"]["median"]),
                snippet_cer=format_metric(data["snippets"]["normalized_cer"]["median"]),
                noise=format_metric(data["pages"]["suspicious_noise_rate"]["median"]),
            )
        )

    paired = summary["paired_comparison"]
    bootstrap = paired["bootstrap_median_difference"]
    test = paired["test"]
    estimate = bootstrap["estimate"]
    if estimate is None:
        direction = "没有足够的成对页面用于比较。"
    elif estimate > 0:
        direction = "差值为正，表示 DeepSeek-OCR-2 的整页标准化 CER 更低。"
    elif estimate < 0:
        direction = "差值为负，表示 Tesseract 的整页标准化 CER 更低。"
    else:
        direction = "两者整页标准化 CER 的中位差为 0。"

    ci = bootstrap["ci95"]
    lines.extend(
        [
            "",
            "## 配对统计",
            "",
            f"- 成对整页数：{paired['paired_pages']}",
            f"- DeepSeek-OCR-2 较优 / Tesseract 较优 / 持平："
            f"{paired['deepseek_better_pages']} / {paired['tesseract_better_pages']} / {paired['ties']}",
            f"- 标准化 CER 中位差（Tesseract - DeepSeek）：{format_metric(estimate)}",
            f"- Bootstrap 95% CI：[{format_metric(ci[0])}, {format_metric(ci[1])}]",
            f"- {test['method']}：p={format_metric(test['p_value'])}",
            f"- 方向解释：{direction}",
            "",
            "## 指标说明",
            "",
            "- 严格 CER 保留空格和换行，可观察版面还原差异。",
            "- 标准化 CER 只统一 Unicode、大小写和空白，不删除乱码或多余正文。",
            "- 局部 CER 会先在整页结果中寻找与标注段落最接近的连续片段，"
            "因此不要求 OCR 和人工标注拥有相同段落边界。",
            "- 可疑噪声率只检测明显重复、控制字符和长符号串，是辅助诊断项。",
            "- 显著性检验以页面为配对单位；最终模型选择还应结合 CER、遗漏、"
            "额外噪声、运行耗时及人工抽查共同判断。",
            "",
        ]
    )
    return "\n".join(lines)


PAGE_FIELDS = [
    "engine",
    "task_name",
    "page_number",
    "task_id",
    "reference_chars",
    "ocr_chars",
    "length_ratio",
    "strict_edit_distance",
    "strict_cer",
    "normalized_reference_chars",
    "normalized_ocr_chars",
    "normalized_edit_distance",
    "normalized_cer",
    "wer",
    "alignment_substitutions",
    "alignment_deletions",
    "alignment_insertions",
    "alignment_insertion_rate",
    "suspicious_noise_rate",
    "elapsed_seconds",
]

SNIPPET_FIELDS = [
    "engine",
    "snippet_id",
    "task_name",
    "page_number",
    "reference_chars",
    "matched_chars",
    "normalized_edit_distance",
    "normalized_cer",
    "wer",
    "alignment_substitutions",
    "alignment_deletions",
    "alignment_insertions",
    "normalized_start",
    "normalized_end",
    "raw_start",
    "raw_end",
]


def evaluate(args: argparse.Namespace) -> int:
    gold_pages = read_jsonl(project_path(args.gold_pages))
    gold_snippets = read_jsonl(project_path(args.gold_snippets))
    validate_gold(gold_pages, "整页标注")
    validate_gold(gold_snippets, "局部标注")

    result_paths = {
        "tesseract": project_path(args.tesseract_results),
        "deepseek_ocr2": project_path(args.deepseek_results),
    }
    result_indexes = {
        engine: index_ocr_results(read_jsonl(path), engine)
        for engine, path in result_paths.items()
    }
    expected_keys = {
        record_key(record) for record in [*gold_pages, *gold_snippets]
    }

    coverage: dict[str, Any] = {}
    missing_by_engine: dict[str, list[tuple[str, int]]] = {}
    for engine, index in result_indexes.items():
        successful = {
            key for key in expected_keys if successful_text(index.get(key)) is not None
        }
        missing = sorted(expected_keys - successful)
        missing_by_engine[engine] = missing
        coverage[engine] = {
            "expected_pages": len(expected_keys),
            "successful_pages": len(successful),
            "missing_or_failed_pages": len(missing),
        }

    if not args.allow_missing and any(missing_by_engine.values()):
        details = []
        for engine, missing in missing_by_engine.items():
            if missing:
                preview = ", ".join(f"{name}:p{page}" for name, page in missing[:5])
                details.append(f"{engine} 缺少 {len(missing)} 页（{preview}）")
        raise ValueError("；".join(details) + "。如需部分评估请使用 --allow-missing")

    page_rows: list[dict[str, Any]] = []
    snippet_rows: list[dict[str, Any]] = []
    snippet_audits: list[dict[str, Any]] = []
    for engine, index in result_indexes.items():
        for gold in gold_pages:
            ocr_record = index.get(record_key(gold))
            if successful_text(ocr_record) is not None:
                page_rows.append(page_score(engine, gold, ocr_record))
        for snippet_number, gold in enumerate(gold_snippets, start=1):
            ocr_record = index.get(record_key(gold))
            if successful_text(ocr_record) is None:
                continue
            row, audit = snippet_score(
                engine, f"S{snippet_number:04d}", gold, ocr_record
            )
            snippet_rows.append(row)
            snippet_audits.append(audit)

    summary = build_summary(
        page_rows,
        snippet_rows,
        coverage,
        args.bootstrap_samples,
        args.seed,
    )
    summary["inputs"] = {
        "gold_pages": str(project_path(args.gold_pages)),
        "gold_snippets": str(project_path(args.gold_snippets)),
        "tesseract_results": str(result_paths["tesseract"]),
        "deepseek_results": str(result_paths["deepseek_ocr2"]),
    }

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "page_scores": output_dir / "page_scores.csv",
        "snippet_scores": output_dir / "snippet_scores.csv",
        "snippet_matches": output_dir / "snippet_matches.jsonl",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
    }
    if not args.overwrite:
        existing = [path for path in output_files.values() if path.exists()]
        if existing:
            raise ValueError(
                "评估输出已存在，请更换目录或使用 --overwrite："
                + ", ".join(str(path) for path in existing)
            )

    write_csv(output_files["page_scores"], page_rows, PAGE_FIELDS)
    write_csv(output_files["snippet_scores"], snippet_rows, SNIPPET_FIELDS)
    write_jsonl(output_files["snippet_matches"], snippet_audits)
    write_json(output_files["summary"], summary)
    output_files["report"].write_text(render_report(summary), encoding="utf-8")

    print(f"page_scores={len(page_rows)}")
    print(f"snippet_scores={len(snippet_rows)}")
    print(f"output_dir={output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare fixed OCR evaluation pages and compare two OCR backends."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Select all pages referenced by the two gold JSONL files."
    )
    prepare_parser.add_argument("--gold-pages", type=Path, default=DEFAULT_GOLD_PAGES)
    prepare_parser.add_argument(
        "--gold-snippets", type=Path, default=DEFAULT_GOLD_SNIPPETS
    )
    prepare_parser.add_argument(
        "--master-tasks", type=Path, default=DEFAULT_MASTER_TASKS
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--overwrite", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate paired Tesseract and DeepSeek OCR outputs."
    )
    evaluate_parser.add_argument("--gold-pages", type=Path, default=DEFAULT_GOLD_PAGES)
    evaluate_parser.add_argument(
        "--gold-snippets", type=Path, default=DEFAULT_GOLD_SNIPPETS
    )
    evaluate_parser.add_argument("--tesseract-results", type=Path, required=True)
    evaluate_parser.add_argument("--deepseek-results", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluate_parser.add_argument("--seed", type=int, default=20260923)
    evaluate_parser.add_argument("--allow-missing", action="store_true")
    evaluate_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "bootstrap_samples", 1) < 1:
        parser.error("--bootstrap-samples 必须大于 0")
    try:
        if args.command == "prepare":
            return prepare_tasks(args)
        return evaluate(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
