"""清洗 RAG 检索证据中的 OCR 与版面技术噪声。

输入：step08_rag_answer.py 生成的一个或多个 ``*_evidence.json``，也可输入目录批量处理。
输出：每个输入对应 ``*_cleaned_evidence.json`` 和 ``*_cleaning_report.json``；输出目录根部
      另写 ``cleaning_summary.json``。原始 evidence JSON 不会被覆盖。
说明：本步骤只判断文本是否技术上可读，不判断证据与医学问题是否相关。后续证据准入和
      最终报告只能使用 ``cleaned_text``，不能继续使用原始 ``text`` 或调试 prompt。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_medical.terminology.dictionary import TermDictionary
from rag_medical.terminology.normalizer import (
    NormalizerSettings,
    load_normalizer_resources,
    normalize_text,
)


FULLWIDTH_LATIN_SEQUENCE = re.compile(
    r"(?:[Ａ-Ｚａ-ｚ]+[ \t　]+){2,}[Ａ-Ｚａ-ｚ]+"
)
ARTIFACT_GLYPH_RUN = re.compile(r"[■□◆◇●○▪▫▯▬═]{4,}")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
REFERENCE_TYPE_RE = re.compile(r"\[\s*[JDMCR]\s*\]", re.IGNORECASE)
REFERENCE_NUMBER_RE = re.compile(r"(?:\[|(?<!\d))\s*\d{1,3}\s*\]")
REFERENCE_YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*[,，]\s*\d{1,3}\s*[（(]\s*\d{1,2}")


# 这些模式只删除明确的出版属性，不触碰标题、摘要、方法、结果和剂量等正文内容。
INLINE_METADATA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "article_number_removed",
        re.compile(r"文章编号\s*[:：]?\s*[0-9０-９()（）./－—-]{6,}"),
    ),
    (
        "classification_number_removed",
        re.compile(r"中图分类号\s*[:：]?\s*[A-Za-zＡ-Ｚａ-ｚ0-9０-９.．/-]{2,}"),
    ),
    (
        "document_code_removed",
        re.compile(r"文献标识码\s*[:：]?\s*[A-Za-zＡ-Ｚａ-ｚ]"),
    ),
)


@dataclass(frozen=True)
class TextCleanResult:
    """单个文本字段的清洗结果；reject_reason 非空表示该文本技术上不可读。"""

    text: str
    actions: list[str]
    metrics: dict[str, float | int]
    reject_reason: str = ""
    repairs: list[dict[str, Any]] | None = None
    unresolved_terms: list[dict[str, Any]] | None = None


# -----------------------------------------------------------------------------
# OCR 拉丁字母修复
# -----------------------------------------------------------------------------


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.tokenize(text))


def should_join_fullwidth_fragments(fragments: list[str], tokenizer: Any) -> bool:
    """保守判断全角 OCR 片段是否原本属于同一个英文词。

    只处理至少三个全角片段，并要求出现 1~2 字母的异常短片段。至少一个片段还必须被
    Qwen tokenizer 拆成多个子词。拼接结果限制在 15 个字母以内，并且 token 数至少
    减少 2、降幅至少 25%。这样可以恢复
    ``pyrazi nami de``，同时避免把 ``in the study`` 这类正常短语直接连在一起。
    """

    if tokenizer is None or not 3 <= len(fragments) <= 8:
        return False
    if not all(fragment.isalpha() and 1 <= len(fragment) <= 8 for fragment in fragments):
        return False
    if not any(len(fragment) <= 2 for fragment in fragments):
        return False

    joined = "".join(fragments)
    # 较长候选通常包含多个真实单词；在没有医学词表的前提下不冒险自动拼接。
    if not 6 <= len(joined) <= 15:
        return False
    fragment_counts = [token_count(tokenizer, fragment) for fragment in fragments]
    if not any(count > 1 for count in fragment_counts):
        return False
    spaced_count = token_count(tokenizer, " ".join(fragments))
    joined_count = token_count(tokenizer, joined)
    reduction = spaced_count - joined_count
    return reduction >= 2 and reduction / max(spaced_count, 1) >= 0.25


def repair_fullwidth_latin(text: str, tokenizer: Any) -> tuple[str, list[dict[str, str]]]:
    """只在 NFKC 前修复全角 OCR 英文，普通半角英文不参与自动拼接。"""

    repairs: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        before = match.group(0)
        left = text[match.start() - 1] if match.start() else ""
        right = text[match.end()] if match.end() < len(text) else ""
        boundary_chars = "'’-－—"
        left_is_latin = bool(re.fullmatch(r"[A-Za-zＡ-Ｚａ-ｚ]", left))
        right_is_latin = bool(re.fullmatch(r"[A-Za-zＡ-Ｚａ-ｚ]", right))
        if (
            left_is_latin
            or right_is_latin
            or bool(left and left in boundary_chars)
            or bool(right and right in boundary_chars)
        ):
            return before
        normalized = unicodedata.normalize("NFKC", before)
        fragments = normalized.split()
        if not should_join_fullwidth_fragments(fragments, tokenizer):
            return before
        after = "".join(fragments)
        repairs.append({"before": before, "after": after})
        return after

    return FULLWIDTH_LATIN_SEQUENCE.sub(replace, text), repairs


# -----------------------------------------------------------------------------
# 确定性轻量清洗
# -----------------------------------------------------------------------------


def remove_line_boilerplate(text: str) -> tuple[str, bool]:
    """删除独占一行的页码和短出版页眉；连续正文中的数字不会被删除。"""

    kept_lines: list[str] = []
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        is_page_number = bool(re.fullmatch(r"(?:第\s*)?\d{1,4}\s*(?:页)?", stripped))
        is_short_journal_header = bool(
            len(stripped) <= 120
            and re.search(r"(?:19|20)\d{2}\s*年", stripped)
            and re.search(r"第\s*\d+\s*卷", stripped)
            and re.search(r"第\s*\d+\s*期", stripped)
        )
        if stripped and (is_page_number or is_short_journal_header):
            changed = True
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines), changed


def normalize_and_strip_artifacts(
    text: str,
    tokenizer: Any,
) -> tuple[str, list[str], list[dict[str, str]]]:
    actions: list[str] = []
    repaired, repairs = repair_fullwidth_latin(text, tokenizer)
    if repairs:
        actions.append("fragmented_fullwidth_latin_repaired")

    normalized = unicodedata.normalize("NFKC", repaired)
    if normalized != repaired:
        actions.append("unicode_nfkc_normalized")

    without_controls = "".join(
        char for char in normalized if char in "\n\t" or unicodedata.category(char) != "Cc"
    )
    if without_controls != normalized:
        actions.append("control_characters_removed")

    cleaned, line_changed = remove_line_boilerplate(without_controls)
    if line_changed:
        actions.append("line_boilerplate_removed")

    without_glyph_runs = ARTIFACT_GLYPH_RUN.sub(" ", cleaned)
    if without_glyph_runs != cleaned:
        actions.append("artifact_glyph_runs_removed")
    cleaned = without_glyph_runs

    for action, pattern in INLINE_METADATA_PATTERNS:
        updated = pattern.sub(" ", cleaned)
        if updated != cleaned:
            actions.append(action)
            cleaned = updated

    # 保留换行边界供审计，但只合并行内多余空格；绝不全局删除单词或剂量内部空格。
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    cleaned = "\n".join(line for line in lines if line).strip()
    return cleaned, list(dict.fromkeys(actions)), repairs


# -----------------------------------------------------------------------------
# 高置信度技术拒绝
# -----------------------------------------------------------------------------


def repetition_coverage(text: str, window_size: int = 40, step: int = 8) -> float:
    """估计重复乱码覆盖率，兼顾相同短片段循环和低字符多样性的中文乱码。"""

    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    covered = [False] * len(compact)

    # 捕捉 ``邓马邓马...`` 等严格循环；模式限制为 1~4 字符且至少重复四次。
    for pattern_size in range(1, 5):
        pattern = re.compile(rf"(.{{{pattern_size}}})\1{{3,}}")
        for match in pattern.finditer(compact):
            covered[match.start() : match.end()] = [True] * (match.end() - match.start())

    # OCR 乱码未必严格循环，因此再用固定窗口检测“几乎只由少数汉字组成”的区域。
    if len(compact) >= window_size:
        for start in range(0, len(compact) - window_size + 1, step):
            window = compact[start : start + window_size]
            cjk_chars = CJK_RE.findall(window)
            if len(cjk_chars) / window_size < 0.85:
                continue
            counts = Counter(cjk_chars)
            top_five_ratio = sum(count for _, count in counts.most_common(5)) / len(cjk_chars)
            if len(counts) <= 8 and top_five_ratio >= 0.75:
                covered[start : start + window_size] = [True] * window_size

    return sum(covered) / len(compact)


def quality_metrics(text: str) -> dict[str, float | int]:
    compact = re.sub(r"\s+", "", text)
    length = len(compact)
    meaningful_count = sum(
        1 for char in compact if CJK_RE.fullmatch(char) or char.isascii() and char.isalnum()
    )
    symbol_count = sum(
        1
        for char in compact
        if not CJK_RE.fullmatch(char) and not (char.isascii() and char.isalnum())
    )
    latin_words = LATIN_WORD_RE.findall(text)
    short_latin_count = sum(len(word) <= 2 for word in latin_words)
    reference_type_count = len(REFERENCE_TYPE_RE.findall(text))
    reference_number_count = len(REFERENCE_NUMBER_RE.findall(text))
    reference_year_count = len(REFERENCE_YEAR_RE.findall(text))
    denominator = max(length, 1)
    return {
        "character_count": length,
        "meaningful_ratio": round(meaningful_count / denominator, 4),
        "symbol_ratio": round(symbol_count / denominator, 4),
        "repetition_coverage": round(repetition_coverage(text), 4),
        "latin_word_count": len(latin_words),
        "short_latin_ratio": round(short_latin_count / max(len(latin_words), 1), 4),
        "cjk_character_count": len(CJK_RE.findall(text)),
        "reference_type_count": reference_type_count,
        "reference_number_count": reference_number_count,
        "reference_year_count": reference_year_count,
    }


def technical_reject_reason(metrics: dict[str, float | int]) -> str:
    length = int(metrics["character_count"])
    if length == 0:
        return "empty_after_cleaning"
    if length >= 80 and float(metrics["repetition_coverage"]) >= 0.60:
        return "high_repetition_noise"
    if (
        length >= 80
        and float(metrics["symbol_ratio"]) >= 0.60
        and float(metrics["meaningful_ratio"]) <= 0.25
    ):
        return "symbol_dominant_noise"
    if (
        int(metrics["latin_word_count"]) >= 10
        and float(metrics["short_latin_ratio"]) >= 0.80
        and int(metrics["cjk_character_count"]) <= 5
    ):
        return "fragmented_latin_noise"
    if (
        length >= 200
        and int(metrics["reference_type_count"]) >= 4
        and int(metrics["reference_number_count"]) >= 4
        and int(metrics["reference_year_count"]) >= 3
    ):
        return "reference_list_dominant"
    return ""


def clean_text(
    text: str,
    tokenizer: Any = None,
    *,
    terminology_dictionary: TermDictionary | None = None,
    terminology_settings: NormalizerSettings | None = None,
) -> TextCleanResult:
    cleaned, actions, repairs = normalize_and_strip_artifacts(str(text or ""), tokenizer)

    # 词典规范化器只处理医学术语的断裂、黏连和高置信拼写误差；它不参与技术拒绝。
    unresolved_terms: list[dict[str, Any]] = []
    if terminology_dictionary is not None:
        terminology_result = normalize_text(
            cleaned,
            dictionary=terminology_dictionary,
            settings=terminology_settings,
        )
        cleaned = terminology_result.normalized_text
        actions.extend(terminology_result.actions)
        repairs.extend(
            {
                "before": item.before,
                "after": item.after,
                "match_type": item.match_type,
                "concept_id": item.concept_id,
                "category": item.category,
            }
            for item in terminology_result.replacements
        )
        unresolved_terms = [
            {
                "text": item.text,
                "reason": item.reason,
                "candidates": list(item.candidates),
                "candidate_scores": list(item.candidate_scores),
            }
            for item in terminology_result.unresolved_spans
        ]

    metrics = quality_metrics(cleaned)
    reason = technical_reject_reason(metrics)
    return TextCleanResult(
        text=cleaned,
        actions=list(dict.fromkeys(actions)),
        metrics=metrics,
        reject_reason=reason,
        repairs=repairs,
        unresolved_terms=unresolved_terms,
    )


# -----------------------------------------------------------------------------
# Evidence JSON 处理与输出
# -----------------------------------------------------------------------------


def clean_evidence_payload(
    payload: dict[str, Any],
    tokenizer: Any = None,
    source_path: str = "",
    *,
    terminology_dictionary: TermDictionary | None = None,
    terminology_settings: NormalizerSettings | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_records = payload.get("evidence")
    if not isinstance(evidence_records, list):
        raise ValueError("input JSON must contain an 'evidence' list")

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for record in evidence_records:
        evidence_id = str(record.get("evidence_id") or "")
        matched_source = str(record.get("matched_text") or record.get("text") or "")
        context_source = str(record.get("text") or matched_source)
        matched_result = clean_text(
            matched_source,
            tokenizer,
            terminology_dictionary=terminology_dictionary,
            terminology_settings=terminology_settings,
        )
        context_result = clean_text(
            context_source,
            tokenizer,
            terminology_dictionary=terminology_dictionary,
            terminology_settings=terminology_settings,
        )

        if context_result.reject_reason:
            status_counts["technical_reject"] += 1
            rejected.append(
                {
                    "evidence_id": evidence_id,
                    "reason": context_result.reject_reason,
                    "title": record.get("title", ""),
                    "chunk_id": record.get("chunk_id", ""),
                    "metrics": context_result.metrics,
                }
            )
            continue

        cleaned_record = dict(record)
        cleaned_record["cleaned_matched_text"] = (
            "" if matched_result.reject_reason else matched_result.text
        )
        cleaned_record["cleaned_text"] = context_result.text
        changed = (
            matched_result.text != matched_source
            or context_result.text != context_source
            or bool(matched_result.reject_reason)
        )
        status = "cleaned" if changed else "unchanged"
        actions = list(dict.fromkeys(matched_result.actions + context_result.actions))
        if matched_result.reject_reason:
            actions.append("matched_text_technical_reject")
        cleaned_record["cleaning_status"] = status
        cleaned_record["cleaning_actions"] = actions
        cleaned_record["cleaning_metrics"] = context_result.metrics
        if matched_result.repairs or context_result.repairs:
            cleaned_record["cleaning_repairs"] = {
                "matched_text": matched_result.repairs or [],
                "text": context_result.repairs or [],
            }
        if matched_result.unresolved_terms or context_result.unresolved_terms:
            # 未决候选仅供离线词典维护，绝不会在当前问答中自动替换或晋升。
            cleaned_record["terminology_unresolved"] = {
                "matched_text": matched_result.unresolved_terms or [],
                "text": context_result.unresolved_terms or [],
            }
        kept.append(cleaned_record)
        status_counts[status] += 1
        action_counts.update(actions)

    cleaned_at = datetime.now(timezone.utc).isoformat()
    cleaned_payload = dict(payload)
    cleaned_payload.update(
        {
            "cleaned_at": cleaned_at,
            "source_evidence_path": source_path,
            "original_evidence_count": len(evidence_records),
            "evidence_count": len(kept),
            "technical_reject_count": len(rejected),
            "evidence": kept,
        }
    )
    report = {
        "cleaned_at": cleaned_at,
        "source_evidence_path": source_path,
        "original_evidence_count": len(evidence_records),
        "kept_evidence_count": len(kept),
        "technical_reject_count": len(rejected),
        "status_counts": dict(status_counts),
        "action_counts": dict(action_counts),
        "rejected_evidence": rejected,
    }
    return cleaned_payload, report


def output_paths(input_path: Path, output_dir: Path, relative_parent: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    if stem.endswith("_evidence"):
        stem = stem[: -len("_evidence")]
    target_dir = output_dir / relative_parent
    return (
        target_dir / f"{stem}_cleaned_evidence.json",
        target_dir / f"{stem}_cleaning_report.json",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def discover_inputs(paths: list[Path]) -> list[tuple[Path, Path]]:
    discovered: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_file():
            candidates = [(path, Path())]
        elif path.is_dir():
            candidates = [
                (candidate, candidate.parent.relative_to(path))
                for candidate in sorted(path.rglob("*_evidence.json"))
                if not candidate.name.endswith("_cleaned_evidence.json")
            ]
        else:
            raise FileNotFoundError(f"input not found: {path}")
        for candidate, relative_parent in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                discovered.append((candidate, relative_parent))
                seen.add(resolved)
    return discovered


def load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean OCR and layout noise from RAG evidence JSON.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Evidence JSON files or directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("models/llm/qwen3-8b"),
        help="Local Qwen tokenizer used for conservative OCR word repair; model weights are not loaded.",
    )
    parser.add_argument(
        "--terminology-config",
        type=Path,
        default=Path("configs/terminology.yaml"),
        help="Read-only terminology dictionary and normalization settings.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.tokenizer_path.exists():
        print(f"tokenizer path not found: {args.tokenizer_path}", file=sys.stderr)
        return 2

    try:
        inputs = discover_inputs(args.inputs)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not inputs:
        print("no *_evidence.json inputs found", file=sys.stderr)
        return 2

    tokenizer = load_tokenizer(args.tokenizer_path)
    try:
        terminology_dictionary, terminology_settings, _ = load_normalizer_resources(
            args.terminology_config
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"failed to load terminology resources: {exc}", file=sys.stderr)
        return 2
    summaries: list[dict[str, Any]] = []
    for input_path, relative_parent in inputs:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        cleaned_payload, report = clean_evidence_payload(
            payload,
            tokenizer,
            str(input_path),
            terminology_dictionary=terminology_dictionary,
            terminology_settings=terminology_settings,
        )
        cleaned_path, report_path = output_paths(input_path, args.output_dir, relative_parent)
        write_json(cleaned_path, cleaned_payload)
        write_json(report_path, report)
        summaries.append(
            {
                "input": str(input_path),
                "cleaned_evidence": str(cleaned_path),
                "cleaning_report": str(report_path),
                "kept": report["kept_evidence_count"],
                "technical_reject": report["technical_reject_count"],
            }
        )
        print(
            f"cleaned={input_path} kept={report['kept_evidence_count']} "
            f"technical_reject={report['technical_reject_count']}"
        )

    batch_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file_count": len(inputs),
        "files": summaries,
    }
    summary_path = args.output_dir / "cleaning_summary.json"
    write_json(summary_path, batch_summary)
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
