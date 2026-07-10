"""
用途：对 RAG 文献和 chunk 做严格医学筛选，构建更可信的医学知识库。
输入：literature_registry.csv 与 rag_chunks.jsonl。
输出：strict/review/excluded 三套 registry、三套 chunk，以及 filter_report.md。
说明：strict 进入默认医学检索；review 留给人工复核；excluded 不进入严格索引。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# -----------------------------------------------------------------------------
# 规则词表
# -----------------------------------------------------------------------------
# 严格筛选的目标是构建 human breast / non-puerperal mastitis 相关知识库。
# 这里使用可审计的规则，而不是 LLM 判断：每个决策都能追溯到命中的关键词。

INCLUDE_TERMS: tuple[tuple[str, int], ...] = (
    ("idiopathic granulomatous mastitis", 6),
    ("granulomatous lobular mastitis", 6),
    ("granulomatous mastitis", 5),
    ("non-puerperal mastitis", 6),
    ("nonpuerperal mastitis", 6),
    ("non-lactational mastitis", 6),
    ("nonlactational mastitis", 6),
    ("periductal mastitis", 5),
    ("plasma cell mastitis", 5),
    ("mammary duct ectasia", 5),
    ("breast tuberculosis", 6),
    ("mammary tuberculosis", 6),
    ("tuberculous mastitis", 6),
    ("tubercular mastitis", 6),
    ("corynebacterium kroppenstedtii", 5),
    ("corynebacterium", 3),
    ("recurrent mastitis", 4),
    ("recurrent granulomatous mastitis", 6),
    ("refractory granulomatous mastitis", 6),
    ("relapsed granulomatous mastitis", 6),
    ("breast abscess", 3),
    ("subareolar abscess", 4),
    ("breast fistula", 4),
    ("mammary fistula", 4),
    ("zuska", 4),
)

TREATMENT_TERMS: tuple[tuple[str, int], ...] = (
    ("treatment", 2),
    ("therapy", 2),
    ("drug therapy", 3),
    ("medical treatment", 3),
    ("combination therapy", 3),
    ("combined drug therapy", 3),
    ("multidrug therapy", 3),
    ("triple therapy", 3),
    ("regimen", 2),
    ("rifampicin", 3),
    ("rifampin", 3),
    ("isoniazid", 3),
    ("ethambutol", 3),
    ("methotrexate", 3),
    ("corticosteroid", 3),
    ("prednisone", 3),
    ("prednisolone", 3),
    ("antibiotic therapy", 3),
    ("antibiotic", 2),
    ("antibiotics", 2),
    ("immunosuppressive", 2),
    ("recurrence", 2),
    ("relapse", 2),
    ("remission", 2),
    ("follow-up", 1),
    ("outcome", 1),
)

HUMAN_CONTEXT_TERMS: tuple[tuple[str, int], ...] = (
    ("patient", 2),
    ("patients", 2),
    ("woman", 2),
    ("women", 2),
    ("female", 1),
    ("clinical", 1),
    ("case report", 1),
    ("breast", 2),
    ("human", 2),
)

EXCLUDE_TERMS: tuple[tuple[str, int], ...] = (
    ("bovine", 8),
    ("dairy cow", 8),
    ("dairy cows", 8),
    ("cow", 7),
    ("cows", 7),
    ("cattle", 7),
    ("herd", 6),
    ("udder", 8),
    ("teat", 7),
    ("intramammary", 8),
    ("milk yield", 7),
    ("somatic cell count", 7),
    ("veterinary", 8),
    ("goat", 7),
    ("goats", 7),
    ("sheep", 7),
    ("camel", 7),
    ("camels", 7),
    ("buffalo", 7),
    ("buffaloes", 7),
    ("sow", 7),
    ("porcine", 7),
    ("mouse mastitis", 7),
    ("murine mastitis", 7),
    ("lactational mastitis", 6),
    ("breastfeeding", 6),
    ("postpartum", 5),
    ("puerperal mastitis", 5),
    ("breast cancer", 4),
    ("inflammatory breast cancer", 5),
    ("carcinoma", 4),
    ("tumor", 3),
    ("tumour", 3),
    ("cell line", 5),
    ("nanoparticle", 4),
    ("plant extract", 4),
)


STRICT_CHUNK_REVIEW_TERMS = {
    "bovine",
    "dairy cow",
    "dairy cows",
    "cow",
    "cows",
    "cattle",
    "herd",
    "udder",
    "teat",
    "milk yield",
    "somatic cell count",
    "veterinary",
    "goat",
    "goats",
    "sheep",
    "camel",
    "camels",
    "buffalo",
    "buffaloes",
    "sow",
    "porcine",
}

STRONG_TARGET_TERMS = {
    "idiopathic granulomatous mastitis",
    "granulomatous lobular mastitis",
    "granulomatous mastitis",
    "non-puerperal mastitis",
    "nonpuerperal mastitis",
    "non-lactational mastitis",
    "nonlactational mastitis",
    "periductal mastitis",
    "plasma cell mastitis",
    "mammary duct ectasia",
    "breast tuberculosis",
    "mammary tuberculosis",
    "tuberculous mastitis",
    "tubercular mastitis",
}


@dataclass(frozen=True)
class Decision:
    decision: str
    include_score: int
    exclude_score: int
    include_matches: list[str]
    exclude_matches: list[str]
    review_reason: str


# -----------------------------------------------------------------------------
# 文本规范化与关键词命中
# -----------------------------------------------------------------------------
# PubMed title/abstract、chunk title/section/text 来源不一，先统一大小写和空白。
# 关键词使用边界匹配，避免 cow 命中 scow 这类偶然子串。


def normalize_text(value: Any) -> str:
    text = " ".join(str(value or "").lower().split())
    return text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")


def contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    plural = "s?" if not term.endswith("s") else ""
    pattern = r"(?<![a-z0-9])" + escaped + plural + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def matched_terms(text: str, weighted_terms: Iterable[tuple[str, int]]) -> tuple[list[str], int]:
    matches: list[str] = []
    score = 0
    for term, weight in weighted_terms:
        if contains_term(text, term):
            matches.append(term)
            score += weight
    return matches, score


def record_text(record: dict[str, Any]) -> str:
    fields = [
        record.get("title", ""),
        record.get("abstract", ""),
        record.get("journal", ""),
        record.get("keywords", ""),
    ]
    return normalize_text(" ".join(str(field or "") for field in fields))


def chunk_text(chunk: dict[str, Any]) -> str:
    # chunk 级筛选刻意不使用 title。文章题名只说明来源文章相关，不能证明该段内容相关。
    fields = [chunk.get("section", ""), chunk.get("text", "")]
    return normalize_text(" ".join(str(field or "") for field in fields))


# -----------------------------------------------------------------------------
# 文章级严格筛选
# -----------------------------------------------------------------------------
# include/review/exclude 三分类：
# - include：强目标疾病 + 治疗/结局/人类临床语境足够明确。
# - exclude：动物、纯泌乳期、纯肿瘤或泛实验研究明显压过目标信号。
# - review：相关但不够确定，保留给人工复核，不进入 strict index。


def classify_text(text: str) -> Decision:
    include_matches, disease_score = matched_terms(text, INCLUDE_TERMS)
    treatment_matches, treatment_score = matched_terms(text, TREATMENT_TERMS)
    human_matches, human_score = matched_terms(text, HUMAN_CONTEXT_TERMS)
    exclude_matches, exclude_score = matched_terms(text, EXCLUDE_TERMS)

    include_score = disease_score + treatment_score + human_score
    all_include_matches = list(dict.fromkeys(include_matches + treatment_matches + human_matches))
    strong_target = any(term in STRONG_TARGET_TERMS for term in include_matches)
    has_treatment_signal = treatment_score >= 2
    has_human_signal = human_score >= 2
    animal_heavy = any(term in exclude_matches for term in [
        "bovine", "dairy cow", "dairy cows", "cow", "cows", "cattle", "herd",
        "udder", "teat", "intramammary", "veterinary", "goat", "goats",
        "sheep", "camel", "camels", "buffalo", "buffaloes", "sow", "porcine",
    ])

    if animal_heavy and include_score < 12:
        return Decision("exclude", include_score, exclude_score, all_include_matches, exclude_matches, "animal/veterinary mastitis signal dominates")
    if exclude_score >= 10 and include_score < exclude_score + 4:
        return Decision("exclude", include_score, exclude_score, all_include_matches, exclude_matches, "exclude terms dominate target evidence")
    if strong_target and has_treatment_signal and include_score >= exclude_score + 3:
        return Decision("include", include_score, exclude_score, all_include_matches, exclude_matches, "")
    if strong_target and has_human_signal and include_score >= exclude_score + 4:
        return Decision("include", include_score, exclude_score, all_include_matches, exclude_matches, "")
    if include_score >= 6 and exclude_score == 0:
        return Decision("review", include_score, exclude_score, all_include_matches, exclude_matches, "related but lacks strong disease-treatment evidence")
    if include_score > 0:
        return Decision("review", include_score, exclude_score, all_include_matches, exclude_matches, "mixed or weak target evidence")
    return Decision("exclude", include_score, exclude_score, all_include_matches, exclude_matches, "no target mastitis evidence")


def classify_record(record: dict[str, Any]) -> Decision:
    return classify_text(record_text(record))


def classify_chunk(chunk: dict[str, Any]) -> Decision:
    return classify_text(chunk_text(chunk))


# -----------------------------------------------------------------------------
# 文件读写与决策标注
# -----------------------------------------------------------------------------
# 输出不覆盖原始 broad corpus，而是写入 data/registry/filtered 与 strict chunk 文件。
# 每条记录都附带分数和命中词，方便医学人工抽查。


def read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_records(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_ids(record: dict[str, Any]) -> list[str]:
    ids = []
    for key in ["pmcid", "pmid", "doi"]:
        value = normalize_text(record.get(key, ""))
        if value:
            ids.append(value)
    return ids


def chunk_ids(chunk: dict[str, Any]) -> list[str]:
    ids = []
    for key in ["pmcid", "pmid", "doi"]:
        value = normalize_text(chunk.get(key, ""))
        if value:
            ids.append(value)
    return ids


def decision_fields(decision: Decision) -> dict[str, Any]:
    return {
        "filter_decision": decision.decision,
        "filter_include_score": decision.include_score,
        "filter_exclude_score": decision.exclude_score,
        "filter_include_matches": "; ".join(decision.include_matches),
        "filter_exclude_matches": "; ".join(decision.exclude_matches),
        "filter_review_reason": decision.review_reason,
    }


def annotate_record(record: dict[str, Any], decision: Decision) -> dict[str, Any]:
    annotated = dict(record)
    annotated.update(decision_fields(decision))
    return annotated


def annotate_chunk(chunk: dict[str, Any], decision: Decision, level: str) -> dict[str, Any]:
    annotated = dict(chunk)
    annotated.update(decision_fields(decision))
    annotated["filter_level"] = level
    return annotated


def split_registry(records: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Decision]]:
    included: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    decisions_by_id: dict[str, Decision] = {}

    for record in records:
        decision = classify_record(record)
        annotated = annotate_record(record, decision)
        if decision.decision == "include":
            included.append(annotated)
        elif decision.decision == "review":
            review.append(annotated)
        else:
            excluded.append(annotated)
        for item_id in record_ids(record):
            decisions_by_id[item_id] = decision
    return included, review, excluded, decisions_by_id


def article_decision_for_chunk(chunk: dict[str, Any], decisions_by_id: dict[str, Decision]) -> Decision | None:
    for item_id in chunk_ids(chunk):
        if item_id in decisions_by_id:
            return decisions_by_id[item_id]
    return None


def has_strict_chunk_noise(decision: Decision) -> bool:
    return any(term in STRICT_CHUNK_REVIEW_TERMS for term in decision.exclude_matches)


def filter_chunks(chunks: list[dict[str, Any]], decisions_by_id: dict[str, Decision]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for chunk in chunks:
        article_decision = article_decision_for_chunk(chunk, decisions_by_id)
        chunk_decision = classify_chunk(chunk)

        if article_decision and article_decision.decision == "exclude":
            excluded.append(annotate_chunk(chunk, article_decision, "article"))
            continue
        if article_decision and article_decision.decision == "include":
            if has_strict_chunk_noise(chunk_decision):
                review.append(annotate_chunk(chunk, chunk_decision, "chunk_noise_review"))
            elif chunk_decision.decision == "include":
                included.append(annotate_chunk(chunk, chunk_decision, "article_and_chunk"))
            else:
                review.append(annotate_chunk(chunk, chunk_decision, "chunk"))
            continue
        if chunk_decision.decision == "include":
            review.append(annotate_chunk(chunk, chunk_decision, "chunk_without_included_article"))
            continue
        if chunk_decision.decision == "review":
            review.append(annotate_chunk(chunk, chunk_decision, "chunk"))
            continue
        excluded.append(annotate_chunk(chunk, chunk_decision, "chunk"))

    return included, review, excluded


def write_report(
    path: Path,
    registry_counts: Counter[str],
    chunk_counts: Counter[str],
    registry_rows: int,
    chunk_rows: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Strict Corpus Filter Report",
        "",
        f"created_at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Registry",
        f"total_records: {registry_rows}",
        f"include: {registry_counts.get('include', 0)}",
        f"review: {registry_counts.get('review', 0)}",
        f"exclude: {registry_counts.get('exclude', 0)}",
        "",
        "## Chunks",
        f"total_chunks: {chunk_rows}",
        f"include: {chunk_counts.get('include', 0)}",
        f"review: {chunk_counts.get('review', 0)}",
        f"exclude: {chunk_counts.get('exclude', 0)}",
        "",
        "## Rule Notes",
        "- strict_include is intended for default medical RAG retrieval.",
        "- review keeps weak or mixed evidence for later manual audit.",
        "- exclude contains animal/veterinary, lactational-only, cancer-only, or non-target records.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------------------------
# 默认读取 broad corpus，输出 strict/review/exclude 三套文件。strict index 后续
# 由 build_strict_index.sh 基于 rag_chunks_strict.jsonl 单独生成。


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly filter RAG corpus for human non-puerperal mastitis medical evidence.")
    parser.add_argument("--registry-in", type=Path, default=Path("data/registry/processed/literature_registry.csv"))
    parser.add_argument("--chunks-in", type=Path, default=Path("data/articles/processed/rag_chunks.jsonl"))
    parser.add_argument("--registry-strict-out", type=Path, default=Path("data/registry/filtered/literature_registry_strict.csv"))
    parser.add_argument("--registry-review-out", type=Path, default=Path("data/registry/filtered/literature_registry_review.csv"))
    parser.add_argument("--registry-excluded-out", type=Path, default=Path("data/registry/filtered/literature_registry_excluded.csv"))
    parser.add_argument("--chunks-strict-out", type=Path, default=Path("data/articles/processed/rag_chunks_strict.jsonl"))
    parser.add_argument("--chunks-review-out", type=Path, default=Path("data/articles/processed/rag_chunks_review.jsonl"))
    parser.add_argument("--chunks-excluded-out", type=Path, default=Path("data/articles/processed/rag_chunks_excluded.jsonl"))
    parser.add_argument("--report-out", type=Path, default=Path("data/registry/filtered/filter_report.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.registry_in.exists():
        print(f"registry input not found: {args.registry_in}", file=sys.stderr)
        return 2
    if not args.chunks_in.exists():
        print(f"chunk input not found: {args.chunks_in}", file=sys.stderr)
        return 2

    registry_records = read_csv_records(args.registry_in)
    chunks = read_jsonl(args.chunks_in)

    registry_included, registry_review, registry_excluded, decisions_by_id = split_registry(registry_records)
    chunks_included, chunks_review, chunks_excluded = filter_chunks(chunks, decisions_by_id)

    base_fields = list(registry_records[0].keys()) if registry_records else []
    filter_fields = [
        "filter_decision",
        "filter_include_score",
        "filter_exclude_score",
        "filter_include_matches",
        "filter_exclude_matches",
        "filter_review_reason",
    ]
    write_csv_records(args.registry_strict_out, registry_included, base_fields + filter_fields)
    write_csv_records(args.registry_review_out, registry_review, base_fields + filter_fields)
    write_csv_records(args.registry_excluded_out, registry_excluded, base_fields + filter_fields)

    write_jsonl(args.chunks_strict_out, chunks_included)
    write_jsonl(args.chunks_review_out, chunks_review)
    write_jsonl(args.chunks_excluded_out, chunks_excluded)

    registry_counts = Counter({"include": len(registry_included), "review": len(registry_review), "exclude": len(registry_excluded)})
    chunk_counts = Counter({"include": len(chunks_included), "review": len(chunks_review), "exclude": len(chunks_excluded)})
    write_report(args.report_out, registry_counts, chunk_counts, len(registry_records), len(chunks))

    print(f"registry_include={len(registry_included)}")
    print(f"registry_review={len(registry_review)}")
    print(f"registry_exclude={len(registry_excluded)}")
    print(f"chunks_include={len(chunks_included)}")
    print(f"chunks_review={len(chunks_review)}")
    print(f"chunks_exclude={len(chunks_excluded)}")
    print(f"strict_chunks_out={args.chunks_strict_out}")
    print(f"report={args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
